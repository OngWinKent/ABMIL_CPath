import time
from functools import partial
from contextlib import suppress
import wandb
from timm.models import model_parameters
from timm.utils import AverageMeter, dispatch_clip_grad
from collections import OrderedDict
from torch.profiler import profile,  ProfilerActivity

from utils import *
from .metrics import get_metric_val
from tqdm import tqdm

class BaseTrainer():
    def __init__(self, engine, args, **kwargs):
        self.engine = engine

    """Train main function"""
    def train(self, model, loader, optimizer, device, amp_autocast, criterion, loss_scaler, scheduler, epoch, train_config):
        start = time.time()
        model.train()

        train_loss_log = 0.0
        accum_steps = train_config.get('accumulation_steps', 1)
        log_iter = train_config.get('log_iter', 1)
        step_count = 0  # Manual counter replace enumerate
        
        self.engine.init_func_train()
        optimizer.zero_grad()

        pbar = tqdm(loader, desc= f'Training epoch {epoch}')
        for batch in pbar:
            step_count += 1
            need_grad_update = (step_count % accum_steps == 0) or (step_count == len(loader))
            need_log_update = (step_count % log_iter == 0) or (step_count == len(loader))

            bag = batch['input']
            label = batch['target']
            batch_size = label.size(0)
            pos = batch.get('pos', None)  
            feat = batch.get('feat', None)

            self.engine.after_get_data_func()

            # Forward pass inside AMP context
            with amp_autocast(device_type=device.type, dtype=torch.float16, enabled=train_config.get('amp')):
                logits, labels, aux_loss, patch_num, keep_num, pad_ratio = self.engine.forward_func(
                    model=model,bag=bag,label=label,criterion=criterion,batch_size=batch_size,epoch=epoch,pos=pos,feat=feat,mil_name=train_config.get('mil_name'))

                if logits is None:
                    continue
                
                bs = logits.size(0)
                logit_loss = criterion(logits.view(bs, -1), labels.view(bs))
                loss = train_config.get('main_alpha') * logit_loss + aux_loss * train_config.get('aux_alpha')
                loss = loss / accum_steps

            # Backward pass
            loss_scaler.scale(loss).backward()
            if need_grad_update:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            self.engine.after_backward_func()
            train_loss_log += loss.item()
            if need_grad_update:
                optimizer.zero_grad()
                if train_config.get('lr_supi') and scheduler is not None:
                    scheduler.step()

            # Log
            if need_log_update:
                # average loss over iteration
                pbar.set_postfix(train_loss= f"{round(train_loss_log/step_count, 4)}")

        self.engine.final_train_func()

        end = time.time()
        train_interval = round(end - start, 2)
        train_loss_log = train_loss_log / len(loader)

        if not train_config.get('lr_supi') and scheduler is not None:
            scheduler.step(epoch + 1)

        return train_loss_log, train_interval

    """Validation main function"""
    def validate(self, args, model, loader, device, criterion, amp_autocast, eval_config, early_stopping=None, epoch=None, status="val"):
        model.eval()
        # Get eval config params
        n_classes = eval_config.get('n_classes')
        seed = eval_config.get('seed') 
        num_bootstrap = eval_config.get('num_bootstrap')
        best_metric_index = eval_config.get('best_metric_index')
        amp = eval_config.get('amp')
        mil_name = eval_config.get('mil_name')

        loss_cls_meter = AverageMeter()
        bag_logit, bag_labels = None, None
        bag_logit_sub = None

        self.engine.init_func_val()

        eval_start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc= 'Validating' if epoch is None else f'Validating epoch {epoch}'):
                bag = batch['input']
                label = batch['target']
                bag_labels = torch.cat((bag_labels, label)) if bag_labels is not None else torch.clone(label)

                self.engine.after_get_data_func()

                with amp_autocast(device_type=device.type, dtype=torch.float16, enabled= amp):
                    logits, labels = self.engine.validate_func(model=model, bag=bag, label=label, mil_name= mil_name)
                    if logits is None:
                        continue

                    if type(logits) in (list, tuple):
                        logits, logits_sub = logits
                    else:
                        logits_sub = None

                    bs = logits.size(0)

                    bag_logit = torch.cat((bag_logit, logits)) if bag_logit is not None else logits.clone()
                    if logits_sub is not None:
                        bag_logit_sub = torch.cat((bag_logit_sub, logits_sub)) if bag_logit_sub is not None else logits_sub.clone()
                    test_loss = criterion(logits.view(bs, -1), labels.view(bs))

                loss_cls_meter.update(test_loss, 1)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

        # Compute evaluation runtime
        eval_end_time = time.time()
        eval_interval = round(eval_end_time - eval_start_time, 2)

        output = get_metric_val(args,bag_logit=bag_logit, bag_labels=bag_labels, model=model, status=status, early_stopping=early_stopping, epoch=epoch, loss_cls_meter=loss_cls_meter, device=device, n_classes=n_classes , seed=seed , num_bootstrap=num_bootstrap, best_metric_index=best_metric_index)
        output = list(output)

        if bag_logit_sub is not None:
            output_sub = get_metric_val(args,bag_logit=bag_logit_sub, bag_labels=bag_labels, model=model, status=status, early_stopping=None, epoch=epoch, loss_cls_meter=loss_cls_meter, device=device, n_classes=n_classes , seed=seed, num_bootstrap=num_bootstrap, best_metric_index=best_metric_index)
            output_sub = list(output_sub)
            output[-1].update(output_sub[-1])
            output[0] = [output[0], output_sub[0]]

        return output, eval_interval