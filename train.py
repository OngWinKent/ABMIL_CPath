import warnings
warnings.filterwarnings("ignore")
import time
import torch
import os
import random
import traceback
import gc
from tqdm import tqdm

from datasets import build_dataloader
import utils
from modules import build_model
from train_utils import build_train, load_pretrained
from engines import build_engine
from options import get_parse_args
from typing import *

"""
Train and evaluate model 
Args:
    args: Command line arguments
    device: PyTorch device object
    dataset: Dictionary containing train/val/test datasets
"""
def main(args) -> None:
    # Set random seed 
    utils.seed_torch(args.seed)
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize metrics storage 
    acs, pre, rec,fs,auc,ck,acs_m,te_auc,te_fs=[],[],[],[],[],[],[],[],[]
    acs_std,auc_std,fs_std,ck_std,acs_m_std = [],[],[],[],[]
    ckc_metric = [auc,acs, pre, rec,fs,ck,acs_m,acs_std,auc_std,fs_std,ck_std,acs_m_std]
    te_ckc_metric = [te_auc,te_fs]
    
    loss_scaler = torch.amp.GradScaler(device=args.device, enabled=not args.amp_unscale)
    amp_autocast = torch.autocast

    # Initialize data loader
    train_loader, val_loader, test_loader = build_dataloader(args= args, image_input= args.image_input)
    print(f"[Dataloader] Train: {len(train_loader)} Val: {len(val_loader)} Test: {len(test_loader)}")

    # Create and define model save directory
    output_base_dir = os.path.join(args.output_path, args.title)
    os.makedirs(output_base_dir, exist_ok=True)
    best_pt_path = os.path.join(output_base_dir, 'model_best.pt')
    last_pt_path = os.path.join(output_base_dir, 'model_last.pt')
    # Create model with encoder + mil / mil only
    model = build_model(
        args= args, device= args.device, enc_name= args.enc_name, mil_name= args.mil_name, hf_token= args.hf_token, image_input= args.image_input)

    # Load pretrained model if available
    model = load_pretrained(
        model= model, device= args.device, pretrained_path= args.pretrained_path, best_pt_path= best_pt_path, last_pt_path= last_pt_path)

    # Initialize criterion,optimizer,scheduler,early-stopping
    criterion, optimizer, scheduler, early_stopping = build_train(args= args, model= model)

    # Init metrics
    best_ckc_metric = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_te = [0. for i in range(len(te_ckc_metric))]
    best_rowd = None

    # Training configuration
    train_config = {'accumulation_steps': args.accumulation_steps,'amp': args.amp,'log_iter': args.log_iter,'main_alpha': args.main_alpha,'aux_alpha': args.aux_alpha,'lr_supi': args.lr_supi, 'mil_name': args.mil_name}
    eval_config = {'amp': args.amp, 'n_classes': args.n_classes, 'seed': args.seed, 'num_bootstrap': args.num_bootstrap, 'mil_name': args.mil_name, 'best_metric_index': args.best_metric_index}

    # Build train and validation engine
    run_train, run_validate = build_engine(args, args.device)

    # Main training loop
    try:
        for epoch in range(1, args.num_epoch+1):
            torch.cuda.empty_cache()
            gc.collect()

            # Training
            train_loss, train_interval = run_train(
                model= model,loader= train_loader,optimizer= optimizer,device= args.device,amp_autocast= amp_autocast,criterion= criterion,loss_scaler= loss_scaler,scheduler= scheduler,epoch= epoch, train_config= train_config)

            # Validation
            (_metric_val, early_stop,test_loss, threshold_optimal, rowd_val), eval_interval = run_validate(
                args, model= model,loader= val_loader,device= args.device,criterion= criterion,amp_autocast= amp_autocast,early_stopping= early_stopping, epoch= epoch, eval_config= eval_config)

            # Console logging
            epoch_interval = train_interval + eval_interval
            remain_duration = epoch_interval * (args.num_epoch-epoch)
            hrs = int(remain_duration//3600)
            mins = int((remain_duration % 3600) // 60)
            print('[Train Log] Epoch [%d/%d] train loss: %.4f, val loss: %.4f, accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f , time: %.2f s, approx completed in %d hrs %d mins' % (epoch, args.num_epoch, train_loss, test_loss, _metric_val[1], _metric_val[0], _metric_val[2], _metric_val[3], _metric_val[4], epoch_interval, hrs, mins))
            rowd_val['epoch'] = epoch

            # Update best metric
            if best_rowd is None:
                best_rowd = rowd_val
            else:
                best_rowd = utils.update_best_metric(best_metric= best_rowd, val_metric= rowd_val)

            # Save the best model in the val_set
            if _metric_val[args.best_metric_index] > best_ckc_metric[args.best_metric_index]:
                best_ckc_metric = _metric_val+[epoch]
                best_rowd['epoch'] = epoch
                best_pt = {'model': model.state_dict(),'teacher': None,'epoch': epoch}
                torch.save(best_pt, best_pt_path)
                print(f"[Checkpoint] Best model saved at: {best_pt_path}")

            # Save the latest trained model
            utils.save_cpk(model, random, scheduler, optimizer, epoch, early_stopping, _metric_val, best_ckc_metric, best_ckc_metric_te, last_pt_path)

            # Early stop break
            if early_stop:
                print(f"[Early Stop] Early stopped at epoch {epoch}")
                break

    except KeyboardInterrupt:
        pass

    # Load the saved trained best model
    if os.path.exists(best_pt_path):
        loaded_best_pt = torch.load(best_pt_path, map_location=args.device.type, weights_only=True) # map_location= "cpu"
        model.load_state_dict(loaded_best_pt.get('model'))
        print(f"Loaded best model saved at epoch {loaded_best_pt.get('epoch')} from {best_pt_path} for test dataset evaluation")
        # Run test accuracy on the best model
        (metric_test,test_loss_log,rowd), test_interval = run_validate(
            args, model=model,loader=test_loader,device= args.device,criterion=criterion,amp_autocast=amp_autocast,status='test', eval_config= eval_config)
        print(f"[Test] Test process took {test_interval} seconds")
        print("[Test] Test dataset performance - accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f" % (metric_test[1], metric_test[0], metric_test[2], metric_test[3], metric_test[4]))
        # update metric 
        [ckc_metric[i].append(metric_test[i]) for i,_ in enumerate(ckc_metric)]

if __name__ == '__main__':
    # Get parsed arguments
    args = get_parse_args()
    # Cuda configurations
    utils.config_cuda(no_determ= args.no_determ, no_deter_algo= args.no_deter_algo)
    print(f"[Local Time] Training starts at {time.asctime( time.localtime(time.time()))}")
    # Run main function
    try:
        main(args=args)
    except:
        traceback.print_exc()
    finally:
        pass
    print(f"[Local Time] Training ends at {time.asctime( time.localtime(time.time()))}")