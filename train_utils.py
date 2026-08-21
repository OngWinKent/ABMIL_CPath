import torch
import torch.nn as nn
from timm.scheduler import create_scheduler_v2

from utils import *

############# Survival Prediction ###################
def nll_loss(hazards,S,Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1)  # censorship status, 0 or 1
    # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, Y, c, logits=None,hazards=None, S=None,alpha=None):
        if alpha is None:
            alpha = self.alpha
        if hazards is None:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
        return nll_loss(hazards, S, Y, c, alpha=alpha)

"""Training modules configuration"""
def build_train(args, model):
    # Loss function 
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "nll_surv":
        criterion = NLLSurvLoss(alpha=0.0)
    else:
        raise NotImplementedError

    _model = model
    lr = args.lr
    lr_enc = lr

    if args.image_input:
        general_param_names = ('feature', 'norm', 'norm1', 'classifier')
        general_params = [p for name, p in _model.mil.named_parameters() if p.requires_grad and any(name.startswith(sp) for sp in general_param_names)]
        other_params = [p for name, p in _model.mil.named_parameters() if p.requires_grad and not any(name.startswith(sp) for sp in general_param_names)]

        if args.warmup_epochs > 0:
            params= [
                {'params': filter(lambda p: p.requires_grad, _model.encoder.parameters()), 'lr': lr_enc,'weight_decay': args.weight_decay},
                # {'params': filter(lambda p: p.requires_grad, model.mil.parameters()), 'lr': lr, 'weight_decay': args.weight_decay},
                {'params': general_params, 'lr': lr, 'weight_decay': args.weight_decay},
                {'params': other_params, 'lr': lr, 'weight_decay': args.weight_decay},
            ]
            # print(list(params[0]['params']))
        else:
            params = [
            {'params': filter(lambda p: p.requires_grad, _model.encoder.parameters()), 'lr': lr_enc,'weight_decay': args.weight_decay},
            # {'params': filter(lambda p: p.requires_grad, model.mil.parameters()), 'lr': lr, 'weight_decay': args.weight_decay}
            {'params': general_params, 'lr': lr, 'weight_decay': args.weight_decay},
            {'params': other_params, 'lr': lr, 'weight_decay': args.weight_decay},
            ]
    else:
        params = [
            {'params': filter(lambda p: p.requires_grad, _model.parameters()), 'lr': lr,'weight_decay': args.weight_decay},]

    # Optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params)
    elif args.opt == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params)
    else:
        raise NotImplementedError

    # Scheduler
    if args.lr_sche == 'cosine':
        scheduler,_ = create_scheduler_v2(optimizer,sched='cosine',num_epochs=args.num_epoch,warmup_lr=args.warmup_lr,warmup_epochs=args.warmup_epochs,min_lr=1e-7)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None
    else:
        raise NotImplementedError

    # Early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(patience= args.patient, stop_epoch=args.max_epoch)
    else:
        early_stopping = None
    
    return criterion, optimizer, scheduler, early_stopping

"""Load pretrained model"""
def load_pretrained(model, device: str, pretrained_path: str, best_pt_path: str, last_pt_path: str):
    if pretrained_path:
        pretrained_pt_path = pretrained_path
    else:
        if os.path.exists(best_pt_path): # Prioritize
            pretrained_pt_path = best_pt_path
        else:
            pretrained_pt_path = last_pt_path

    if os.path.exists(pretrained_pt_path) and pretrained_pt_path.endswith('.pt'):
        pretrained_pt = torch.load(pretrained_pt_path, map_location= device, weights_only=True)
        model.load_state_dict(pretrained_pt.get('model'))
        print(f"[Pretrained Model] Loaded pretrained model weights saved at epoch {pretrained_pt.get('epoch')} from {pretrained_pt_path} for training")
    else:
        print(f"[Pretrained Model] No pretrained model weights loaded")

    return model