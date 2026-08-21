import warnings
warnings.filterwarnings("ignore")
import time
import torch
import os
import numpy as np
import random
import traceback
from timm.utils import AverageMeter
import gc

from datasets import build_dataloader
import utils
from modules import build_model
from train_utils import build_train
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

    amp_autocast = torch.autocast

    # Initialize data loader
    #train_loader, val_loader, test_loader = build_img_dataloader(args= args)
    train_loader, val_loader, test_loader = build_dataloader(args= args, image_input= args.image_input)
    print(f"[Dataloader] Train: {len(train_loader)} Val: {len(val_loader)} Test: {len(test_loader)}")

    # Create and define model save directory
    output_base_dir = os.path.join(args.output_path, args.title)
    best_pt_path = os.path.join(output_base_dir, 'model_best.pt')
    # Create model (encoder + mil)
    model= build_model(args= args, device= args.device, enc_name= args.enc_name, mil_name= args.mil_name, hf_token= args.hf_token, image_input= args.image_input)

    # Eval configuration
    eval_config = {'amp': args.amp, 'n_classes': args.n_classes, 'seed': args.seed, 'num_bootstrap': args.num_bootstrap, 'mil_name': args.mil_name, 'best_metric_index': args.best_metric_index}

    # Initialize criterion,optimizer,scheduler,early-stopping
    criterion, optimizer, scheduler, early_stopping = build_train(args= args, model= model)

    # Build train and validation engine
    run_train, run_validate= build_engine(args= args, device= args.device)

    if args.pretrained_path:
        pretrained_pt_path = args.pretrained_path
    else:
        pretrained_pt_path = best_pt_path
    # Check model path existence
    if not os.path.exists(pretrained_pt_path) or not pretrained_pt_path.endswith(".pt"):
        raise Exception(f"[Pretrained Error] Pretrained model path {pretrained_pt_path} does not exist for validation")
    # Load the saved trained best model
    pretrained_pt = torch.load(pretrained_pt_path, map_location= args.device, weights_only=True)
    model.load_state_dict(pretrained_pt.get('model'))
    print(f"[Model] Loaded best model saved at epoch {pretrained_pt.get('epoch')} from {pretrained_pt_path} for test dataset evaluation")
    # Run test accuracy on the best model
    (metric_test,test_loss_log,rowd), test_interval = run_validate(args=args, model=model,loader=test_loader,device= args.device,criterion=criterion,amp_autocast=amp_autocast,status='test',eval_config= eval_config)
    print(f"[Test] Test process took {test_interval} seconds")
    print("[Test] Test dataset performance - accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f" % (metric_test[1], metric_test[0], metric_test[2], metric_test[3], metric_test[4]))

if __name__ == '__main__':
    # Get parsed arguments
    args = get_parse_args()
    # Cuda configurations
    utils.config_cuda(no_determ= args.no_determ, no_deter_algo= args.no_deter_algo)
    print(f"[Local Time] {time.asctime( time.localtime(time.time()))}")
    # Run main function
    try:
        main(args=args)
    except:
        traceback.print_exc()
    finally:
        pass