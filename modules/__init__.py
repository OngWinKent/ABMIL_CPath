import torch
import timm

from .abmil import DAttention,AttentionGated
from .abmilx import DAttentionX
from .clam import CLAM_MB,CLAM_SB
from .dsmil import MILNet
from .transmil import TransMIL
from .mean_max import MeanMIL,MaxMIL
from .dtfd import DTFD
from .rrt import RRTMIL
from .e2e import E2E
from .encoders import ResNetEncoder
from .e2e_pooling import MeanPooling
from .vit_mil import ViTMIL
from .gigap import GIGAPMIL
from .chief import CHIEF,ConvStem
from wikg import WiKG
from .utils import get_mil_model_params

import os 
from utils import ModelEmaV3

from huggingface_hub import hf_hub_download

def load_enc_ckp(args,enc,enc_init_path):
    try:
        enc_ckp = torch.load(enc_init_path,weights_only=True,map_location='cpu')
    except:
        enc_ckp = torch.load(enc_init_path,map_location='cpu')
    
    if 'state_dict' in enc_ckp:
        enc_ckp = enc_ckp['state_dict']

    if 'model' in enc_ckp:
        enc_ckp = enc_ckp['model']
    new_state_dict = {}
    for key in enc_ckp:
        # 'classifier.0.weight' modify as 'classifier.weight'
        if 'encoder.' in key:
            #if not '.num_batches_tracked' in key:
            new_key = key.replace('encoder.', '')
            if 'module.' in new_key:
                new_key = new_key.replace('module.', '')
        # torchvision ckp
        elif 'resnet.' in key:
            new_key = key.replace('resnet.', '')
        else:
            new_key = key
        if not 'model.layer3.1.bn2.num_batches_tracked' in new_key:
            new_state_dict[new_key] = enc_ckp[key]
    enc_ckp = new_state_dict
    info = enc.load_state_dict(enc_ckp,strict=False)

    if args.rank == 0:
        print(f"Enc Loading: {enc_init_path}")
        print(f"Results: {info}")
    
    return enc

"""Initialize end to end model"""
def build_model(args, device: torch.device, enc_name: str, mil_name: str, hf_token: str, image_input: bool):
    print(f"[Model] Model successfully built. Encoder name: {enc_name}, MIL name: {mil_name}")
    if image_input: # Image input with .lmdb
        print(f"[Model] Sucessfully set up end-to-end model with image input")
        # Setup encoder model
        encoder = build_encoder(enc_name= enc_name, hf_token= hf_token)
        # Setup mil model
        mil, others = build_mil(args, mil_name= mil_name, device= device)
        model = E2E(encoder, mil, device, args).to(device)
    else: # Feature input
        print(f"[Model] Sucessfully set up mil model only with feature input")
        model, others = build_mil(args, mil_name= mil_name, device= device)
    return model

"""Initialize encoder model -> Encode model"""
def build_encoder(enc_name: str, hf_token: str):
    if enc_name == 'r50':
        model = ResNetEncoder(pretrained= True)
    elif enc_name == 'r50v2':
        model = ResNetEncoder('resnet50.tv2_in1k',pretrained= True)
    elif enc_name == 'r18':
        model = ResNetEncoder('resnet18.tv_in1k',pretrained= True)
    elif enc_name == 'r18a':
        kwargs = {'features_only': True, 'pretrained': True, 'num_classes': 0,'out_indices': (4,)}
        model = ResNetEncoder('resnet18.tv_in1k',kwargs=kwargs,pretrained= True)
    elif enc_name == 'uni':
        if hf_token is None:
            raise Exception(f"Get access from https://huggingface.co/MahmoodLab/UNI and create access token")
        os.environ["HF_TOKEN"] = hf_token
        # UNI Encoder
        if 'UNI_CKPT_PATH' not in os.environ or not os.environ['UNI_CKPT_PATH']:
            os.environ['UNI_CKPT_PATH'] = '/XXXwsi_data/ckp/uni/pytorch_model.bin'
        # Check if UNI_CKPT_PATH exists in os.environ and if the file actually exists
        ckpt_path = os.environ.get("UNI_CKPT_PATH")
        if not ckpt_path or not os.path.exists(ckpt_path):
            # Download from Hugging Face if not found locally
            ckpt_path = hf_hub_download(
                repo_id="MahmoodLab/UNI",
                filename="pytorch_model.bin",
                token= hf_token  # Pass token explicitly as a safeguard
            )
            os.environ["UNI_CKPT_PATH"] = ckpt_path
        # Load UNI encoder 
        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    elif enc_name == 'chief':
        model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False,num_classes=0)
        
    elif enc_name == 'gigap':
        if hf_token is None:
            raise Exception(f"Get access from https://huggingface.co/prov-gigapath/prov-gigapath and create access token")
        os.environ["HF_TOKEN"] = hf_token
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False,num_classes=0)
    else:
        raise NotImplementedError(f'{enc_name} not implemented')
    return model

"""Initialize mil method -> Mil architecture"""
def build_mil(args, mil_name: str, device: torch.device):
    others = {}

    genera_model_params,genera_trans_params = get_mil_model_params(args)

    if mil_name == 'rrtmil':
        model = RRTMIL(epeg_k=args.epeg_k,crmsa_k=args.crmsa_k,region_num=args.region_num,n_heads=args.rrt_n_heads,n_layers=args.rrt_n_layers,**genera_model_params).to(device)

    elif mil_name == 'abmil':
        model = DAttention(**genera_model_params).to(device)

    elif mil_name == 'abmilx':
        _ = genera_trans_params.pop('attn_type')
        model = DAttentionX(
            **genera_trans_params,
            D = args.abx_D,
            attn_type = 'mlp',
            attn_bias = args.abx_attn_bias,
            attn_plus = args.abx_attn_plus,
            pad_v = args.abx_pad_v,
            attn_plus_embed_new=args.abx_attn_plus_embed_new,
            ).to(device)
        
    elif mil_name == 'gabmil':
        model = AttentionGated(**genera_model_params).to(device)

    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif mil_name == 'clam_sb':
        model = CLAM_SB(**genera_model_params).to(device)

    elif mil_name == 'clam_mb':
        model = CLAM_MB(**genera_model_params).to(device)

    elif mil_name == 'transmil':
        model = TransMIL(**genera_trans_params).to(device)

    elif mil_name == 'vitmil':
        model = ViTMIL(**genera_trans_params).to(device)

    elif mil_name == 'dsmil':
        model = MILNet(**genera_model_params).to(device)
        if args.aux_alpha == 0.:
            args.main_alpha = 0.5
            args.aux_alpha = 0.5

    elif mil_name == 'dtfd':
        model = DTFD(device=device, lr=args.lr, weight_decay=args.weight_decay, steps=args.num_epoch, input_dim=args.input_dim, n_classes=args.n_classes).to(device)

    elif mil_name == 'meanmil':
        model = MeanMIL(**genera_model_params).to(device)

    elif mil_name == 'maxmil':
        model = MaxMIL(**genera_model_params).to(device)

    elif mil_name == 'meanP':
        model = MeanPooling(**genera_model_params).to(device)

    elif mil_name == 'gigap':
        model = GIGAPMIL(**genera_model_params).to(device)

    elif mil_name == 'chief':
        model = CHIEF(**genera_model_params,dataset=args.datasets.lower()).to(device)
        if 'CHIEF_MIL_PATH' not in os.environ or not os.environ['CHIEF_MIL_PATH']:
            os.environ['CHIEF_MIL_PATH'] = '/XXXwsi_data/ckp/chief/CHIEF_pretraining.pth'
        if os.path.exists(os.environ['CHIEF_MIL_PATH']):
            state_dict = torch.load(os.environ['CHIEF_MIL_PATH'], map_location="cpu")["model"]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                for k in missing_keys:
                    print("Missing ", k)

            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    print("Unexpected ", k)
        else:
            raise NotImplementedError

    elif mil_name == 'wikg':
        model = WiKG(**genera_model_params)
        
    else:
        raise NotImplementedError
    
    return model, others