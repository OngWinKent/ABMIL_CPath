import torch
from .common_mil import CommonMIL
from modules.e2e import Normalize
from contextlib import suppress

class E2E(CommonMIL):
    def __init__(self, args, device) -> None:
        super().__init__(args)
        self.dataset = None
        self.loader = None
        self.device = device
        self.max_psize = args.max_patch_train or args.same_psize # Maximum patch size for training
        self.norm = Normalize(device,args.channels_last) # Normalization module for patches
        self.static_pos=[self.max_psize] # Static position, seems related to patch selection or positional encoding
        self.inference_mode = torch.inference_mode if args.freeze_enc else suppress # Context manager for inference mode

    def init_func_train(self):
        self.training = True # Set training mode

    def after_get_data_func(self):
        pass # Placeholder for actions after data is fetched

    """Main forward pass logic for training."""
    def forward_func(self,model,bag,label,criterion,batch_size,mil_name,pos=None,feat=None,**kwargs):
        if type(bag) in (tuple,list): # If bag is a tuple/list, it might contain (patches, positions) or (patches, features)
            bag,ps = bag # Unpack bag and patch selection info (ps)
            
        if len(bag.size()) == 5: # If bag has 5 dimensions (e.g., [1, N, C, H, W]), squeeze the first dim
            bag = bag.squeeze(0)

        patch_num = bag.size(0) # Total number of patches in the bag
        selected_patches = bag # Use all patches
        remaining_patches = None
        ori_indices = None

        keep_num = selected_patches.size(0) # Number of patches kept
        if pos is not None:
            if len(pos.shape) == 2: # Ensure pos has a batch dimension
                pos = pos.unsqueeze(0)

        # Forward pass through the model
        #x_input = (selected_patches, remaining_patches)
        x_input = selected_patches if remaining_patches is None else (selected_patches, remaining_patches)
        if 'dsmil' in mil_name or 'clam' in mil_name:
            # For DS-MIL or CLAM models, which might have specific input formats or auxiliary losses
            logits,aux_loss,_ = model(x_input,ps=ori_indices,B=batch_size,label=label,loss=criterion,pos=pos)
        else:
            # For other models
            logits = model(x_input,ori_indices,pos=pos,B=batch_size,feat=feat)
            aux_loss = 0. # No auxiliary loss by default for other models

        return logits,label,aux_loss,patch_num,keep_num,0. # Return model outputs and related info
    
    def after_backward_func(self):
        pass
    
    def final_train_func(self):
        pass

    """Validation forward pass logic."""
    def validate_func(self,model,bag,label, mil_name):
        if type(bag) in (tuple,list):
            if isinstance(bag[1], torch.Tensor):
                # Input is (patches, positions)
                bag, pos = bag
                ps = None
            # batch input
            else:
                # Input is (patches, patch_selection_info) or similar
                bag,ps = bag
                pos = None
        else:
            ps = None # No patch selection info
            pos = None # No positional info

        if len(bag.size()) == 5: # If bag has 5 dimensions, squeeze the first one
            bag = bag.squeeze(0)

        patch_num = bag.size(0) # Total number of patches
        all_patches = bag # Use all patches for validation
        del bag # Free memory
        keep_num = all_patches.size(0) # Number of patches kept (all of them)
        # Use only main model for validation
        logits = model(all_patches,pos=pos)
        logits = logits[0] if 'dsmil' in mil_name else logits # Adjust output format for DS-MIL
    
        return logits,label # Return logits and labels
    
    def init_func_val(self):
        pass