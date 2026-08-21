import os
import torch
import numpy as np
import re
from torch.utils.data import Dataset
from .data_utils import *
from pathlib import Path

class FeatClsDataset(Dataset):
    def __init__(self, file_name=None, file_label=None,root=None,persistence=True,keep_same_psize=0,is_train=False,_type='nsclc',args=None):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(FeatClsDataset, self).__init__()

        self.patient_name = file_name
        self.slide_label = []
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        self.same_psize_pad_type = args.same_psize_pad_type
        self.h5_path = args.h5_path

        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    _feat = torch.load(os.path.join(self.root,'pt_files',_sides[_idx]),weights_only=True)
                    if keep_same_psize:
                        _feat = get_smae_psize(_feat,keep_same_psize,args.same_psize_pad_type,args.min_seq_len)
                    self.slide_name.append(_feat)
                    self.slide_label.append(file_label[i])
                else:
                    self.slide_name.append(_sides[_idx])
                    self.slide_label.append(file_label[i])
        if _type.lower().startswith('bio'):
            self.slide_label = [int(_l) for _l in self.slide_label]
        else:
            if 'nsclc' in _type.lower():
                self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]
            elif 'brca' in _type.lower():
                self.slide_label = [ 0 if _l == 'IDC' else 1 for _l in self.slide_label]
            elif 'call' in _type.lower():
                self.slide_label = [ 0 if _l == 'normal' else 1 for _l in self.slide_label]
            
            #elif re.search(r'panda', _type.lower()) is not None:
            #    self.slide_label = [int(_l) for _l in self.slide_label]
            
            elif _type.lower() in ('panda', 'camelyon16'):
                self.slide_label = [int(_l) for _l in self.slide_label]
            else:
                raise NotImplementedError

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        label = self.slide_label[idx]

        if self.h5_path is not None:
            _pos = get_seq_pos_fn(os.path.join(self.h5_path,Path(file_path).stem+'.h5'))
        else:
            _pos = None

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path),weights_only=True)
            if self.keep_same_psize:
                if _pos is not None:
                    features,_pos[1] = get_smae_psize(features,self.keep_same_psize,self.same_psize_pad_type,pos=_pos[1])
                else:
                    features = get_smae_psize(features,self.keep_same_psize,self.same_psize_pad_type)

        slide_id = Path(file_path).stem
        outputs = {'input': features, 'target':int(label), "slide_id": slide_id}

        if _pos is not None:
            _pos = torch.cat(_pos,dim=0)
            if (_pos.shape[0] - 1) != features.shape[0]:
                print(_pos.shape)
                print(features.shape)
                raise AssertionError
            outputs['pos'] = _pos

        return outputs