from copy import deepcopy
from torch.utils.data import DataLoader
from .dataset_img import ClsLMDBDataset
from .dataset_feat import FeatClsDataset
from .data_utils import *
from typing import *

"""Dataloader main config function"""
def build_dataloader(args, image_input: bool):
    # Initialize dataframe
    df = get_data_dfs(dataset_name= args.datasets, csv_file= args.csv_path, seed= args.seed, val_ratio= args.val_ratio)
    # Split dataset using split field
    train_dfs, test_dfs, val_dfs = get_split_dfs(df= df)

    # Create dataset dictionary
    dataset = { 'train': train_dfs, 'test': test_dfs, 'val': val_dfs}
    if image_input: # Image .lmdb input
        train_loader,val_loader,test_loader = build_img_loader(args= args, dataset= dataset)
    else: # Pre-extracted feature .pt input
        train_loader,val_loader,test_loader = build_feat_dataloader(args= args, dataset= dataset)
    return train_loader,val_loader,test_loader

"""Image dataloader"""
def build_img_loader(args, dataset):
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}
    if args.lmdb:
        loader_kwargs['collate_fn'] = collate_fn_nbs
        if args.num_workers > 1:
            loader_kwargs['prefetch_factor'] = args.prefetch_factor
        
    loader_kwargs_val = deepcopy(loader_kwargs)
    loader_kwargs_test = deepcopy(loader_kwargs)

    if args.num_workers_test is not None:
        loader_kwargs_test['num_workers'] = args.num_workers_test
        loader_kwargs_val['num_workers'] = args.num_workers_test
        if args.num_workers_test < 2:
            loader_kwargs_test['prefetch_factor'] = None
            loader_kwargs_val['prefetch_factor'] = None

    if args.image_input and args.batch_size > 1:
        if args.sel_type == 'random':
            loader_kwargs['collate_fn'] = collate_fn_img_batch
        else:
            loader_kwargs['collate_fn'] = collate_fn_img_batch_list
        
    # Get dataset
    train_set = ClsLMDBDataset(args,args.env_train,dataset['train'],persistence=args.persistence,keep_same_psize= args.same_psize, mode="train",_type=args.datasets,channels_last=args.channels_last,img_size=args.img_size, h5_root=args.h5_path)
    #_test_img_size = 224
    _test_img_size = args.img_size
    test_set = ClsLMDBDataset(args,args.env,dataset['test'],persistence=args.persistence,_type=args.datasets,channels_last=args.channels_last,mode="test",h5_root=args.h5_path,img_size=_test_img_size)
    val_set = ClsLMDBDataset(args,args.env,dataset['val'],persistence=args.persistence,_type=args.datasets,channels_last=args.channels_last,mode="val",h5_root=args.h5_path,img_size=_test_img_size)

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size,shuffle=True, drop_last=args.drop_last,**loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=1,shuffle=False, **loader_kwargs_val)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_kwargs_test)

    # Prefetch config: Setting fp16 here will affect performance
    train_loader = PrefetchLoader(train_loader,device=args.device,need_transform=args.img_transform != 'none', transform_type=args.img_transform,img_size=args.img_size,trans_chunk=args.img_trans_chunk,crop_scale=args.crop_scale,load_gpu_later=args.load_gpu_later_train)
    assert not args.no_prefetch_test
    val_loader = PrefetchLoader(val_loader,device=args.device,is_train=False,load_gpu_later=args.load_gpu_later,trans_chunk=args.img_trans_chunk)
    test_loader = PrefetchLoader(test_loader,device=args.device, is_train=False,load_gpu_later=args.load_gpu_later,trans_chunk=args.img_trans_chunk)
        
    return train_loader,val_loader,test_loader

"""Feature data loader"""
def build_feat_dataloader(args, dataset):
    train_loader = _get_feat_dataloader(args,dataset['train'],root=args.dataset_root,train=True)
    val_loader = _get_feat_dataloader(args,dataset['val'],root=args.dataset_root,train=False)
    test_loader = _get_feat_dataloader(args,dataset['test'],root=args.dataset_root,train=False)
    return train_loader,val_loader,test_loader

def _get_feat_dataloader(args, dataset, root, train=True):
    keep_same_psize = args.same_psize
    
    # Extract slide IDs (p) and labels (l) from DataFrame, tuple, or list
    if hasattr(dataset, 'columns'):  # Pandas DataFrame
        cols_lower = {str(c).lower(): c for c in dataset.columns}
        
        label_key = next((cols_lower[k] for k in ['label', 'labels', 'target', 'class', 'category'] if k in cols_lower), None)
        slide_key = next((cols_lower[k] for k in ['slide_id', 'slide_ids', 'slide', 'filename', 'file', 'id', 'name', 'path'] if k in cols_lower), None)
        
        if label_key is not None and slide_key is not None:
            p = dataset[slide_key].values
            l = dataset[label_key].values
        elif label_key is not None:
            l = dataset[label_key].values
            other_cols = [c for c in dataset.columns if c != label_key]
            p = dataset[other_cols[0]].values
        else:
            p, l = dataset.iloc[:, 0].values, dataset.iloc[:, 1].values
    elif isinstance(dataset, (tuple, list)):
        p, l = dataset[0], dataset[1]
    else:
        p, l = dataset

    # Auto-swap if p and l are inverted (ensures 'l' contains numeric labels)
    try:
        int(l[0])
    except (ValueError, TypeError):
        p, l = l, p

    if train:
        _dataset = FeatClsDataset(p, l, root, persistence=args.persistence, keep_same_psize=keep_same_psize, is_train=True, _type=args.datasets, args=args)
    else:
        _dataset = FeatClsDataset(p, l, root, persistence=args.persistence, _type=args.datasets, args=args)

    loader_kwargs = {'pin_memory': args.pin_memory}

    if train:
        _dataloader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, num_workers=args.num_workers, **loader_kwargs)
    else:
        _num_workers_test = args.num_workers_test or args.num_workers
        _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=_num_workers_test, pin_memory=args.pin_memory)
    _dataloader = PrefetchLoader(_dataloader,device=args.device,need_norm=False)
    return _dataloader