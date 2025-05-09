import logging
import os
from argparse import ArgumentParser
import torch
from torch.nn.parallel import DataParallel
from einops import rearrange
from mmcv import Config
from models import make_PNBA
from utils import data_loader_for_rep
import random
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

def get_logger(log_level):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    return logging.getLogger()

def parse_args():
    parser = ArgumentParser(description='Extract latent variables')
    parser.add_argument('config', help='config file path')
    return parser.parse_args()

def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_trial(model, dataset, idx, mouse, all_mouse_dict, tier_name, include_ca=False, include_video=False):
    batch = dataset.__getitem__(idx)
    trial_name = str(idx)
    
    batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
    for bk, bv in batch_kwargs.items():
        batch_kwargs[bk] = torch.unsqueeze(bv, 0)
    
    responses = batch_kwargs['responses']
    video = batch_kwargs['videos']
    
    with torch.no_grad():
        stacked_responses = responses.cuda()
        stacked_video = video.cuda()
        
        joint_mean, joint_logvar, video_mean, video_log_var, ca_mean, ca_log_var = model(
            responses=stacked_responses.permute(0, 2, 1),
            mice=mouse,
            video=stacked_video,
            t=None
        )
        
        joint_mean = joint_mean.permute(0, 3, 1, 2).detach().cpu()
        video_mean = video_mean.permute(0, 3, 1, 2).detach().cpu()
        ca_mean = ca_mean.permute(0, 3, 1, 2).detach().cpu()
        stacked_responses = stacked_responses.detach().cpu()
        
        joint_mean = rearrange(joint_mean, 'b t c n -> (b t) (c n)')
        video_mean = rearrange(video_mean, 'b t c n -> (b t) (c n)')
        ca_mean = rearrange(ca_mean, 'b t c n -> (b t) (c n)')
        stacked_responses = rearrange(stacked_responses, 'b t n -> (b t) (n)')
        
        joint_mean = joint_mean.numpy()
        video_mean = video_mean.numpy()
        ca_mean = ca_mean.numpy()
        stacked_responses = stacked_responses.numpy()
        
        saved_info = {
            'video_mean': video_mean,
            'ca_mean': ca_mean
        }
        
        if include_video:
            stacked_video = stacked_video.permute(0, 2, 1, 3, 4).detach().cpu()
            stacked_video = rearrange(stacked_video, 'b t c h w -> (b t) c h w')
            saved_info['video'] = stacked_video.numpy()
        if include_ca:
            saved_info['ca'] = stacked_responses
        if trial_name not in all_mouse_dict[mouse][tier_name]:
            all_mouse_dict[mouse][tier_name][trial_name] = {}
        all_mouse_dict[mouse][tier_name][trial_name] = saved_info

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    set_random_seed(cfg.seed)
    logger = get_logger(cfg.log_level)
    
    # Distributed settings
    dist = False
    if args.launcher != 'none':
        dist = True
        init_dist(**cfg.dist_params)
        if torch.distributed.get_rank() != 0:
            logger.setLevel('ERROR')
        logger.info('Enabled distributed training.')
    else:
        logger.info('Disabled distributed training.')

    num_workers = cfg.data_workers * len(cfg.gpus)
    
    # Define mice datasets
    mice = [
        'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',      
        'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',  
        'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20', 
        'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20',   
        'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20',
    ]
    
    # Create dataloaders
    data_path = [os.path.join(cfg.data_root, m) for m in mice]
    all_loaders = data_loader_for_rep(
        paths=data_path, 
        frames=cfg.model_dict['proj_dict']['frames'],
        num_workers=num_workers,
        batch_size=1
    )
    
    # Build model
    model = make_PNBA(
        dataloaders=all_loaders,
        proj_dict=cfg.model_dict['proj_dict'],  
        encoder_decoder_dict=cfg.model_dict['encoder_decoder_dict'],  
        video_dict=cfg.model_dict['video_dict']
    )
    print('Model initialized successfully')
    del all_loaders
    
    # Define train and validation mice
    mice_train = [m for m in mice if m not in [
        'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20'
    ]]
    
    mice_val = [
        'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20'
    ]
    
    # Create train and validation dataloaders
    data_path_train = [os.path.join(cfg.data_root, m) for m in mice_train]
    train_loaders = data_loader_for_rep(
        paths=data_path_train, 
        frames=299,
        num_workers=num_workers,
        batch_size=1
    )
    
    data_path_val = [os.path.join(cfg.data_root, m) for m in mice_val]
    val_loaders = data_loader_for_rep(
        paths=data_path_val, 
        frames=299,
        num_workers=num_workers,
        batch_size=1
    )
    
    # Load model weights
    assert cfg.load_from is not None, "Model checkpoint path (cfg.load_from) must be specified"
    model.load_state_dict(torch.load(cfg.load_from)['state_dict'], strict=True)
    model = DataParallel(model, device_ids=cfg.gpus).cuda()
    frames = cfg.model_dict['proj_dict']['frames']
    model.eval()
    
    # Create storage for results
    all_mouse_dict = {mouse: {'train': {}, 'oracle': {}, 'val': {}} for mouse in mice}

    # Process training mice
    for mouse in mice_train:
        print(f'Processing: {mouse}')
        dataset_train = train_loaders['train'][mouse].dataset
        tiers = dataset_train.trial_info.tiers
        
        # Process train trials
        for idx in range(len(tiers)):
            if tiers[idx] == 'train':
                process_trial(model, dataset_train, idx, mouse, all_mouse_dict, 'train')
            elif tiers[idx] == 'oracle':
                process_trial(model, dataset_train, idx, mouse, all_mouse_dict, 'oracle')
    
    # Process validation mice
    for mouse in mice_val:
        print(f'Processing: {mouse}')
        dataset_val = val_loaders['train'][mouse].dataset
        tiers = dataset_val.trial_info.tiers
        
        # Process validation trials
        for idx in range(len(tiers)):
            if tiers[idx] == 'train':
                process_trial(model, dataset_val, idx, mouse, all_mouse_dict, 'train')
            elif tiers[idx] == 'oracle':
                process_trial(model, dataset_val, idx, mouse, all_mouse_dict, 'val', include_video=True)
    
    # Save results
    os.makedirs('./middle_state', exist_ok=True)
    with open(f'./middle_state/all_rep.pkl', "wb") as tf:
        pickle.dump(all_mouse_dict, tf)
    
    print("Processing complete. Results saved.")

if __name__ == '__main__':
    main()