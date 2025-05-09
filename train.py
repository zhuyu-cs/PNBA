import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from einops import rearrange
from mmcv import Config
from mmcv.runner import Runner

from models import make_PNBA
from utils import mouse_video_loader_train, mouse_video_loader_val, get_correlations
from neuralpredictors.training import LongCycler
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

def get_logger(log_level):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger

def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    return parser.parse_args()

def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    set_random_seed(cfg.seed)

    logger = get_logger(cfg.log_level)
    
    logger.info('Start training.')

    num_workers = cfg.data_workers * len(cfg.gpus)
    batch_size = cfg.batch_size
    shuffle = True
    
    mice = ['dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
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
    data_path = [os.path.join(cfg.data_root, m) for m in mice]
    all_loaders = mouse_video_loader_train(paths = data_path, 
                                    frames = cfg.model_dict['proj_dict']['frames'],
                                    num_workers = num_workers,
                                    batch_size = batch_size)
    # build model   
    model = make_PNBA(dataloaders=all_loaders,
                        proj_dict=cfg.model_dict['proj_dict'],  
                        encoder_decoder_dict=cfg.model_dict['encoder_decoder_dict'],
                        video_dict=cfg.model_dict['video_dict']
                        )
    
    del all_loaders
    mice_train = ['dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
        # 'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
        'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',      
        'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',  
        'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20', 
        'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20',   
        # 'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20',
    ]
    data_path_train = [os.path.join(cfg.data_root, m) for m in mice_train]
    train_loaders = mouse_video_loader_train(paths = data_path_train, 
                                    frames = cfg.model_dict['proj_dict']['frames'],
                                    num_workers = num_workers,
                                    batch_size = batch_size)
    
    mice_val = [ 
        'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce', 
        'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20',
    ]
    data_path_val = [os.path.join(cfg.data_root, m) for m in mice_val]
    val_loaders = mouse_video_loader_val(paths = data_path_val, 
                                    frames = 299,
                                    num_workers = num_workers,
                                    batch_size = 1)
    
    model = DataParallel(model, device_ids=cfg.gpus).cuda()
    
    frames = cfg.model_dict['proj_dict']['frames']

    def batch_processor(model, data, train_mode,cur_iters):
        data_key, real_data = data
        video = real_data['videos']
        responses =  real_data['responses']
        
        frames = video.shape[2]    
        if train_mode:  
            index = real_data['index']
        else:
            b,m,t = responses.shape[0],responses.shape[1], responses.shape[2]
            base_index = torch.arange(m * t, device=video.device).view(1, m, t)
            index = base_index.repeat(b, 1, 1) 
            video = rearrange(video, 'b m n t h w -> (b m) n t h w')
            responses = rearrange(responses, 'b m t n -> (b m) t n')
            index = rearrange(index, 'b m t -> (b m) t')
            
        video = video.cuda(non_blocking=True)
        responses = responses.cuda(non_blocking=True)  
        index = index.cuda(non_blocking=True)
        pred_video, pred_video_from_ca, restore_ca_from_video, pred_ca, pld_video_kl_loss,pld_ca_kl_loss,probmatch_loss, cross_kl_loss = model(responses=responses.permute(0,2,1), 
                                                                   video=video,
                                                                   mice=data_key,
                                                                   t=index)
        loss_ca = torch.sum(pred_ca+1e-8 - (responses+1e-8) * torch.log(pred_ca+1e-8))
        
        loss_ca += torch.sum(restore_ca_from_video+1e-8 - (responses+1e-8) * torch.log(restore_ca_from_video+1e-8))
        loss_video = torch.sum((pred_video - video)**2)
        loss_video += torch.sum((pred_video_from_ca-video)**2)
        batch=video.shape[0]
        total_loss = (loss_ca + loss_video + pld_ca_kl_loss.sum()+ pld_video_kl_loss.sum() + probmatch_loss.sum() + cross_kl_loss.sum())/responses.shape[0]
        mse = F.mse_loss(pred_ca, responses, reduce='mean')
        cos_similarity = F.cosine_similarity(pred_ca.reshape(batch,-1), 
                                        responses.reshape(batch,-1), 
                                        dim=-1).mean()
        
        mse_video = torch.mean((pred_video - video)**2, dim=[1, 2, 3, 4])
        psnr = torch.mean(10. * torch.log10(1. / (mse_video + 1e-8)))
        mse_video_cross = torch.mean((pred_video_from_ca - video)**2, dim=[1, 2, 3, 4])
        cross_psnr = torch.mean(10. * torch.log10(1. / (mse_video_cross + 1e-8)))
        

        log_vars = OrderedDict()
        log_vars['video_mse'] = loss_video.item()/responses.shape[0]
        log_vars['video_psnr'] = psnr.item()
        log_vars['ca2video_psnr'] = cross_psnr.item()
        log_vars['ca_poisson_loss'] = loss_ca.item()/responses.shape[0]
        log_vars['pld_video_kl_loss'] = pld_video_kl_loss.sum().item()/ responses.shape[0]
        log_vars['pld_ca_kl_loss'] = pld_ca_kl_loss.sum().item()/ responses.shape[0]
        log_vars['cos_similarity'] = cos_similarity.item()
        log_vars['mse_ca'] = mse.item()        
        log_vars['cross_kl_loss'] = cross_kl_loss.sum().item()/responses.shape[0]
        log_vars['probmatch_loss'] = probmatch_loss.sum().item()/responses.shape[0]
        log_vars['whole_loss'] = total_loss.item()
        
        if train_mode == False:
            validation_correlation = get_correlations(
                                        model_output=restore_ca_from_video,
                                        responses=responses,
                                        per_neuron=False,
                                    )
            log_vars['correlation'] = validation_correlation.item()
        
        outputs = dict(loss=total_loss, log_vars=log_vars, num_samples=responses.shape[0])
        return outputs

    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        log_level=cfg.log_level)
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)

    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run([LongCycler(train_loaders['train']), LongCycler(val_loaders['train'])], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
