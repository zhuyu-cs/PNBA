import torch
import torch.nn as nn
from .encoder_decoder_ca import (LDProjector, invLDProjector,
                                Ca_Encoder, Ca_Decoder,
                                ResnetBlock,nonlinearity)
from .encoder_decoder_video import VideoEncoder, VideoDecoder
from einops import rearrange
import torch.nn.functional as F
import math
import numpy as np

def compute_probmatch_loss(video_mean, video_var, ca_mean, ca_var, a=1.0, b=0.0):
    """
    video_mean, video_var: mean and log-variance for (mixed) images (B, c, n, T)
    ca_mean, ca_var: mean and log-variance for captions (B, c, n, T)
    a, b: learnable affine transform parameters
    """
    B, c, n, T = video_mean.shape

    # Reshape and merge B and T dimensions
    video_mean = video_mean.reshape(B, -1)
    ca_mean = ca_mean.reshape(B, -1)
    video_var = video_var.reshape(B, -1)
    ca_var = ca_var.reshape(B, -1)

    mu_dist = ((video_mean.unsqueeze(1) - ca_mean.unsqueeze(0)) ** 2).sum(-1)
    
    sigma_dist = ((video_var.unsqueeze(1) + ca_var.unsqueeze(0))).sum(-1)
    # # Compute logits
    logits = -a * (mu_dist+sigma_dist) + b

    labels = torch.eye(B, device=ca_mean.device)
    # Compute matching loss
    match_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
    return match_loss

def compute_structure_mix_kl(post_mean, post_log_var, modal_mean, modal_log_var):
    """
    Compute Structure-Preserving KL divergence for each time step
    Args:
        post_mean, post_log_var: posterior distribution parameters (B, c, n, T)
        modal_mean, modal_log_var: modal-specific distribution parameters (B, c, n, T)
    """
    B, c, n, T = post_mean.shape
    
    # Reshape to (B*T, c*n)
    post_mean = post_mean.reshape(B, -1)
    post_log_var = post_log_var.reshape(B, -1)
    modal_mean = modal_mean.reshape(B, -1)
    modal_log_var = modal_log_var.reshape(B, -1)
    
    kl_loss = -0.5 * torch.sum(1 + post_log_var - modal_log_var - 
                              ((post_mean - modal_mean).pow(2) + post_log_var.exp()) / modal_log_var.exp(),
                              dim=-1)
    return kl_loss

def compute_symmetric_kl(video_mean, video_logvar, ca_mean, ca_logvar):
    B=video_mean.shape[0]
    video_mean = video_mean.reshape(B, -1)
    video_logvar = video_logvar.reshape(B, -1)
    ca_mean = ca_mean.reshape(B, -1)
    ca_logvar = ca_logvar.reshape(B, -1)

    kl_v_c = -0.5 * torch.sum(1 + video_logvar - ca_logvar - 
                             ((video_mean - ca_mean).pow(2) + video_logvar.exp()) / ca_logvar.exp(),
                             dim=-1)
    
    kl_c_v = -0.5 * torch.sum(1 + ca_logvar - video_logvar - 
                             ((ca_mean - video_mean).pow(2) + ca_logvar.exp()) / video_logvar.exp(),
                             dim=-1)
    
    symmetric_kl = (kl_v_c + kl_c_v) 
    
    return symmetric_kl

class PNBA(nn.Module):
    def __init__(self, 
                 mices_dict,
                 proj_dict,
                 encoder_decoder_dict,
                 video_dict,
    ):
        super().__init__()

        # input projectors.
        self.in_projectors = LDProjector(mices_dict=mices_dict,
                                        in_dim=proj_dict['in_dim'], 
                                        proj_neuron=proj_dict['proj_neuron'], 
                                        base_channel=proj_dict['base_channel'], 
                                        dropout=proj_dict['dropout']
        )
        
        # feature encoder
        self.encoder = Ca_Encoder(
            ch=encoder_decoder_dict['base_channel'],
            ch_mult=encoder_decoder_dict['ch_mult'],
            num_res_blocks=encoder_decoder_dict['num_res_blocks'],
            attn_resolutions=encoder_decoder_dict['attn_resolutions'],
            dropout=encoder_decoder_dict['dropout'],
            resamp_with_conv=encoder_decoder_dict['resamp_with_conv'], 
            in_channels=proj_dict['base_channel'], 
            resolution=proj_dict['proj_neuron'],
            z_channels=encoder_decoder_dict['z_channels'], 
            double_z=False
        )
        
        self.video_encoder = VideoEncoder(in_channels=video_dict['in_channels'],
                                        features=video_dict['features'],
                                        temporal_kernel=video_dict['temporal_kernel'],
                                        num_blocks=video_dict['num_blocks'],
        )
        
        self.video_embed_mean = ResnetBlock(in_channels=encoder_decoder_dict['z_channels'], 
                                            out_channels=encoder_decoder_dict['z_channels'], 
                                            dropout=encoder_decoder_dict['dropout'])
        self.video_embed_log_var = ResnetBlock(in_channels=encoder_decoder_dict['z_channels'], 
                                            out_channels=encoder_decoder_dict['z_channels'], 
                                            dropout=encoder_decoder_dict['dropout'])
        
        self.ca_embed_mean = ResnetBlock(in_channels=encoder_decoder_dict['z_channels'], 
                                        out_channels=encoder_decoder_dict['z_channels'], 
                                        dropout=encoder_decoder_dict['dropout'])
        self.ca_embed_logvar = ResnetBlock(in_channels=encoder_decoder_dict['z_channels'], 
                                        out_channels=encoder_decoder_dict['z_channels'], 
                                        dropout=encoder_decoder_dict['dropout'])
        
        # feature decoder
        self.decoder = Ca_Decoder(
            ch=encoder_decoder_dict['base_channel']*2, 
            ch_mult=encoder_decoder_dict['ch_mult'], 
            num_res_blocks=encoder_decoder_dict['num_res_blocks'], 
            attn_resolutions=encoder_decoder_dict['attn_resolutions'],  
            dropout=encoder_decoder_dict['dropout'], 
            resamp_with_conv=encoder_decoder_dict['resamp_with_conv'],  
            in_channels=proj_dict['base_channel'], 
            resolution=proj_dict['proj_neuron'],
            z_channels=encoder_decoder_dict['z_channels']
        )

        # output projectors
        self.output_projectors = invLDProjector(mices_dict=mices_dict,
                                                in_dim=proj_dict['in_dim'],
                                                base_channel=proj_dict['base_channel'], 
                                                dropout=proj_dict['dropout'],
                                                )
        
        self.video_decoder = VideoDecoder(in_channels=video_dict['in_channels'],
                                        features=np.flip(video_dict['features']),
                                        temporal_kernel=video_dict['temporal_kernel'],
                                        num_blocks=np.flip(video_dict['num_blocks']),
        )
        
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, responses, video, mice): 
        # ca: B N t
        # mice: id of mouse.

        # project original trace to the same spatial size (Neuron), eg., B 1 7440 32 -> B 128 1024 32 
        proj_ca = self.in_projectors(responses) 
        # encode
        embed_ca = self.encoder(proj_ca) # B c N t
        b, c = embed_ca.shape[:2]
        embed_video = self.video_encoder(video)
        t_video, h, w = embed_video.shape[2:]
        embed_video_for_ca = rearrange(embed_video, 'b c t h w -> b c t (h w)').permute(0,1,3,2)
        if embed_video_for_ca.shape[-1]*embed_video_for_ca.shape[-2] != embed_ca.shape[-1]*embed_ca.shape[-2]:
            embed_video_for_ca = F.interpolate(embed_video_for_ca, size=(embed_ca.shape[-1]*embed_ca.shape[-2]//t_video, t_video), mode="nearest").reshape(embed_ca.shape)
        else:
            embed_video_for_ca = embed_video_for_ca.reshape(embed_ca.shape)
        
        ca_mean = self.ca_embed_mean(embed_ca)
        ca_log_var =self.ca_embed_logvar(embed_ca)
        video_mean = self.video_embed_mean(embed_video_for_ca)
        video_log_var = self.video_embed_log_var(embed_video_for_ca)

        post_mean = (ca_mean/(1+torch.exp(ca_log_var-video_log_var))) + (video_mean/(1 + torch.exp(video_log_var-ca_log_var))) 
        post_log_var = video_log_var + ca_log_var - torch.log(torch.exp(video_log_var) + torch.exp(ca_log_var))
        
        pld_video_kl_loss = compute_structure_mix_kl(post_mean, post_log_var, ca_mean, ca_log_var)
        pld_ca_kl_loss = compute_structure_mix_kl(post_mean, post_log_var, video_mean, video_log_var)
        
        probmatch_loss = compute_probmatch_loss(video_mean, video_log_var.exp(), ca_mean, ca_log_var.exp(), a=self.a, b=self.b)
        cross_kl_loss = compute_symmetric_kl(video_mean, video_log_var, ca_mean, ca_log_var)
        
        plds = post_mean + torch.exp(0.5 * post_log_var) * torch.randn_like(post_mean).to(ca_mean.device)
        
        ################### Restoration.##################
        # # decode
        restore_proj_ca = self.decoder(plds) # B c N t
        restored_ca = self.output_projectors(restore_proj_ca, mice).permute(0,2,1)
        # restore video to ouput
        video_sample = F.interpolate(plds.reshape(b, c, -1, t_video), size=(h*w, t_video), mode="nearest")
        video_sample = rearrange(video_sample.permute(0,1,3,2), 'b c t (h w) -> b c t h w',t=t_video,h=h,w=w)
        restore_video = self.video_decoder(video_sample)

        # self decode ca
        ca_sample = ca_mean+torch.exp(0.5 * ca_log_var) * torch.randn_like(ca_mean).to(ca_mean.device)
        
        # self decode video
        video_sample = video_mean+torch.exp(0.5 * video_log_var) * torch.randn_like(ca_mean).to(ca_mean.device)
        # ################### Cross-modality neural encoding and decoding.###########################
        #### neural encoding
        video_sample = self.decoder(video_sample)
        restore_ca_from_video = self.output_projectors(video_sample, mice).permute(0,2,1)
        
        #### neural decoding
        ca_sample = F.interpolate(ca_sample.reshape(b, c, -1, t_video), size=(h*w, t_video), mode="nearest")
        ca_sample = rearrange(ca_sample.permute(0,1,3,2), 'b c t (h w) -> b c t h w',t=t_video,h=h,w=w)
        restore_video_from_ca = self.video_decoder(ca_sample)
        return restore_video, restore_video_from_ca, restore_ca_from_video, restored_ca, pld_video_kl_loss, pld_ca_kl_loss, probmatch_loss,cross_kl_loss
