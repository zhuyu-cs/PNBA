# model settings
model_dict = dict(  
    proj_dict=dict(
        in_dim=1,
        proj_neuron=256,
        frames=16,
        num_param=4,
        dropout=0.,
        base_channel=128
    ),
    encoder_decoder_dict=dict(  
        base_channel=64, 
        ch_mult=(1,2,2,4), 
        num_res_blocks=(1,1,1,1),      
        attn_resolutions=[],   
        dropout=0., 
        resamp_with_conv=True,
        double_z=False,
        give_pre_end=False,
        z_channels=8,             
    ),
    video_dict=dict(
        in_channels=1,
        features=[32, 64, 128, 8],
        num_blocks=[4,3,2,1],
        temporal_kernel=5,
    )
)

# dataset settings
data_root = '/data/Video2activity'
batch_size = 32
seed = 42

# optimizer and learning rate
optimizer = dict(type='AdamW', lr=1e-4,betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=1.0 / 3,
    periods=[200, 200],
    restart_weights=[1, 1],
    min_lr=[1e-4, 1e-7],
)

# runtime settings
gpus = [0,1,2,3,4,5,6,7]
dist_params = dict(backend='nccl')
data_workers = 2  
checkpoint_config = dict(interval=100) 
workflow = [('train', 100),('val',1)]
total_epochs = 400
resume_from = None
load_from = None #'./work_dir/PNBA_mouse_V1/checkpoints/epoch_400.pth'
work_dir = './work_dir/PNBA_mouse_V1'

# logging settings
log_level = 'INFO'
log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook'),
    ])
