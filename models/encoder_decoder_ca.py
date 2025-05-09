import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import numpy as np
import torch.nn.functional as F

class LDProjector(torch.nn.Module):
    def __init__(
        self,
        mices_dict, 
        in_dim=1,
        proj_neuron=1024, 
        base_channel=128,
        dropout=0.0,
    ):
        super().__init__()

        self.proj_neuron = proj_neuron
        self.neuron_expand = nn.Conv2d(in_channels=in_dim,
                                       out_channels=base_channel,
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=(0,1),
                                       bias=True
                                       )
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=base_channel, eps=1e-6, affine=True) 
        self.act = nn.LeakyReLU()
        # the first resblock
        self.resblock1 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        # atten block
        self.atten = AttnBlock(base_channel)
        # the second resblock
        self.resblock2 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base_channel, eps=1e-6, affine=True)  #nn.ModuleDict({key:nn.GroupNorm(num_groups=32, num_channels=base_channel, eps=1e-6, affine=True) for key in mices_dict.keys()})
        
         
    def forward(self, x):
        # id_ = self.key_id[mice].float().to(x.device)
        len_t = x.shape[-1]
        x = self.neuron_expand(x.unsqueeze(1))    # N 128 N t       
        x = torch.nn.functional.adaptive_max_pool2d(x, output_size=(self.proj_neuron, len_t)) # N 128 1024 t       
        x = self.norm1(x)
        x = self.act(x)
        # the first resblock
        x = self.resblock1(x)
        # the attention block
        x = self.atten(x)
        # the second resblock
        x = self.resblock2(x)

        x = self.norm_out(x)
        x = self.act(x)
        return  x       # N 128 1024 t
         
class Interpolate(nn.Module):
    def __init__(self,  num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
    
    def forward(self, x):
        len_t = x.shape[-1]
        return torch.nn.functional.interpolate(x, size=(self.num_neurons, len_t), mode="bicubic")

class invLDProjector(torch.nn.ModuleDict):

    def __init__(
        self,
        mices_dict, 
        in_dim=1, 
        base_channel=128,
        dropout=0.1
    ):
        super().__init__()
        self.interpolate = nn.ModuleDict({ key: Interpolate(num_neurons) for key, num_neurons in mices_dict.items()})
        # the first resblock
        self.resblock1 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        # the second resblock
        self.resblock2 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        # final out
        self.process1 = nn.Conv2d(base_channel,
                                     base_channel,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=base_channel, eps=1e-6, affine=True) 
        self.act = nn.LeakyReLU()
        self.process2 = nn.Conv2d(base_channel,
                                     base_channel//2,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=base_channel//2, eps=1e-6, affine=True)
        self.out = nn.Conv2d(base_channel//2,
                                     in_dim,
                                     kernel_size=1)

    def forward(self, x, mice):
        x = self.interpolate[mice](x)
        # the first resblock
        x = self.resblock1(x)
        # the second resblock
        x = self.resblock2(x)
        
        # final out
        x = self.process1(x) 
        x = self.norm1(x) 
        x = self.act(x) 
        x = self.process2(x) 
        x = self.norm2(x) 
        x = self.act(x) 
        return F.elu(self.out(x)).squeeze(1)+1

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    if in_channels<=32:
        num_groups=in_channels
    else:
        num_groups=32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                    dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        n,t = x.shape[2:]
        x = torch.nn.functional.interpolate(x, size=(n*4,t), mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=(4, 1), 
                                        stride=(4, 1),
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=(4,1), stride=(4,1))
        return x

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=False)
        self.proj_out = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        
        # Reshape input
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Compute q, k, v
        qkv = self.qkv(x).reshape(b, h*w, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = torch.empty(b, h*w, h*w, dtype=q.dtype, device=q.device)
        attn = torch.baddbmm(attn, q, k.transpose(-2, -1), beta=0, alpha=c**-0.5)
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        x = torch.bmm(attn, v)

        # Apply projection
        x = self.proj_out(x)

        # Reshape output
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # Add residual connection
        return residual + x

class Ca_Encoder(nn.Module):
    def __init__(self, *, ch=64, ch_mult=(1,2,4,8), num_res_blocks=(2,2,2,2),
                 attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=128,
                 resolution=1024, z_channels=512, double_z=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels

        assert len(ch_mult) == len(num_res_blocks), "ch_mult and num_res_blocks must have the same length"
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block 
            down.attn = attn
            
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)                
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)                

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Ca_Decoder(nn.Module):
    def __init__(self, *, ch=64, ch_mult=(1,2,4,8), num_res_blocks=(2,2,2,2),
                 attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=128,
                 resolution=1024, z_channels=512, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        assert len(ch_mult) == len(num_res_blocks), "ch_mult and num_res_blocks must have the same length"
        self.num_res_blocks = num_res_blocks
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

