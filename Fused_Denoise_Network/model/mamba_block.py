import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
# from 
from .vmamba import MambaIR
from einops import rearrange

from .masks import CentralMaskedConv2d
class Denoise(nn.Module):
    def __init__(self, inp_channel = 3, channel=48, img_size = [120, 120], stride1 = 2, stride2 = 3, 
                 depths1 = [1, 1, 1, 1], depths2 = [1, 1, 1, 1], 
                 pd_a =5, pd_b = 2, pd_pad=2, R3 = True, R3_T = 16, R3_p = 0.16):  # channel 32
        super(Denoise, self).__init__()

        self.vi_Mamba = vi_Mamba(dim=channel, img_size = img_size, depths = depths1, stride1=stride1, stride2=stride2)
        self.ir_Mamba = ir_Mamba(dim=channel, img_size = img_size, depths = depths2, stride1=stride1, stride2=stride2)
        self.tail = nn.Sequential(nn.Conv2d(channel*2*2,  channel,    kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel,    channel//2, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                #   nn.Conv2d(channel//2, channel//2, kernel_size=1),
                                #   nn.ReLU(inplace=True),
                                  nn.Conv2d(channel//2, inp_channel,     kernel_size=1)
                                  )
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
    def forward(self, x1, x2, pd=None):  # 3, H, W
        if pd is None: pd = self.pd_a

        # do PD
        if pd > 1:
            x1 = pixel_shuffle_down_sampling(x1, f=pd, pad=self.pd_pad)
            x2 = pixel_shuffle_down_sampling(x2, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            x1 = F.pad(x1, (p, p, p, p))
            x2 = F.pad(x2, (p, p, p, p))
        f_vi = self.vi_Mamba(x1)
        f_ir = self.ir_Mamba(x2)
        fused = self.tail(torch.cat((f_vi, f_ir), dim=1))

        if pd > 1:
            fused = pixel_shuffle_up_sampling(fused, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            fused = fused[:, :, p:-p, p:-p]
        return fused

    def fined_fused(self, x1, x2):
        b, c, h, w = x1.shape

        # pad images for PD process
        if h % self.pd_b != 0:
            x1 = F.pad(x1, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
            x2 = F.pad(x2, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x1 = F.pad(x1, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
            x2 = F.pad(x2, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor
        fused = self.forward(x1, x2, pd=self.pd_b)
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return fused[:, :, :h, :w]
        else:
            xt = x1 
            xt2 = x2
            denoised = torch.empty(*(xt.shape), self.R3_T, device=xt.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(xt)
                mask = indice < self.R3_p

                tmp_input = torch.clone(fused).detach()
                tmp_input[mask] = xt[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                indice2 = torch.rand_like(xt2)
                mask2 = indice2 < self.R3_p /2.0

                tmp_input2 = torch.clone(fused).detach()

                tmp_input2[mask2] = xt2[mask2]
                p = self.pd_pad
                tmp_input2 = F.pad(tmp_input2, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised_tem = self.vi_Mamba(tmp_input)
                    denoised_tem2 = self.ir_Mamba(tmp_input2)
                    denoised[..., t] = self.tail(denoised_tem)
                else:
                    denoised_tem = self.vi_Mamba(tmp_input)[:, :, p:-p, p:-p]
                    denoised_tem2 = self.ir_Mamba(tmp_input2)[:, :, p:-p, p:-p]
                    denoised[..., t] = self.tail(torch.cat((denoised_tem, denoised_tem2), dim=1))

            return torch.mean(denoised, dim=-1)
        # return 0



class vi_Mamba(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim = 48,
                 img_size=[120, 120],
                 bias=False,
                 depths = [4, 6, 6, 4],
                 stride1 =2,
                 stride2 = 3
                 ):

        super(vi_Mamba, self).__init__()
        self.head_vi = nn.Sequential(nn.Conv2d(inp_channels, dim, kernel_size=1),
                                  nn.ReLU(inplace=True)
                                  )
        
        self.bsn_vi1 = CentralMaskedConv2d(dim, dim, kernel_size=2 * stride1 - 1, stride=1, padding=stride1 - 1)
        self.bsn_vi2 = CentralMaskedConv2d(dim, dim, kernel_size=2 * stride2 - 1, stride=1, padding=stride2 - 1)
        self.bsn1 =  nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=1),
                                   nn.ReLU(inplace=True)
        )
        
        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = MambaIR(img_size=img_size, embed_dim=dim, depths=[depths[0]])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[1]])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = MambaIR(img_size=img_size, embed_dim=dim*2**2, depths=[depths[2]])
        self.decoder_level3 = MambaIR(img_size=img_size, embed_dim=dim*2**2, depths=[depths[2]])
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[1]])
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[0]])
        self.refinement = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[3]])
        # self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x1):
        x1 = self.head_vi(x1)
        bsn_x11 = self.bsn_vi1(x1)
        bsn_x12 = self.bsn_vi2(x1)
        inp_enc_level1 = self.bsn1(torch.cat((bsn_x11, bsn_x12), dim = 1))

        # inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_dec_level3 = out_enc_level3

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # out_dec_level1 = self.refinement(out_dec_level1)
        # out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1

class ir_Mamba(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 img_size = [120, 120],
                 bias=False,
                 depths = [4, 6, 6, 4],
                 stride1 = 2,
                 stride2 = 3,
                 ):

        super(ir_Mamba, self).__init__()
        self.head_ir = nn.Sequential(nn.Conv2d(inp_channels, dim, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.bsn_ir1 = CentralMaskedConv2d(dim, dim, kernel_size=2 * stride1 - 1, stride=1, padding=stride1 - 1)
        
        self.bsn_ir2 = CentralMaskedConv2d(dim, dim, kernel_size=2 * stride2 - 1, stride=1, padding=stride2 - 1)

        self.bsn2 =  nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=1),
                                   nn.ReLU(inplace=True)
                                   )
        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = MambaIR(img_size=img_size, embed_dim=dim, depths=[depths[0]])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[1]])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = MambaIR(img_size=img_size, embed_dim=dim*2**2, depths=[depths[2]])
        self.decoder_level3 = MambaIR(img_size=img_size, embed_dim=dim*2**2, depths=[depths[2]])
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[1]])
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[0]])
        self.refinement = MambaIR(img_size=img_size, embed_dim=dim*2**1, depths=[depths[3]])
        # self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x2):
        x2 = self.head_ir(x2)
        bsn_x21 = self.bsn_ir1(x2)
        bsn_x22 = self.bsn_ir2(x2)
        inp_dec_level1 = self.bsn2(torch.cat((bsn_x21, bsn_x22), dim = 1))
        # inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_dec_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_dec_level3 = out_enc_level3

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # out_dec_level1 = self.refinement(out_dec_level1)
        # out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1


        # return out_dec_level1
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    x1 = torch.rand(1, 3, 20, 20).cuda()
    x2 = torch.rand(1, 3, 20, 20).cuda()
    model = Denoise().cuda()
    out = model.fined_fused(x1, x2)
    print(out.size())