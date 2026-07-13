import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from vm_base import MambaIR
from einops import rearrange

class DecomNet(nn.Module):
    def __init__(self, inp_channel = 3, channel=32, kernel_size=3):  # channel 32
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(inp_channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, 5, padding=2, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        self.net1_recon = nn.Conv2d(channel, 1,  1, padding=0, padding_mode='replicate')


        self.TDN_Mamba_R = TDN_Mamba(dim=32, depths = [1, 1, 1, 1])
    #     self.output = nn.Conv2d(int(channel *2), 3, kernel_size=3, stride=1, padding=1, bias=False)

    #     self.sobel_x = torch.tensor([[[[-1, 0, 1],
    #                             [-2, 0, 2],
    #                             [-1, 0, 1]]]], dtype=torch.float32)

    #     self.sobel_y = torch.tensor([[[[-1, -2, -1],
    #                                 [ 0,  0,  0],
    #                                 [ 1,  2,  1]]]], dtype=torch.float32)

    #     # Laplacian 核
    #     self.laplacian = torch.tensor([[[[0,  1,  0],
    #                                     [1, -4,  1],
    #                                     [0,  1,  0]]]], dtype=torch.float32)
    #     self.r1 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
    #     self.r2 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
    #     self.l = nn.Parameter(torch.tensor(1.0, requires_grad=True))
    #     self.laplacian_conv = nn.Sequential(nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate'),
    #                                     nn.ReLU(),
    #                                     nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
    #                                     nn.ReLU(),
    #                                     nn.Conv2d(channel, channel, 1))
    #     self.sobel_conv = nn.Conv2d(channel, channel, 1)
    # def apply_filter(self, feature_maps, kernel):
    #     """对 feature_maps 进行 Sobel 或 Laplacian 计算"""
    #     kernel = kernel.to(feature_maps.device)
    #     B, C, H, W = feature_maps.shape
        
    #     # 扩展核以匹配输入通道数，并设置分组卷积
    #     kernel = kernel.repeat(C, 1, 1, 1)  # 形状变为 (C, 1, 3, 3)
        
    #     # 使用分组卷积，每个输入通道独立处理
    #     return F.conv2d(feature_maps, kernel, 
    #                     padding=1,       # 保持空间分辨率
    #                     groups=C)        # 关键修改：设置分组数为输入通道数
    def forward(self, input_im):  # 3, H, W
        # R = self.TDN_R(input_im)
        f, R = self.TDN_Mamba_R(input_im)
        # feats0 = self.net1_conv0(input_im)
        feats0 = f
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        # feats1 = self.net1_conv0(input_im)
        # feats1 = self.net2_convs(feats0)
        # out1 = self.net2_recon(feats1)       
        vi_r = torch.sigmoid(R)
        vi_l = torch.sigmoid(outs)
        # ir_l = torch.sigmoid(out1)
        # return vi_r, vi_l, ir_l
        return vi_r, vi_l



##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## MDLA improved by designing Multi-scale Convolution
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv_3 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv_dwconv_5 = nn.Conv2d(dim * 3, dim * 3, kernel_size=5, stride=1, padding=2, groups=dim * 3, bias=bias)
        self.qkv_dwconv_7 = nn.Conv2d(dim * 3, dim * 3, kernel_size=7, stride=1, padding=3, groups=dim * 3, bias=bias)

        self.q_proj = nn.Conv2d(dim * 3, dim, kernel_size=1 ,stride=1, padding=0, bias=bias)
        self.k_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.v_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.qkv(x)
        qkv_1 = self.qkv_dwconv_3(x)
        q_1, k_1, v_1 = qkv_1.chunk(3, dim=1)

        qkv_2 = self.qkv_dwconv_5(x)
        q_2, k_2, v_2 = qkv_2.chunk(3, dim=1)

        qkv_3 = self.qkv_dwconv_7(x)
        q_3, k_3, v_3 = qkv_3.chunk(3, dim=1)

        q = self.q_proj(torch.cat([q_1, q_2, q_3], dim=1))
        k = self.k_proj(torch.cat([k_1, k_2, k_3], dim=1))
        v = self.v_proj(torch.cat([v_1, v_2, v_3], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

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


##########################################################################
# improved the information Multi-scale conv and lightweights design
class TDN(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 2, 2, 4], #2，3，3，4
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'
                 ):

        super(TDN, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
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

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1
    
class TDN_Mamba(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 bias=False,
                 depths = [4, 6, 6, 4]
                 ):

        super(TDN_Mamba, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.encoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1 = MambaIR(img_size=128, embed_dim=dim, depths=[depths[0]])
        self.down1_2 = Downsample(dim)
        # self.encoder_level2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2 = MambaIR(img_size=128, embed_dim=dim*2**1, depths=[depths[1]])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        # self.encoder_level3 = nn.Sequential(*[
            # TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
            #                  bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3 = MambaIR(img_size=128, embed_dim=dim*2**2, depths=[depths[2]])
        # self.decoder_level3 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level3 = MambaIR(img_size=128, embed_dim=dim*2**2, depths=[depths[2]])
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2 = MambaIR(img_size=128, embed_dim=dim*2**1, depths=[depths[1]])
        self.up2_1 = Upsample(int(dim * 2 ** 1))

        # self.decoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = MambaIR(img_size=128, embed_dim=dim*2**1, depths=[depths[0]])

        # self.refinement = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement = MambaIR(img_size=128, embed_dim=dim*2**1, depths=[depths[3]])
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
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

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        return inp_enc_level1, out_dec_level1


        # return out_dec_level1

if __name__ == "__main__":
    x = torch.rand(24, 3, 128, 128).cuda()
    model = TDN_Mamba().cuda()
    out = model(x)
    print(out.size())