from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling

from .APBSN import APBSN
from .CSCBSN import CSCBSN
from .MMBSN import MMBSN
# from .APBSN_mamba import APBSN_mamba as APBSN
class BSN(nn.Module):

    # @classmethod
    # @lru_cache(maxsize=1)
    def bsn_model(cls):
        return cls()
#5 1 1
    # def __init__(self, type='APBSN', pd_a=5, pd_b=2, pd_pad=2 , R3=True, R3_T=16, R3_p=0.16,    
    #              in_ch=3, bsn_base_ch=128, bsn_num_module=3, DCL1_num=2, DCL2_num=7, mask_type='o'):  # mamba
    def __init__(self, type='APBSN', pd_a=5, pd_b=2, pd_pad=1 , R3=True, R3_T=8, R3_p=0.16,
                 in_ch=3, bsn_base_ch=128, bsn_num_module=9, DCL1_num=2, DCL2_num=7, mask_type='o'):
        '''
        Args:
            type           : BSN model type
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
            mask_type      : types of mask,eg:'o_fsz', 'o_r_c'
        '''
        super().__init__()
        # base_ch = bsn_base_ch
        # _mask= mask_type.split('_')  # 'o' + 'a45'
        # mask_number = len(_mask)  # 2
        ly = []
        ly += [ nn.Conv2d(bsn_base_ch*2*2, bsn_base_ch*2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(bsn_base_ch*2, bsn_base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(bsn_base_ch,    bsn_base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(bsn_base_ch//2, bsn_base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(bsn_base_ch//2, 3,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)
        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        
        # define network
        if type == 'APBSN':
            self.bsn1 = APBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'CSCBSN':
            self.bsn1 = CSCBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'MMBSN':
            self.bsn1 = MMBSN(in_ch, in_ch, bsn_base_ch, DCL1_num, DCL2_num, mask_type)
        else:
            raise NotImplementedError('bsn type %s is not implemented' % type)

        if type == 'APBSN':
            self.bsn2 = APBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'CSCBSN':
            self.bsn2 = CSCBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'MMBSN':
            self.bsn2 = MMBSN(in_ch, in_ch, bsn_base_ch, DCL1_num, DCL2_num, mask_type)
        else:
            raise NotImplementedError('bsn type %s is not implemented' % type)
    def forward(self, img, img2, pd=None):  # 训练
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a

        # do PD
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
            pd_img2 = pixel_shuffle_down_sampling(img2, f=pd, pad=self.pd_pad)
            # print(pd_img.size())
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))
            pd_img2 = F.pad(img2, (p, p, p, p))

        # forward blind-spot network
        # pd_img_denoised = self.bsn(pd_img)
        pd_img_denoised_feature = self.bsn1(pd_img)
        # input("Enter...")
        pd_img_denoised_feature2 = self.bsn2(pd_img2)
        # print(pd_img_denoised_feature.size())
        pd_img_denoised = self.tail(torch.cat((pd_img_denoised_feature, pd_img_denoised_feature2), dim=1))

        # do inverse PD
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]

        # return img_pd_bsn
        return img_pd_bsn
    def initinit(self, vis, ir):
        return torch.max(vis, ir)
    

    def compute_gradient(self, img):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)

        grad = torch.abs(grad_x) + torch.abs(grad_y)
        return grad

    def maxmax(self, vis_img, ir_img):
        # vis_img: [B, 3, H, W], ir_img: [B, 3, H, W]

        # 分别计算每个通道的梯度图
        vis_grads = []
        ir_grads = []

        for c in range(3):
            vis_channel = vis_img[:, c:c+1, :, :]
            ir_channel = ir_img[:, c:c+1, :, :]

            vis_grad = self.compute_gradient(vis_channel)
            ir_grad = self.compute_gradient(ir_channel)

            vis_grads.append(vis_grad)
            ir_grads.append(ir_grad)

        vis_grad_img = torch.cat(vis_grads, dim=1)  # [B, 3, H, W]
        ir_grad_img = torch.cat(ir_grads, dim=1)    # [B, 3, H, W]

        # 创建 mask，表示红外梯度更大
        mask = (ir_grad_img > vis_grad_img).float()  # [B, 3, H, W]

        # 融合图像：在每个像素、每个通道上选择梯度较大对应的原始像素值
        fused_img = vis_img * (1 - mask) + ir_img * mask  # [B, 3, H, W]

        return fused_img

    def denoise1(self, x, x2):  # 推理
        '''
        Denoising process for inference.
        '''
        b, c, h, w = x.shape

        # pad images for PD process
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
            x2 = F.pad(x2, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
            x2 = F.pad(x2, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor
        img_pd_bsn = self.forward(img=x, img2=x2, pd=self.pd_b)
        

        # print(img_pd_bsn.size())
        # img_pd_bsn = self.tail(img_pd_bsn_feature)
        # print(img_pd_bsn[:, :, :h, :w].shape)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:, :, :h, :w]
        else:
            xt = self.initinit(x, x2)
            xt2 = self.initinit(x, x2)
            # xt2 = self.initinit(x, x2)
            # xt = x 
            # xt2 = x2
            denoised = torch.empty(*(xt.shape), self.R3_T, device=xt.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(xt)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = xt[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                indice2 = torch.rand_like(xt2)
                mask2 = indice2 < self.R3_p

                tmp_input2 = torch.clone(img_pd_bsn).detach()

                tmp_input2[mask2] = xt2[mask2]
                p = self.pd_pad
                tmp_input2 = F.pad(tmp_input2, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised_tem = self.bsn1(tmp_input)
                    denoised_tem2 = self.bsn2(tmp_input2)
                    denoised[..., t] = self.tail(denoised_tem)
                else:
                    # print(tmp_input.size())
                    # denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
                    # print(denoised[..., t].size())
                    denoised_tem = self.bsn1(tmp_input)[:, :, p:-p, p:-p]
                    denoised_tem2 = self.bsn2(tmp_input2)[:, :, p:-p, p:-p]
                    denoised[..., t] = self.tail(torch.cat((denoised_tem, denoised_tem2), dim=1))

            return torch.mean(denoised, dim=-1)
    def denoise(self, x, x2):  # 推理
        '''
        Denoising process for inference.
        '''
        b, c, h, w = x.shape

        # pad images for PD process
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
            x2 = F.pad(x2, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
            x2 = F.pad(x2, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor
        img_pd_bsn = self.forward(img=x, img2=x2, pd=self.pd_b)
        

        # print(img_pd_bsn.size())
        # img_pd_bsn = self.tail(img_pd_bsn_feature)
        # print(img_pd_bsn[:, :, :h, :w].shape)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:, :, :h, :w]
        else:
            xt = x 
            xt2 = x2
            denoised = torch.empty(*(xt.shape), self.R3_T, device=xt.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(xt)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = xt[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                indice2 = torch.rand_like(xt2)
                mask2 = indice2 < self.R3_p

                tmp_input2 = torch.clone(img_pd_bsn).detach()

                tmp_input2[mask2] = xt2[mask2]
                p = self.pd_pad
                tmp_input2 = F.pad(tmp_input2, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised_tem = self.bsn1(tmp_input)
                    denoised_tem2 = self.bsn2(tmp_input2)
                    denoised[..., t] = self.tail(denoised_tem)
                else:
                    # print(tmp_input.size())
                    # denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
                    # print(denoised[..., t].size())
                    denoised_tem = self.bsn1(tmp_input)[:, :, p:-p, p:-p]
                    denoised_tem2 = self.bsn2(tmp_input2)[:, :, p:-p, p:-p]
                    denoised[..., t] = self.tail(torch.cat((denoised_tem, denoised_tem2), dim=1))

            return torch.mean(denoised, dim=-1)



# def compute_gradient(self, img):
    #     # Sobel算子
    #     sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    #     sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)

    #     grad_x = F.conv2d(img, sobel_x, padding=1)
    #     grad_y = F.conv2d(img, sobel_y, padding=1)

    #     grad = torch.abs(grad_x) + torch.abs(grad_y)
    #     return grad

    # def maxmax(self, vis_img, ir_img):
    #     # vis_img: [B, 3, H, W], ir_img: [B, 1, H, W]
        
    #     vis_grads = []
    #     for c in range(3):
    #         vis_channel = vis_img[:, c:c+1, :, :]
    #         vis_grad = self.compute_gradient(vis_channel)
    #         vis_grads.append(vis_grad)
    #     ir_grads = []
    #     for c in range(3):
    #         ir_channel = ir_img[:, c:c+1, :, :]
    #         ir_grad = self.compute_gradient(ir_channel)
    #         ir_grads.append(ir_grad)

    #     vis_grad_img = torch.cat(vis_grads, dim=1)  # [B, 3, H, W]
    #     ir_grad_img = torch.cat(ir_grads, dim=1)  # [B, 3, H, W]

    #     # ir_grad = self.compute_gradient(ir_img)  # [B, 1, H, W]
    #     # ir_grad_img = ir_grad.repeat(1, 3, 1, 1)  # 复制为3通道 [B, 3, H, W]

    #     # 对应每个像素、每个通道取最大值
    #     joint_grad = torch.max(vis_grad_img, ir_grad_img)  # [B, 3, H, W]

    #     return joint_grad

# if __name__ == '__main__':
if __name__ == "__main__":
    x = torch.rand(1, 3, 30, 30).cuda()
    y = torch.rand(1, 3, 30, 30).cuda()
    model = BSN().cuda()
    # input = torch.cat((x, y), dim=1)
    out = model(x, y)
    print(out.size())