import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rgb_to_grayscale
Sobel = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
Robert = np.array([[0, 0],
                   [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def gradient_no_abs(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def forward(self, R_low, R_high, L_low, L_high, I_low, I_high):
        L_low_3  = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        self.recon_loss_low  = F.l1_loss(R_low * L_low_3,  I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        self.recon_loss_crs_low  = F.l1_loss(R_high * L_low_3, I_low)
        self.recon_loss_crs_high = F.l1_loss(R_low * L_high_3, I_high)
        self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())

        self.Ismooth_loss_low   = self.smooth(L_low, R_low)
        self.Ismooth_loss_high  = self.smooth(L_high, R_high)

        self.loss_Decom = self.recon_loss_high + 0.3 * self.recon_loss_low + 0.001 * self.recon_loss_crs_low + \
                          0.001 * self.recon_loss_crs_high + 0.1 * (self.Ismooth_loss_low + self.Ismooth_loss_high) + 0.1 * self.equal_R_loss

        return self.loss_Decom, self.recon_loss_low + self.recon_loss_high, self.equal_R_loss, self.Ismooth_loss_low + self.Ismooth_loss_high
class all_loss_function(nn.Module):
    def __init__(self):
        super(all_loss_function, self).__init__()
    def gradient(self, input_tensor, direction):
        # 定义平滑核
        smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2).cuda()
        smooth_kernel_y = smooth_kernel_x.transpose(2, 3)  # 转置以获得 y 方向的核

        # 根据方向选择对应的核
        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        else:
            raise ValueError("Direction must be 'x' or 'y'")
        # 对输入进行卷积计算梯度
        gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding='same'))

        # 归一化梯度
        grad_min = gradient_orig.min()
        grad_max = gradient_orig.max()
        grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 1e-4)
        
        return grad_norm
    def mutual_i_input_loss(self, input_I_low, input_im):  # 亮度平滑损失


        # 假设 input_im 是形状为 [batch_size, channels, height, width] 的 RGB 图像
        input_gray = rgb_to_grayscale(input_im)
        # input_gray = input_im  # 假设输入已经是灰度图或需要保持不变

        # 计算 x 和 y 方向的梯度
        low_gradient_x = self.gradient(input_I_low, "x")
        input_gradient_x = self.gradient(input_gray, "x")
        low_gradient_y = self.gradient(input_I_low, "y")
        input_gradient_y = self.gradient(input_gray, "y")

        # 计算 x 和 y 方向的损失
        x_loss = torch.abs(low_gradient_x / torch.clamp(input_gradient_x, min=0.01))
        y_loss = torch.abs(low_gradient_y / torch.clamp(input_gradient_y, min=0.01))

        # 平均化损失
        mut_loss = torch.mean(x_loss + y_loss)
        return mut_loss
    def mutual_i_loss(self, input_I_low):
        """
        相互一致性损失 (mutual_i_loss)

        参数:
            input_I_low: torch.Tensor，输入的低照度图像

        返回:
            mutual_loss: torch.Tensor，相互一致性损失
        """
        # 计算 x 和 y 方向的梯度
        low_gradient_x = self.gradient(input_I_low, "x")
        low_gradient_y = self.gradient(input_I_low, "y")

        # 计算 x 和 y 方向的损失
        x_loss = low_gradient_x * torch.exp(-10 * low_gradient_x)
        y_loss = low_gradient_y * torch.exp(-10 * low_gradient_y)

        # 计算总损失
        mutual_loss = torch.mean(x_loss + y_loss)
        return mutual_loss

    def forward(self, vi, ir, vi_r, vi_l, ir_l):
        recon_vis = torch.mean((vi_r * vi_l - vi)**2)
        recon_ir = torch.mean((ir_l - ir)**2)
        # mutual_i_input_loss = 0
        mutual_i_input_loss = self.mutual_i_input_loss(vi_l, vi)
        mutual_i_loss = self.mutual_i_loss(vi_l)
        # loss_decom =  1000 * recon_vis + 2000 * recon_ir + 7 * mutual_i_input_loss + 9 * mutual_i_loss
        loss_decom =  1000 * recon_vis + 3000 * recon_ir + 100 * mutual_i_input_loss + 9 * mutual_i_loss
        return loss_decom, recon_vis, recon_ir, mutual_i_input_loss, mutual_i_loss
  
def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm