import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.models as models
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
        x_loss = low_gradient_x * torch.exp(-20 * low_gradient_x)
        y_loss = low_gradient_y * torch.exp(-20 * low_gradient_y)

        # 计算总损失
        mutual_loss = torch.mean(x_loss + y_loss)
        return mutual_loss
    def RGB_Y(self, input_tensor):
        # 将张量转换为 numpy 数组，并调整形状为 OpenCV 需要的格式 (H, W, C)
        input_images = input_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()  # 转换为 (2, 128, 128, 3)
        input_images = (input_images * 255).astype(np.uint8)  # 将值从 [0, 1] 转换为 [0, 255]

        # 初始化一个列表，用于存储 Y 通道
        y_channels = []

        # 遍历批次中的每一张图像
        for img in input_images:
            # 将图像从 RGB 转换为 YCrCb
            img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            
            # 分离 Y, Cr, Cb 通道
            y_channel, _, _ = cv2.split(img_ycrcb)
            
            # 将 Y 通道添加到列表
            y_channels.append(y_channel)

        # 将 Y 通道列表转换为 numpy 数组
        y_channels = np.stack(y_channels)  # 形状为 (2, 128, 128)

        # 将 Y 通道扩展为单通道张量 (2, 1, 128, 128)
        y_tensor = torch.from_numpy(y_channels).unsqueeze(1).float() / 255.0

        # 将 Y 通道复制到三通道，得到形状为 (2, 3, 128, 128) 的张量
        y_tensor_3c = y_tensor.expand(-1, 3, -1, -1)
        return y_tensor_3c
    def RGB_Y_histogram_equalization(self, input_tensor):
        # 将张量转换为 numpy 数组，并调整形状为 OpenCV 需要的格式 (H, W, C)
        
        input_images = input_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()  # 转换为 (2, 128, 128, 3)
        input_images = (input_images * 255).astype(np.uint8)  # 将值从 [0, 1] 转换为 [0, 255]

        # 初始化一个列表，用于存储处理后的图像
        output_images = []

        # 遍历批次中的每一张图像
        for img in input_images:
            # 将图像从 RGB 转换为 YCrCb
            img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            
            # 分离 Y, Cr, Cb 通道
            y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)
            
            # 对 Y 通道进行直方图均衡化
            y_channel_eq = cv2.equalizeHist(y_channel)
            
            # 将均衡化后的 Y 通道复制到三通道
            y_channel_eq_3c = cv2.merge([y_channel_eq, y_channel_eq, y_channel_eq])
            
            # 将处理后的图像添加到输出列表
            output_images.append(y_channel_eq_3c)

        # 将输出列表转换为 numpy 数组
        output_images = np.stack(output_images)  # 形状为 (2, 128, 128, 3)

        # 将 numpy 数组转换回 PyTorch 张量，并调整形状为 (2, 3, 128, 128)
        output_tensor = torch.from_numpy(output_images).permute(0, 3, 1, 2).float() / 255.0
        return output_tensor
    def Perceptual_loss(self, vis, vis_r):
        vis = vis.cuda()
        vis_r = vis_r.cuda()
        vgg_model = models.vgg19()
        pre_file = torch.load('vgg19-dcbb9e9d.pth')
        vgg_model.load_state_dict(pre_file)
        # 查看模型整体结构
        # structure = torch.nn.Sequential(*list(vgg_model.children())[:])
        # print(structure)
        
        # # 查看模型各部分名称
        # print('模型各部分名称', vgg_model._modules.keys())

        # features = torch.nn.Sequential(*list(vgg_model.children())[0][16:30])

        # print('features of vgg19: ', features)  :11  11:21
        vgg_model_featrure1 = vgg_model.features[:11]
        vgg_model_featrure2 = vgg_model.features[11:21]
        # vgg_model_featrure1 = vgg_model.features[:11]
        # vgg_model_featrure2 = vgg_model.features[11:21]
        # print(torch.nn.Sequential(*list(vgg_model_featrure1)))
        # print(torch.nn.Sequential(*list(vgg_model_featrure2)))

        vgg_model_featrure1 = vgg_model_featrure1.cuda()
        vgg_model_featrure2 = vgg_model_featrure2.cuda()
        # vgg_model_featrure = vgg_model.features
        # # print(*list(vgg_model_featrure.children()))
        # vgg_model_featrure = vgg_model_featrure.cuda()
        out1 = vgg_model_featrure1(vis)
        out2 = vgg_model_featrure1(vis_r)
        out3 = vgg_model_featrure2(out1)
        out4 = vgg_model_featrure2(out2)
        # print(out1.size())

        # 假设 orignal_conv3_feature 和 Generate_conv3_feature 是 PyTorch 的张量
        feature4_loss = torch.mean(torch.abs(out2- out1))
        feature5_loss = torch.mean(torch.abs(out3- out4))
        Perceptualloss = feature4_loss + feature5_loss
        # print(Perceptualloss.size())
        return Perceptualloss
    # def forward(self, vi, ir, vi_r, vi_l, ir_l):
    def forward(self, vi, ir, vi_r, vi_l):
        recon_vis = torch.mean((vi_r * vi_l - vi)**2)
        # recon_ir = torch.mean((ir_l - ir)**2)
        # mutual_i_input_loss = 0
        mutual_i_input_loss = self.mutual_i_input_loss(vi_l, vi)
        mutual_i_loss = self.mutual_i_loss(vi_l)
        vi_r_y = self.RGB_Y(vi_r)  # 得到Y通道
        vi_y = self.RGB_Y_histogram_equalization(vi)  # Y通道经过直方均衡化
        Per_loss = self.Perceptual_loss(vi_y, vi_r_y)
        # Per_loss = torch.tensor(0.0)
        # Per_loss = torch.zeros(1).cuda()
        # noise--loss_decom = 400* recon_vis + 5* mutual_i_input_loss + 2* mutual_i_loss + 20*Per_loss -200
        # wo-noise--loss_decom = 500* recon_vis + 7* mutual_i_input_loss + 2* mutual_i_loss + 25*Per_loss  -50
        loss_decom =1.0* recon_vis + 0.014* mutual_i_input_loss + 0.01* mutual_i_loss + 0.04*Per_loss
        # return loss_decom, recon_vis, recon_ir, mutual_i_input_loss, mutual_i_loss, Per_loss
        return loss_decom, recon_vis, mutual_i_input_loss, mutual_i_loss, Per_loss

  
def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm