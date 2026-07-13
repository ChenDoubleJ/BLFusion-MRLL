import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2
from losses import fusion_loss
from loss_decom_TDN import Decom_Loss, all_loss_function

# def read_data(root: str):
#     assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

#     train_root = os.path.join(root, "train")
#     val_root = os.path.join(root, "test")
#     assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
#     assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

#     train_images_low_path = []
#     train_images_high_path = []
#     val_images_low_path = []
#     val_images_high_path = []

#     supported = [".jpg", ".JPG", ".png", ".PNG"]
#     train_high_root = os.path.join(train_root, "high")
#     train_low_root= os.path.join(train_root, "low")

#     val_high_root = os.path.join(val_root, "high")
#     val_low_root = os.path.join(val_root, "low")
#     train_low_path = [os.path.join(train_low_root, i) for i in os.listdir(train_low_root)
#                   if os.path.splitext(i)[-1] in supported]
#     train_high_path= [os.path.join(train_high_root, i) for i in os.listdir(train_high_root)
#                   if os.path.splitext(i)[-1] in supported]

#     val_low_path = [os.path.join(val_low_root, i) for i in os.listdir(val_low_root)
#                   if os.path.splitext(i)[-1] in supported]
#     val_high_path= [os.path.join(val_high_root, i) for i in os.listdir(val_high_root)
#                   if os.path.splitext(i)[-1] in supported]

#     assert len(train_low_path)==len(train_high_path),' The length of train dataset does not match. low:{}, high:{}'.format(len(train_low_path),len(train_high_path))
#     assert len(val_low_path)==len(val_high_path),' The length of val dataset does not match. low:{}, high:{}'.format(len(val_low_path),len(val_high_path))
#     print("image pair check finish")

#     for index in range(len(train_low_path)):
#         img_low_path=train_low_path[index]
#         img_high_path=train_high_path[index]
#         train_images_low_path.append(img_low_path)
#         train_images_high_path.append(img_high_path)

#     for index in range(len(val_low_path)):
#         img_low_path=val_low_path[index]
#         img_high_path=val_high_path[index]
#         val_images_low_path.append(img_low_path)
#         val_images_high_path.append(img_high_path)

#     total_dataset_nums = len(train_low_path) + len(train_high_path) + len(val_low_path) + len(val_high_path)
#     print("{} images were found in the dataset.".format(total_dataset_nums))
#     print("{} low light images for training.".format(len(train_low_path)))
#     print("{} normal light images for training ref.".format(len(train_high_path)))
#     print("{} low light images for validation.".format(len(val_low_path)))
#     print("{} normal light images for validation ref.".format(len(val_high_path)))

#     return train_low_path, train_high_path, val_low_path, val_high_path
def read_data_test(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    # train_images_low_path = []
    # train_images_high_path = []
    # val_images_low_path = []
    # val_images_high_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]
    train_vi = os.path.join(train_root, "vi_l")
    train_vi_r = os.path.join(train_root, "vi_r")
    train_ir = os.path.join(train_root, "ir")
    val_vi = os.path.join(val_root, "vi_l")
    val_vi_r = os.path.join(val_root, "vi_r")
    val_ir = os.path.join(val_root, "ir")
    val_vis = os.path.join(val_root, "vis")
    # val_visible = os.path.join(val_root, "visible")
    # val_infrared = os.path.join(val_root, "infrared")
    train_vi_path = [os.path.join(train_vi, i) for i in os.listdir(train_vi)
                  if os.path.splitext(i)[-1] in supported]
    train_vi_r_path = [os.path.join(train_vi_r, i) for i in os.listdir(train_vi_r)
                  if os.path.splitext(i)[-1] in supported]
    train_ir_path= [os.path.join(train_ir, i) for i in os.listdir(train_ir)
                  if os.path.splitext(i)[-1] in supported]

    val_vi_path = [os.path.join(val_vi, i) for i in os.listdir(val_vi)
                  if os.path.splitext(i)[-1] in supported]
    val_vi_r_path = [os.path.join(val_vi_r, i) for i in os.listdir(val_vi_r)
                  if os.path.splitext(i)[-1] in supported]
    val_ir_path= [os.path.join(val_ir, i) for i in os.listdir(val_ir)
                  if os.path.splitext(i)[-1] in supported]
    val_vis_path= [os.path.join(val_vis, i) for i in os.listdir(val_vis)
                  if os.path.splitext(i)[-1] in supported]

    # assert len(train_visible_path)==len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.format(len(train_visible_path),len(train_infrared_path))
    # assert len(val_visible_path)==len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.format(len(val_visible_path),len(val_infrared_path))
    # print("image pair check finish")

    # for index in range(len(train_low_path)):
    #     img_low_path=train_low_path[index]
    #     img_high_path=train_high_path[index]
    #     train_images_low_path.append(img_low_path)
    #     train_images_high_path.append(img_high_path)

    # for index in range(len(val_low_path)):
    #     img_low_path=val_low_path[index]
    #     img_high_path=val_high_path[index]
    #     val_images_low_path.append(img_low_path)
    #     val_images_high_path.append(img_high_path)

    # total_dataset_nums = len(train_low_path) + len(train_high_path) + len(val_low_path) + len(val_high_path)
    # print("{} images were found in the dataset.".format(total_dataset_nums))
    # print("{} low light images for training.".format(len(train_low_path)))
    # print("{} normal light images for training ref.".format(len(train_high_path)))
    # print("{} low light images for validation.".format(len(val_low_path)))
    # print("{} normal light images for validation ref.".format(len(val_high_path)))

    return train_vi_path, train_vi_r_path, train_ir_path, val_vi_path, val_vi_r_path, val_ir_path, val_vis_path
def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    # train_images_low_path = []
    # train_images_high_path = []
    # val_images_low_path = []
    # val_images_high_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]
    train_vi = os.path.join(train_root, "vi_l")
    train_vi_r = os.path.join(train_root, "vi_r")
    train_ir = os.path.join(train_root, "ir")
    val_vi = os.path.join(val_root, "vi_l")
    val_vi_r = os.path.join(val_root, "vi_r")
    val_ir = os.path.join(val_root, "ir")
    # val_visible = os.path.join(val_root, "visible")
    # val_infrared = os.path.join(val_root, "infrared")
    train_vi_path = [os.path.join(train_vi, i) for i in os.listdir(train_vi)
                  if os.path.splitext(i)[-1] in supported]
    train_vi_r_path = [os.path.join(train_vi_r, i) for i in os.listdir(train_vi_r)
                  if os.path.splitext(i)[-1] in supported]
    train_ir_path= [os.path.join(train_ir, i) for i in os.listdir(train_ir)
                  if os.path.splitext(i)[-1] in supported]

    val_vi_path = [os.path.join(val_vi, i) for i in os.listdir(val_vi)
                  if os.path.splitext(i)[-1] in supported]
    val_vi_r_path = [os.path.join(val_vi_r, i) for i in os.listdir(val_vi_r)
                  if os.path.splitext(i)[-1] in supported]
    val_ir_path= [os.path.join(val_ir, i) for i in os.listdir(val_ir)
                  if os.path.splitext(i)[-1] in supported]

    # assert len(train_visible_path)==len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.format(len(train_visible_path),len(train_infrared_path))
    # assert len(val_visible_path)==len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.format(len(val_visible_path),len(val_infrared_path))
    # print("image pair check finish")

    # for index in range(len(train_low_path)):
    #     img_low_path=train_low_path[index]
    #     img_high_path=train_high_path[index]
    #     train_images_low_path.append(img_low_path)
    #     train_images_high_path.append(img_high_path)

    # for index in range(len(val_low_path)):
    #     img_low_path=val_low_path[index]
    #     img_high_path=val_high_path[index]
    #     val_images_low_path.append(img_low_path)
    #     val_images_high_path.append(img_high_path)

    # total_dataset_nums = len(train_low_path) + len(train_high_path) + len(val_low_path) + len(val_high_path)
    # print("{} images were found in the dataset.".format(total_dataset_nums))
    # print("{} low light images for training.".format(len(train_low_path)))
    # print("{} normal light images for training ref.".format(len(train_high_path)))
    # print("{} low light images for validation.".format(len(val_low_path)))
    # print("{} normal light images for validation ref.".format(len(val_high_path)))

    return train_vi_path, train_vi_r_path, train_ir_path, val_vi_path, val_vi_r_path, val_ir_path

def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    # loss_function = Decom_Loss()
    # loss_function = all_loss_function()

    if torch.cuda.is_available():
        # loss_function = loss_function.to(device)
        loss_function = fusion_loss().to(device)


    total_loss = torch.zeros(1).to(device)
    loss_ssim = torch.zeros(1).to(device)
    loss_max = torch.zeros(1).to(device)
    loss_color = torch.zeros(1).to(device)
    loss_grad = torch.zeros(1).to(device)
    loss_l1 = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        vi_l, vi_r, ir, name = data

        if torch.cuda.is_available():
            vi_l = vi_l.to(device)
            vi_r = vi_r.to(device)
            ir = ir.to(device)
        I_ir = ir[:, :1, :, :]
        I_input = torch.cat((vi_r, I_ir), dim=1)
        # print(I_input.size())
        fuse = model(vi_r, ir)
        # R_high, L_high = model(I_high)

        loss, loss_ssim, loss_max, loss_color, loss_grad, loss_l1  = loss_function(vi_r, ir, fuse)

        loss.backward()

        total_loss += loss.detach()
        loss_ssim += loss_ssim.detach()
        loss_max += loss_max.detach()
        loss_color += loss_color.detach()
        loss_grad += loss_grad.detach()
        loss_l1 += loss_l1.detach()


        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  loss_ssim: {:.3f}  loss_max: {:.3f}  loss_color: {:.3f} loss_grad: {:.3f} loss_l1: {:.3f}  lr: {:.6f}".format(epoch, total_loss.item() / (step + 1),
            loss_ssim.item() / (step + 1), loss_max.item() / (step + 1), loss_color.item() / (step + 1), loss_grad.item() / (step + 1), loss_l1.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()
    lr_scheduler.step()

    return total_loss.item() / (step + 1), loss_ssim.item() / (step + 1), loss_max.item() / (step + 1), loss_color.item() / (step + 1), loss_grad.item() / (step + 1), loss_l1.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    # # loss_function = Decom_Loss()
    # loss_function = all_loss_function()
    # if torch.cuda.is_available():
    #     # loss_function = loss_function.to(device)
    # loss_function = fusion_loss.to(device)


    val_total_loss = torch.zeros(1).to(device)
    val_loss_ssim = torch.zeros(1).to(device)
    val_loss_max = torch.zeros(1).to(device)
    val_loss_color = torch.zeros(1).to(device)
    val_loss_grad = torch.zeros(1).to(device)
    val_loss_l1 = torch.zeros(1).to(device)
    model.eval()

    # val_total_loss = torch.zeros(1).to(device)
    # val_rec_vis_loss = torch.zeros(1).to(device)
    # val_rec_ir_loss = torch.zeros(1).to(device)
    # val_mc_loss = torch.zeros(1).to(device)
    # val_smooth_loss = torch.zeros(1).to(device)
    save_epoch = 100

    if torch.cuda.is_available():
        # loss_function = loss_function.to(device)
        loss_function = fusion_loss().to(device)
    
    if epoch != 0 and (epoch % save_epoch == 0 or epoch == 399):
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        vi_l, vi_r, ir, name = data
        if torch.cuda.is_available():
            vi_l = vi_l.to(device)
            vi_r = vi_r.to(device)
            ir = ir.to(device)
        I_ir = ir[:, :1, :, :]
        val_input = torch.cat((vi_r, I_ir), dim=1)
        # print(I_input.size())
        fuse_val = model.denoise1(vi_r, ir)
        # fuse_val = model.fined_fused(vi_r, ir)

        # fuse_val = model(vi_r, ir)
        # # R_high, L_high = model(I_high)
        # vi_r, vi_l, ir_l = model(I_input)
        # # R_high, L_high = model(I_high)

        val_loss, val_loss_ssim, val_loss_max, val_loss_color, val_loss_grad, val_loss_l1  = loss_function(vi_r, ir, fuse_val)

        if epoch!=0 and (epoch % save_epoch == 0 or epoch == 299):
            # R_low_img = tensor2numpy_R(vi_r)
            # R_high_img = tensor2numpy_R(vi_l)
            # L_low_img = tensor2numpy_L(ir_l)
            # L_high_img = tensor2numpy_L(ir_l)
            # save_pic(R_low_img, evalfold_path, str(step) + "vi_r")
            # save_pic(R_high_img, evalfold_path, str(step) + "vi_l")
            # save_pic(L_low_img, evalfold_path, str(step) + "ir_l")
            # save_pic(L_high_img, evalfold_path, str(step) + "ir_l1")
            # R_high_img = torch.cat((vi_l, vi_l, vi_l),dim=1)
            fuse_image = tensor2numpy_R(fuse_val)
            visible_r = tensor2numpy_R(vi_r)
            # L_low_img = torch.cat((ir_l, ir_l, ir_l),dim=1)
            # L_low_img = ir_l
            infrared = tensor2numpy_L(ir)
            # L_high_img= torch.cat((ir_l, ir_l, ir_l),dim=1)
            # L_high_img = ir_l
            # L_high_img = tensor2numpy_L(L_high_img)
            save_pic(fuse_image, evalfold_path, name[0])
            # save_pic(visible_r, evalfold_path, str(step) + "vi_r")
            # save_pic(infrared, evalfold_path, str(step) + "ir")
            # save_pic(L_high_img, evalfold_path, str(step) + "ir_l1")

        # loss, loss_rec, loss_equal_R, loss_smooth = loss_function(R_low, R_high, L_low, L_high, I_low, I_high)
        # loss, recon_vis, recon_ir, smooth_loss, mc_loss  = loss_function(I_vi, I_ir, vi_r, vi_l, ir_l)
        val_total_loss += val_loss
        val_loss_ssim += val_loss_ssim
        val_loss_max += val_loss_max
        val_loss_color += val_loss_color
        val_loss_grad += val_loss_grad
        val_loss_l1 += val_loss_l1 
        data_loader.desc = "[val epoch {}] val_loss: {:.3f}  val_loss_ssim: {:.3f}  val_loss_max: {:.3f}  val_loss_color: {:.3f} val_loss_grad: {:.3f} val_loss_l1: {:.3f} lr: {:.6f}".format(epoch, val_total_loss.item() / (step + 1),
            val_loss_ssim.item() / (step + 1), val_loss_max.item() / (step + 1) / (step + 1),  val_loss_color.item() / (step + 1),val_loss_grad.item() / (step + 1), val_loss_l1.item() / (step + 1), lr)

    return val_total_loss.item() / (step + 1), val_loss_ssim.item() / (step + 1), val_loss_max.item() / (step + 1) / (step + 1),  val_loss_color.item() / (step + 1),val_loss_grad.item() / (step + 1), val_loss_l1.item() / (step + 1)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    # return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,300, eta_min=1e-6)

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)

def tensor2numpy_R(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_L(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3