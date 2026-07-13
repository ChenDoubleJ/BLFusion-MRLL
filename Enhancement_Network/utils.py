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
    train_visible = os.path.join(train_root, "vi")
    train_infrared= os.path.join(train_root, "ir")

    val_visible = os.path.join(val_root, "vi")
    val_infrared = os.path.join(val_root, "ir")
    # train_visible = os.path.join(train_root, "vi")
    # train_infrared= os.path.join(train_root, "ir")

    # val_visible = os.path.join(val_root, "low")
    # val_infrared = os.path.join(val_root, "low")
    train_visible_path = [os.path.join(train_visible, i) for i in os.listdir(train_visible)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path= [os.path.join(train_infrared, i) for i in os.listdir(train_infrared)
                  if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible, i) for i in os.listdir(val_visible)
                  if os.path.splitext(i)[-1] in supported]
    val_infrared_path= [os.path.join(val_infrared, i) for i in os.listdir(val_infrared)
                  if os.path.splitext(i)[-1] in supported]

    assert len(train_visible_path)==len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.format(len(train_visible_path),len(train_infrared_path))
    assert len(val_visible_path)==len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.format(len(val_visible_path),len(val_infrared_path))
    print("image pair check finish")

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

    return train_visible_path, train_infrared_path, val_visible_path, val_infrared_path

def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    # loss_function = Decom_Loss()
    loss_function = all_loss_function()

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    total_loss = torch.zeros(1).to(device)
    rec_vis_loss = torch.zeros(1).to(device)
    rec_ir_loss = torch.zeros(1).to(device)
    smooth_loss = torch.zeros(1).to(device)
    mc_loss = torch.zeros(1).to(device)
    Per_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_vi, I_ir, name = data

        if torch.cuda.is_available():
            I_vi = I_vi.to(device)
            I_ir = I_ir.to(device)
        I_ir = I_ir[:, :1, :, :]
        # I_input = torch.cat((I_vi, I_ir), dim=1)
        I_input = I_vi

        vi_r, vi_l = model(I_input)


        # loss, recon_vis, recon_ir, smooth_loss, mc_loss, Per_loss  = loss_function(I_vi, I_ir, vi_r, vi_l, ir_l)
        loss, recon_vis, smooth_loss, mc_loss, Per_loss  = loss_function(I_vi, I_ir, vi_r, vi_l)


        loss.backward()

        # accu_total_loss += loss.detach()
        # accu_rec_loss += loss_rec.detach()
        # accu_equal_R_loss += loss_equal_R.detach()
        # accu_smooth_loss += loss_smooth.detach()
            
        total_loss += loss.detach()
        rec_vis_loss += recon_vis.detach()
        # rec_ir_loss += recon_ir.detach()
        smooth_loss += smooth_loss.detach()
        mc_loss += mc_loss.detach()
        Per_loss += Per_loss.detach()


        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  Rec vis loss: {:.3f}  rec ir loss: {:.3f}  smooth loss: {:.3f} mc loss: {:.3f} Per loss: {:.3f}  lr: {:.6f}".format(epoch, total_loss.item() / (step + 1),
            rec_vis_loss.item() / (step + 1), rec_ir_loss.item() / (step + 1), smooth_loss.item() / (step + 1), mc_loss.item() / (step + 1), Per_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()
    lr_scheduler.step()
    
    return total_loss.item() / (step + 1), rec_vis_loss.item() / (step + 1), smooth_loss.item() / (step + 1), mc_loss.item() / (step + 1), Per_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    # loss_function = Decom_Loss()
    loss_function = all_loss_function()

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    model.eval()

    val_total_loss = torch.zeros(1).to(device)
    val_rec_vis_loss = torch.zeros(1).to(device)
    val_rec_ir_loss = torch.zeros(1).to(device)
    val_mc_loss = torch.zeros(1).to(device)
    val_smooth_loss = torch.zeros(1).to(device)
    val_Per_loss = torch.zeros(1).to(device)
    save_epoch = 60


    if torch.cuda.is_available():
        loss_function = loss_function.to(device)
    
    if epoch % save_epoch == 0 or epoch == 299:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_vi, I_ir, name = data

        if torch.cuda.is_available():
            I_vi = I_vi.to(device)
            I_ir = I_ir.to(device)
        I_ir = I_ir[:, :1, :, :]
        # input_val = torch.cat((I_vi, I_ir), dim=1)
        input_val = I_vi
        vi_r, vi_l = model(input_val)


        # vi_r, vi_l, ir_l = model(input_val)

        # loss, recon_vis, recon_ir, smooth_loss, mc_loss, Per_loss = loss_function(I_vi, I_ir, vi_r, vi_l, ir_l)
        loss, recon_vis, smooth_loss, mc_loss, Per_loss = loss_function(I_vi, I_ir, vi_r, vi_l)

        if epoch != 0 and (epoch % save_epoch == 0 or epoch == 299):
            R_high_img = torch.cat((vi_l, vi_l, vi_l),dim=1)
            # R_high_img = vi_l
            R_low_img = tensor2numpy_R(vi_r)
            R_high_img = tensor2numpy_R(R_high_img)
            # L_low_img = torch.cat((ir_l, ir_l, ir_l),dim=1)
            # L_low_img = ir_l
            # L_low_img = tensor2numpy_R(L_low_img)
            # L_high_img= torch.cat((ir_l, ir_l, ir_l),dim=1)
            # L_high_img = ir_l
            # L_high_img = tensor2numpy_L(L_high_img)
            save_pic(R_low_img, evalfold_path, name[0]+"vi_r")
            save_pic(R_high_img, evalfold_path,  name[0] + "vi_l")
            # save_pic(L_low_img, evalfold_path, name[0] + "ir_l")
            # save_pic(L_high_img, evalfold_path, str(step) + "ir_l1")


        # loss, loss_rec, loss_equal_R, loss_smooth = loss_function(R_low, R_high, L_low, L_high, I_low, I_high)
        # loss, recon_vis, smooth_loss, mc_loss, Per_loss  = loss_function(I_vi, I_ir, vi_r, vi_l)
        val_total_loss += loss
        val_rec_vis_loss += recon_vis
        # val_rec_ir_loss += recon_ir
        val_mc_loss += mc_loss
        val_smooth_loss += smooth_loss
        val_Per_loss += Per_loss

        data_loader.desc = "[val epoch {}] loss: {:.3f}  Rec vis loss: {:.3f}  Rec ir loss: {:.3f}  smooth loss: {:.3f} mc loss: {:.3f} Per loss: {:.3f} lr: {:.6f}".format(epoch, val_total_loss.item() / (step + 1),
            val_rec_vis_loss.item() / (step + 1), val_rec_ir_loss.item() / (step + 1) / (step + 1),  val_smooth_loss.item() / (step + 1),val_mc_loss.item() / (step + 1), val_Per_loss.item() / (step + 1), lr)

    return val_total_loss.item() / (step + 1), val_rec_vis_loss.item() / (step + 1), val_smooth_loss.item() / (step + 1),val_mc_loss.item() / (step + 1), val_Per_loss.item() / (step + 1),

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