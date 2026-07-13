import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyDataSet, transform

from TDN_network import DecomNet as create_model
from utils import read_data, train_one_epoch, evaluate, create_lr_scheduler
import datetime
import transforms as T
from model.get_model import BSN

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.enabled = False

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/TDN_train_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_valloss = 1e5
    start_epoch = 0

    train_vi_l_path, train_vi_r_path, train_ir_path,val_vi_l_path, val_vi_r_path, val_ir_path = read_data(args.data_path)

    # data_transform = {
    #     "train": T.Compose([T.RandomCrop(128),
    #                         T.RandomHorizontalFlip(0.5),
    #                         T.RandomVerticalFlip(0.5),
    #                         T.ToTensor()]),

    #     "val": T.Compose([T.ToTensor()])}
    # data_transform = transform

    train_dataset = MyDataSet(train_vi_l_path,
                              train_vi_r_path,
                              train_ir_path,
                              transform=transform(mode="train"))  

    val_dataset = MyDataSet(val_vi_l_path,
                            val_vi_r_path,
                            val_ir_path,
                            transform=transform(mode="val"))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)

    model = BSN().to(device)

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5E-5)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_loss_ssim, train_loss_max, \
        train_loss_color, train_loss_grad, train_loss_l1, lr = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch)
        tb_writer.add_scalar("Train/train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("Train/train_loss_ssim", train_loss_ssim, epoch)
        tb_writer.add_scalar("Train/train_loss_max", train_loss_max, epoch)
        tb_writer.add_scalar("Train/train_loss_color", train_loss_color, epoch)
        tb_writer.add_scalar("Train/train_loss_grad", train_loss_grad, epoch)
        tb_writer.add_scalar("Train/train_loss_l1", train_loss_l1, epoch)
        tb_writer.add_scalar("Train/train_lr", lr, epoch)
        # validate
        # if epoch <=200:
        #     val_epoch = 30
        # else:
        #     val_epoch = 10
        val_epoch = 100
        if epoch != 0 and (epoch % val_epoch == 0 or epoch == 399): 
            val_loss, val_loss_ssim, val_loss_max, \
            val_loss_color, val_loss_grad, val_loss_l1 = evaluate(model=model,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=epoch, lr=lr, filefold_path=file_img_path)
            tb_writer.add_scalar("Val/val_loss", val_loss, epoch)
            tb_writer.add_scalar("Val/val_loss_ssim", val_loss_ssim, epoch)
            tb_writer.add_scalar("Val/val_loss_max", val_loss_max, epoch)
            tb_writer.add_scalar("Val/val_loss_color", val_loss_color, epoch)
            tb_writer.add_scalar("Val/val_loss_grad", val_loss_grad, epoch)
            tb_writer.add_scalar("Val/val_loss_l1", val_loss_l1, epoch)


            if val_loss < best_valloss:
                if args.use_dp == True:
                    save_file = {"model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch,
                                "args": args}
                else:
                    save_file = {"model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch,
                                "args": args}
                torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
                best_valloss = val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600) # 300
    parser.add_argument('--batch_size', type=int, default=16)  # 16  4

    parser.add_argument('--lr', type=float, default=0.001)  # default:0.0001
    parser.add_argument('--data_path', type=str,
                        default="dataset/realscene")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu_id', default='1', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()

    main(opt)
