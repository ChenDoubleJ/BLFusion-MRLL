import torch
from PIL import Image
from torchvision import transforms
from TDN_network import DecomNet as create_model
import numpy as np
import cv2
import os
from my_dataset import MyDataSet, transform,transform_test, MyDataSet_test
from utils import read_data, read_data_test
import transforms as T
from thop import profile
import time
from model.get_model import BSN

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = {
        "train": T.Compose([T.RandomCrop(128),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.ToTensor()])}
    train_vi_l_path, train_vi_r_path, train_ir_path, val_vi_l_path, val_vi_r_path, val_ir_path, val_vis_path= read_data_test("dataset/MRLL")

    val_dataset = MyDataSet_test(val_vi_l_path,
                            val_vi_r_path,
                            val_ir_path,
                            val_vis_path,
                            transform=transform_test(mode="val"))


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)
    # model = create_model().to(device)
    model = BSN().to(device)

    model_weight_path = "./weights/mul_noise/checkpoint_Diff_TDN.pth"
    # model_weight_path = "./experiments/TDN_train_20251204-205830/weights/checkpoint.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    model.eval()
    cnt = 0
    for img in val_loader:
        cnt += 1
        # img = Image.open(img_path)
        # img = resize(img)
        # img = data_transform(img)
        vi_l, vi_r, ir, vis, name = img
        # vis = vis.unsqueeze(0)
        # ir = ir.unsqueeze(0)
        if torch.cuda.is_available():
            vi_r = vi_r.to(device)
            ir = ir.to(device)
            vis = vis.to(device)
        fir = ir[:, :1, :, :]
        input = torch.cat((vi_r, fir), dim = 1)
        time1 = time.time()
        with torch.no_grad():
            fused = model.denoise1(vi_r, ir)
            # fused = model(vi_r, ir)
        time2 = time.time()
        # print(time2-time1)
        # img_vi = img_vi.squeeze(0).detach().cpu().numpy()
        # img_ir = img_ir.squeeze(0).detach().cpu().numpy()
        # img_vi = np.transpose(img_vi,(1,2,0))
        # img_ir = np.transpose(img_ir,(1,2,0))

        vi_r = vi_r.squeeze(0).detach().cpu().numpy()
        # vi_l = torch.cat([vi_l,vi_l,vi_l],dim=1)
        # vi_l = vi_l.squeeze(0).detach().cpu().numpy()
        # ir_l = torch.cat([ir_l,ir_l,ir_l],dim=1)
        # ir_l = ir_l.squeeze(0).detach().cpu().numpy()
        vi_r = np.transpose(vi_r,(1,2,0))
        # vi_l = np.transpose(vi_l,(1,2,0))
        # ir_l = np.transpose(ir_l,(1,2,0))
        fused = fused.squeeze(0).detach().cpu().numpy()
        fused = np.transpose(fused, (1, 2, 0))
        ir = ir.squeeze(0).detach().cpu().numpy() 
        ir = np.transpose(ir, (1, 2, 0))
        vis = vis.squeeze(0).detach().cpu().numpy()
        vis = np.transpose(vis, (1, 2, 0))
        # name=getnameindex(str(cnt))

        # name = str(cnt)
        savepic(fused, name[0], flag='fused')
        # savepic(ir, name[0], flag='ir')
        # savepic(vi_r, name[0], flag="vi_r")
        # savepic(vis, name[0], flag="vis")
        # savepic(img_vi, name[0], flag="img_vi")
        # savepic(img_ir, name[0], flag="img_ir")

def savepic(outputpic, name, flag):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]

    root = "./results/MRLL/"
    root_path = os.path.join(root, flag)

    if os.path.exists("./results") is False:
        os.makedirs("./results")
    if os.path.exists(root) is False:
        os.makedirs(root)
    if os.path.exists(root_path) is False:
        os.makedirs(root_path)
    path = root_path + "/{}.png".format(name)
    cv2.imwrite(path, outputpic)
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    print("complete compute {}.png and save".format(name))

def loadfiles(root):
    images_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]
    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    for index in range(len(images)):
        img_path = images[index]
        images_path.append(img_path)

    print("find {} images for computing.".format(len(images_path)))
    return images_path

def getnameindex(path):
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    path = path.replace("\\", "/")
    label = path.split("/")[-1].split(".")[0]
    return label

def resize(image):
    original_width, original_height = image.size

    new_width = original_width - (original_width % 8)
    new_height = original_height - (original_height % 8)
    resized_image = image.resize((new_width, new_height))
    return resized_image

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()