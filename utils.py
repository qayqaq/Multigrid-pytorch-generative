# import os
# import shutil

# path = 'D:/秦傲洋/folding/demo_2l_plane'
# des1 = 'D:/秦傲洋/Multigrid_pytorch/data/plane/step1'
# des2 = 'D:/秦傲洋/Multigrid_pytorch/data/plane/step2'
# n = 10000001
# print(str(n)[-5:-1])
# for root, dirpath, filelist in os.walk(path):
#     if dirpath == []:
#         file1 = os.path.join(root, filelist[0])
#         file2 = os.path.join(root, filelist[1])
#         if os.path.isfile(file1):
#             shutil.copy(file1, des1)
#             file1 = os.path.join(des1, filelist[0])
#             file_new = os.path.join(des1, str(n)[-4:] + '.jpg')
#             os.rename(file1, file_new)
#         if os.path.isfile(file2):
#             shutil.copy(file2, des2)
#             file2 = os.path.join(des2, filelist[1])
#             file_new = os.path.join(des2, str(n)[-4:] + '.jpg')
#             os.rename(file2, file_new)
#         n += 1
import warnings
warnings.filterwarnings("ignore")


import torch
from PIL import Image
import numpy as np
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import skimage.io

def merge_images(images, space=0, mean_img=None):
    num_images = images.shape[0]
    canvas_size = int(np.ceil(np.sqrt(num_images)))
    h = images.shape[1]
    w = images.shape[2]
    canvas = np.zeros((canvas_size * h + (canvas_size-1) * space,  canvas_size * w + (canvas_size-1) * space, 3), np.uint8)

    for idx in range(num_images):
        image = images[idx,:,:,:]
        if mean_img:
            image += mean_img
        i = idx % canvas_size
        j = idx // canvas_size
        min_val = np.min(image)
        max_val = np.max(image)
        image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        canvas[j*(h+space):j*(h+space)+h, i*(w+space):i*(w+space)+w,:] = image
    return canvas


def save_images(images, file_name, space=0, mean_img=None):
    skimage.io.imsave(file_name, merge_images(images, space, mean_img))


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def fold(orig_im, fold_action, isplot, config):
    fold_x = fold_action[:, 0]
    fold_y = fold_action[:, 1]
    fold_angle = fold_action[:, 2]
# def fold(orig_im, fold_x, fold_y, fold_angle, isplot, config):
    '''
    :param orig_im: BS*3*img_size*img_size tensor
    :param fold_y: BS*1 tensor
    :param fold_angle: BS*1 tensor
    :return: final_folded_im BS*3*img_size*img_size tensor
    '''
    img_size = config.img_size
    pad = config.pad
    pad_size = int(img_size + pad * 2)  # pad后图片大小
    half_size = int(pad_size / 2)  # pad后图片大小的一半


    def rot_img(x, theta, dtype):
        rot_mat = torch.zeros((len(theta), 2, 3))
        rot_mat[:, 0, 0] = torch.cos(theta)
        rot_mat[:, 0, 1] = -torch.sin(theta)
        rot_mat[:, 1, 0] = torch.sin(theta)
        rot_mat[:, 1, 1] = torch.cos(theta)
        grid = F.affine_grid(rot_mat, x.size()).type(dtype)
        x = F.grid_sample(x, grid, padding_mode="border")
        return x

    def tran_img(x, dis, dtype):
        tran_mat = torch.zeros((len(dis), 2, 3))
        tran_mat[:, 0, 0] = tran_mat[:, 1, 1] = 1
        tran_mat[:, 0, 2] = -dis * 2 / pad_size
        grid = F.affine_grid(tran_mat, x.size()).type(dtype)
        x = F.grid_sample(x, grid, padding_mode="border")
        return x

    def mir_img(x, dis, dtype):
        mir_mat = torch.zeros((len(dis), 2, 3))
        mir_mat[:, 0, 0] = mir_mat[:, 1, 1] = 1
        mir_mat[:, 0, 2] = dis * 4 / pad_size
        mir_mat[:, 0, 0] = -1
        grid = F.affine_grid(mir_mat, x.size()).type(dtype)
        x = F.grid_sample(x, grid, padding_mode="border")
        return x

    fold_x_0 = fold_x - fold_y * torch.tan(fold_angle)

    transform1 = T.Compose([T.Resize((img_size, img_size)), T.Pad((pad, pad), fill=1)])

    # 求图片中心到折痕的垂直距离，也是后面平移的距离
    dis = (fold_x_0 - img_size / 2) * torch.cos(fold_angle.clone()) + img_size / 2 * torch.sin(fold_angle.clone())
    # 对图片做pad，否则平移时可能会出范围
    im = transform1(orig_im).type(dtype)
    # 对图片先以图片中心为旋转中心反方向即顺时针旋转theta，使折痕竖直
    rotated_im = rot_img(im, -fold_angle, dtype)
    # 对图片反方向平移距离dis，使竖直折痕平分图片
    tran_rotated_im = tran_img(rotated_im, -dis, dtype)
    # 对图片进行水平的镜像
    mir_tran_rotated_im = mir_img(tran_rotated_im, torch.zeros_like(dis), dtype)
    # 把原图片和镜像过后的图片贴在一起并保留相应的部分，用sigmoid函数代替不可微的torch.where(x>y,x,y)
    # folded_tran_rotated_im = torch.cat((torch.sigmoid((0.9 - mir_tran_rotated_im[:, :, :, :half_size]) * 1000) * (
    #                                     mir_tran_rotated_im[:, :, :, :half_size] - tran_rotated_im[:, :, :, :half_size]) + tran_rotated_im[:, :, :, :half_size],
    #                                     torch.sigmoid((rotated_im[:, :, :, half_size:pad_size] - 0.1) * 1000) * (
    #                                     torch.ones_like(rotated_im[:, :, :, half_size:pad_size]) - rotated_im[:, :, :, half_size:pad_size]) + rotated_im[:, :, :, half_size:pad_size]),
    #                                     dim=-1)

    xxx = torch.sigmoid((0.9 - mir_tran_rotated_im[:, :, :, :half_size]) * 100) * (
            mir_tran_rotated_im[:, :, :, :half_size] - tran_rotated_im[:, :, :, :half_size]) + tran_rotated_im[:, :, :, :half_size]

    # plt.figure()
    # plt.imshow(xxx.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    # plt.show()
    # yyy = torch.sigmoid((rotated_im[:, :, :, half_size:pad_size] - 0.1) * 1000) * (
    #         torch.ones_like(rotated_im[:, :, :, half_size:pad_size]) - rotated_im[:, :, :,
    #                                                                    half_size:pad_size]) + rotated_im[:, :, :,
    #                                                                                           half_size:pad_size]
    yyy = torch.ones_like(tran_rotated_im[:, :, :, half_size:pad_size])

    folded_tran_rotated_im = torch.cat((xxx, yyy), dim=-1)




    # 把图片平移回去
    folded_rotated_im = tran_img(folded_tran_rotated_im, dis, dtype)
    # 把图片旋转回去
    folded_im = rot_img(folded_rotated_im, fold_angle.clone(), dtype)
    # 把图片剪切回去
    final_folded_im = folded_im[:, :, pad:img_size + pad, pad:img_size + pad]




    if isplot:
        plt.figure()
        plt.subplot(331), plt.axis('off')
        plt.imshow(orig_im.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        plt.subplot(332), plt.axis('off')
        plt.imshow(im.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        plt.subplot(333), plt.axis('off')
        plt.imshow(rotated_im.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        plt.subplot(334), plt.axis('off')
        plt.imshow(tran_rotated_im.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        plt.subplot(335), plt.axis('off')
        plt.imshow(mir_tran_rotated_im.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        plt.subplot(336), plt.axis('off')
        plt.imshow(folded_tran_rotated_im.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        plt.subplot(337), plt.axis('off')
        plt.imshow(folded_rotated_im.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        plt.subplot(338), plt.axis('off')
        plt.imshow(folded_im.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        plt.subplot(339), plt.axis('off')
        plt.imshow(final_folded_im.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        plt.show()

        figure, ax = plt.subplots()
        ax.set_xlim(left=0, right=img_size), ax.set_ylim(img_size)
        x = fold_x_0.cpu().detach().numpy()
        theta = fold_angle.cpu().detach().numpy()
        l = 500
        plt.imshow(final_folded_im.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        line1 = [(x - l * math.sin(theta), -l * math.cos(theta)), (x + l * math.sin(theta), l * math.cos(theta))]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))
        plt.plot()
        plt.show()

    return final_folded_im


class config():
    def __init__(self, img_size, pad):
        self.img_size = img_size
        self.pad = pad


# n = 100001
# target_dir = 'D:\\秦傲洋\\Multigrid_pytorch\\data\\cifar10'
# fold_action = torch.Tensor([[40., 20., np.pi/6]])
# config = config(64, 50)
# resource_dir = 'D:\\秦傲洋\\Multigrid_learning\\data\\CIFAR-10-images-5000\\train\\'
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#         'dog', 'frog', 'horse', 'ship', 'truck']
# for cls in classes:
#     resource_dir = 'D:\\秦傲洋\\Multigrid_learning\\data\\CIFAR-10-images-5000\\train\\' + cls
    
#     for root, dirs, files in os.walk(resource_dir):
#         for file in files:

#             print(str(n)[-5:])
            
#             des_path_1 = os.path.join(target_dir, 'step1', str(n)[-5:] + '.jpg')
#             des_path_2 = os.path.join(target_dir, 'step2', str(n)[-5:] + '.jpg')
#             file_path = os.path.join(root, file)

            # img = Image.open(file_path)
            # img = T.Compose([T.Resize((64, 64))])(img)
            # img.save(des_path_1)
            # img = T.ToTensor()(img)
            # img = fold(img.unsqueeze(0), fold_action, 0, config).squeeze(0)
            # img = T.ToPILImage()(img)
            # img.save(des_path_2)
#             n += 1

# resource_dir = 'D:\\秦傲洋\\Multigrid_pytorch\\data\\cifar10\\step1'
# target_dir = 'D:\\秦傲洋\\Multigrid_pytorch\\data\\cifar10\\step2'
# for file in os.listdir(resource_dir):
#     img = Image.open(os.path.join(resource_dir, file))
#     img = T.Compose([T.Resize((64, 64))])(img)
#     img = T.ToTensor()(img)
#     img = fold(img.unsqueeze(0), fold_action, 1, config).squeeze(0)
#     img = T.ToPILImage()(img)
#     img.save(os.path.join(target_dir, file))
