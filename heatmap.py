# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_CAM(mask, x_crop):
        # 生成热力图
    x_pic = x_crop.copy
    heatmap = mask
    # heatmap = np.reshape(heatmap, (1,))
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.squeeze()
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    x_pic_numpy = x_pic.cpu().detach().numpy()
    # im2 = cv2.resize(x_pic, (mask.shape[0],
    #                          mask.shape[1]))
    x_pic_numpy = np.transpose(x_pic_numpy[0], (1, 2, 0))  # 将通道数放到最后一维，以便调整大小
    x_pic_resized = cv2.resize(x_pic_numpy, (mask.shape[1], mask.shape[0]))
    # x_pic_resized = np.transpose(x_pic_resized, (2, 0, 1))  # 转换回原来的维度顺序
    # x_pic_resized = torch.from_numpy(x_pic_resized).unsqueeze(0)  # 转换回 PyTorch 数据类型
    # 检查数据类型并转换
    if x_pic_resized.dtype != heatmap.dtype:
        x_pic_resized = x_pic_resized.astype(heatmap.dtype)

    cv2.addWeighted(heatmap, 0.5, x_pic_resized, 0.5, 0, heatmap)
    # 将OpenCV格式图像转换为RGB格式
    rgb_result = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 创建PIL图像对象
    pil_image = Image.fromarray(rgb_result)

    return pil_image




if __name__ == '__main__':
    pil_image = draw_CAM()
    # 在每一次siamese_track内调用draw_CAM，然后在保存result前运行以下
    heatmap_path = os.path.join('../results_imgs', 'heatmap', video.video_dir)
    if not isdir(heatmap_path): makedirs(heatmap_path)
    path = os.path.join(heatmap_path,"{:06d}.{}".format(f, 'png'))
    pil_image.save(path)
