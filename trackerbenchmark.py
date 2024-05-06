from __future__ import division
import os
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config
from otbdatasets import DatasetFactory
from utils.pyvotkit.region import vot_overlap, vot_float2str

thrs = np.arange(0.3, 0.5, 0.05)

parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', default='E:/deep-learning/object_tracker/SiamMask-RBOCA/experiments/siammask_sharp/config_vot.json', help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='E:/deep-learning/object_tracker/SiamMask-RBOCA/experiments/siammask_sharp/snapshot_sonar_cpca/ccask30.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', default=True, action='store_true', help='whether use mask output')
parser.add_argument('--refine',default=True, action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2016', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization',default=True,dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')

if __name__ == '__main__':
    video_num = 0
    global args, logger, v_id
    args = parser.parse_args()
    cfg = load_config(args)

    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)
    dataset_root = 'H:/dataset/object_tracker/sonar/val/VOT2016'
    # dataset = DatasetFactory.create_dataset(name=args.dataset,
    #                                         dataset_root=dataset_root,
    #                                         load_img=False)

    # TODO：算法评测结果所在路径
    eval_result_path = 'E:/deep-learning/object_tracker/SiamMask-master/results/VOT'
    all_eval_trackers = os.listdir(eval_result_path)
    all_eval_trackers.sort()
    each_eval_tracker = [os.path.join(eval_result_path, eval_tracker) for eval_tracker in
                         all_eval_trackers]  # evaluation results of all trackers
    each_eval_tracker.sort()

    # choose one tracker
    one_tracker = each_eval_tracker[4]  #
    each_eval_result = os.listdir(one_tracker)
    each_eval_result.sort()  # ['Basketball.txt', 'Biker.txt', 'Bird1.txt', ... ,'Walking2.txt', 'Woman.txt']
    # print(each_eval_result)

    # each_eval_result_path = [os.path.join(one_tracker, eval_result) for eval_result in each_eval_result]
    # print(each_eval_result_path)
    #
    # ---------------------------------------------
    all_trakcers_result = []
    for tracker in each_eval_tracker:
        each_eval_result_path = [os.path.join(tracker, eval_result) for eval_result in each_eval_result]
        all_trakcers_result.append(each_eval_result_path)
    # print('all_trackers_result', all_trakcers_result)  # [['./Results_OTB100/CCOT/Basketball.txt', ... ,], ...]
    # ---------------------------------------------


    all_list = []  # all video of one tracker
    for num in range(0, len(all_trakcers_result)):
        one_list = []  # a video of one tracker
        for i in range(0,len(each_eval_result)):

            with open(all_trakcers_result[num][i]) as eval_result:  # choose a video of one tracker
                dataset = []
                lines = eval_result.readlines()

                # read datas in txt file, transform to String formation
                for line in lines:
                    temp1 = line.strip('\n')
                    temp2 = temp1.split('\t')
                    dataset.append(temp2)

                new_dataset = [new_line[0].split(',') for new_line in dataset]  # .split(',')按逗号分割字符串
                # print('new_dataset', new_dataset)
                # str转化成int型
                for i in range(0, len(new_dataset)):
                    for j in range(len(new_dataset[i])):
                        new_dataset[i][j] = int(float(new_dataset[i][j]))
                one_list.append(new_dataset)
        all_list.append(one_list)

    # print(all_list)
    #
    # every frame in a video
    # video_name = each_eval_result[0]
    # frames_list = video_name[:-4]
    # folder_path = os.path.join(dataset_root, frames_list)
    # frames_path = [os.path.join(dataset_root, frames_list,file_name) for file_name in os.listdir(folder_path)]

    # print(frames_path)
    for idx,video in enumerate(each_eval_result):
        video = video[:-4]
        folder_path = os.path.join(dataset_root, str(video))
        file_list = os.listdir(folder_path)
        frames_path = [os.path.join(dataset_root, video, img_name)for img_name in file_list if img_name.endswith('.jpg')]

        # TODO：画图结果保存路径
        dst_pic_path = os.path.join('../results_imgs', 'track_compare', video)

        # 判断是否有文件夹，如果没有则新建   如果有的话，说明已经生成过了。需要删除原文件后再次执行程序
        f = os.path.exists(dst_pic_path)
        if f is False:
            #
            os.makedirs(dst_pic_path)
            # show the tracking results
            for index, path in enumerate(frames_path):
                img = cv2.imread(path)
                im_show = img.copy()
                # print(img.shape)
                # --------------------------------------
                # results of trackers
                # all_list[a][b] 解释：a为某个算法，b为某个算法的某帧结果
                # a 对应的算法 : CCOT CFNet DaSiamRPN DeepSRDCF ECO fDSST GradNet MDNet Ours OursOld SiamDWfc SiamDWrpn SiamFC SiamRPN SRDCF Staple
                # TODO:选择想要画的算法结果，注意：需要改all_list[a][index]中的a这一项
                track_gt = all_list[0][idx][index]  # Staple
                track_gt_1 = all_list[1][idx][index]  # siamcar
                track_gt_2 = all_list[2][idx][index]  # siammask
                track_gt_3 = all_list[3][idx][index]  # ours
                track_gt_4 = all_list[4][idx][index]  # siamrpn
                track_gt_5 = all_list[5][idx][index]  # siamrpnpp
                # draw bounding boxes
                if len(track_gt_2) > 1 and len(track_gt_3) > 1  and len(track_gt_1) > 1 and len(track_gt_4) > 1 and len(track_gt_5) > 1:
                    cv2.polylines(im_show, [np.array(track_gt, int).reshape((-1, 1, 2))], True, (255,165,0), 3)  # 橙色
                    cv2.polylines(im_show, [np.array(track_gt_1, int).reshape((-1, 1, 2))], True, (255, 0, 0), 3)  # 蓝色
                    cv2.polylines(im_show, [np.array(track_gt_2, int).reshape((-1, 1, 2))], True, (0, 0, 255), 3)  # 红色
                    cv2.polylines(im_show, [np.array(track_gt_3, int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 绿色
                    cv2.polylines(im_show, [np.array(track_gt_4, int).reshape((-1, 1, 2))], True, (255, 0, 255), 3)  # 紫色
                    cv2.polylines(im_show, [np.array(track_gt_5, int).reshape((-1, 1, 2))], True, (0, 165, 255), 3)  # 黄色
                    cv2.putText(im_show, str(index), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video, im_show)
                    cv2.waitKey(1)

                # if len(track_gt_1)>1 and len(track_gt_2)>1 and len(track_gt_3)>1 and len(track_gt_4)>1:



                    cv2.imwrite(dst_pic_path + '/'  + '_img_{}.jpg'.format(index), im_show)
            # cv2.imshow('src_img', img)
            # cv2.waitKey(0)
        else:
            print('已经生成过({})，请删除原文件后重新执行程序'.format(video))



