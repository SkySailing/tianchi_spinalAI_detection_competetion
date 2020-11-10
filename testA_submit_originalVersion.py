# author: 
# contact: onlydgi@foxmail.com
# datetime:2020/8/9 下午3:58
# software: PyCharm
"""
文件说明：run for the testA50
"""
import json
import os
import time
from torch import nn

# old_path = os.getcwd()
# print(old_path)
# new_path = "/".join(old_path.split("/")[:-1])
# os.chdir(new_path)
# print(os.getcwd())

from dataProcessing.makeSubmitJson import pickeT2fromtestdata
from imgLabelPlot import dicom2array
from imgLabelPlot import dicom_metainfo

#coding=utf-8

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from fcn import FCN8s,VGGNet
from unet import unet_model, unet_parts
import random
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2

from trian import get_peak_points
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from imgLabelPlot import dicom_metainfo, dicom2array

config_test = dict()
config_test['showFlag'] = 0
config_test['tranform'] = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

config_test['lr'] = 0.000001
config_test['momentum'] = 0.9
config_test['weight_decay'] = 1e-4
config_test['start_epoch'] = 0
config_test['epoch_num'] = 600
config_test['batch_size'] = 16
config_test['save_freq'] = 10
config_test['sigma'] = 5.
# config_test['root_dir'] = r"../data/testB50.csv"
config_test['test_jsondir'] = r"./tcdata/round2_series_map.json"
config_test['test_img_dir'] = r"./tcdata/round2test"
config_test['checkout'] = r"UNet_double_PosiLoss_final_model.ckpt"
    # r"Unet_final_model.ckpt"

# MedicalDataAugmentationTool-VerSe-master/dataset.py'


normMean = [0.168036]
normStd = [0.177935]
normTransform = transforms.Normalize(normMean, normStd)
if config_test['tranform'] == 1:
    submitTestTransform = transforms.Compose([
        transforms.ToTensor()
        # ,normTransform
    ])
else:
    submitTestTransform=None

class myTestAdata(Dataset):
    def __init__(self, jsondir, img_dir, transforms=None):
        self.transform = transforms
        self.testImg = pickeT2fromtestdata(jsondir, img_dir)
            # pd.read_csv(img_csvdir,header=None).values

    def __getitem__(self, idx):
        imgdir = self.testImg[idx]
        img_arr = dicom2array(imgdir)        # 获取具体的图片数据，二维数据
        #  (hieght * width)
        origi_shape = np.array(img_arr.shape)
        if img_arr.shape != (256, 256):
            img_aug = cv2.resize(img_arr, (256, 256))
        else:
            img_aug = img_arr

        # 'studyUid','seriesUid','instanceUid','zindx'
        tag_list = ['0020|000d', '0020|000e', '0008|0018', '0020|0013']
        studyUid, seriesUid, instanceUid, zindx= dicom_metainfo(imgdir, tag_list)
        if self.transform is not None:
            img_aug = self.transform(img_aug)   # 在这里做transform，转为tensor等等
        # print(img_aug.shape)
        # print(img_aug)
        return origi_shape, img_aug, instanceUid, seriesUid, studyUid, zindx

    def __len__(self):
        return len(self.testImg)

def resize_pos(x1, y1, src_size, tar_size):
    """
    :param x1:
    :param y1:
    :param src_size: width * hight
    :param tar_size: width * hight
    :return:
    """
    w1 = src_size[1]
    h1 = src_size[0]
    w2 = tar_size[0]
    h2 = tar_size[1]
    y1 = np.array(y1).astype(np.float32)
    x1 = np.array(x1).astype(np.float32)
    h1 = np.array(h1).astype(np.float32)
    w1 = np.array(w1).astype(np.float32)
    h2 = np.array(h2).astype(np.float32)
    w2 = np.array(w2).astype(np.float32)
    # print("y1:", y1,"h1", h1, "h2", h2)
    # print("x1", x1,"w1", w1, "w2", w2)
    res_h = np.int(np.round(np.float64(y1 / h1 * h2)))
    res_w = np.int(np.round(np.float64(x1 / w1 * w2)))
    # print(x2, y2)
    return res_w, res_h

def plotPre_GT(img,gt,pred):
    j = 1
    for gtPoint in gt:
        cv2.circle(img, (gtPoint[0], gtPoint[1]), 8, thickness = 2, color=(0, 0, 255))
        cv2.putText(img, str(j), (gtPoint[0] - 10, gtPoint[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        j += 1
    i = 1
    for pre_point in pred:
        cv2.putText(img, str(i), (pre_point[0]+4, pre_point[1]), cv2.FONT_HERSHEY_SIMPLEX,  0.6, (255, 0, 0), 1)
        cv2.rectangle(img, (pre_point[0]-2, pre_point[1]-2), (pre_point[0]+2, pre_point[1]+2), (255, 0, 0), 2)
        i += 1
    plt.imshow(img*255.0)
    plt.show()
    return

def oneUid(posi_res, type_res, instanceUid_res, seriesUid_res, studyUid_res, zindx):
    onestudent = OrderedDict()
    onestudent["studyUid"] = studyUid_res
    onestudent["version"] = "v0.1"

    # make the data value
    student_data = OrderedDict()
    student_data["seriesUid"] = seriesUid_res
    student_data["instanceUid"] = instanceUid_res

    # make the annotation value
    annotation_data = OrderedDict()
    annotation_data["annotator"] = 72

    # make the data value
    data_point = OrderedDict()
    point_data = []

    # make the point data
    idx_identification = {0: 'T12-L1', 1: 'L1', 2: 'L1-L2', 3: 'L2', 4: 'L2-L3', 5: 'L3', 6: 'L3-L4', 7: 'L4',
                          8: 'L4-L5', 9: 'L5', 10: 'L5-S1'}

    for i in range(len(posi_res)):
        point = OrderedDict()
        point["coord"] = posi_res[i]
        tag = OrderedDict()
        if i in [0, 2, 4, 6, 8, 10]:
            tag['disc'] = type_res[i]
        else:
            tag["vertebra"] = type_res[i]
        tag["identification"] = idx_identification[i]
        point["tag"] = tag
        point["zIndex"] = int(zindx)
        point_data.append(point)
    data_point['point'] = point_data
    annotation_data["data"] = data_point
    student_data["annotation"] = [annotation_data]
    onestudent["data"] = [student_data]
    return onestudent

def makejson(pred_res_point,pred_res_type, instanceUid, pred_res_seriesUid, pred_res_studyUid, pred_res_zindx):
    sz = len(pred_res_type)
    res = []
    for i in range(sz):
        onestudy = oneUid(pred_res_point[i], pred_res_type[i], instanceUid[i], pred_res_seriesUid[i], pred_res_studyUid[i], pred_res_zindx[i])
        res.append(onestudy)
    json_str = json.dumps(res)
    cur_time_str = time.strftime("%Y_%m_%d_H%H_%M_%S",time.localtime(time.time()))
    print("success make json file at:", cur_time_str)
    # filename = "../data/" + 'submit_testB_' + cur_time_str + '.json'
    filename = "result.json"
    with open(filename, 'w') as json_file:
        json_file.write(json_str)
    return res

def random_pick(some_list, probabilities):
    """
    pick random elements by probabilities
    :param some_list:
    :param probabilities:
    """
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


if __name__ == '__main__':
    # load model from

    # net = unet_model.UNet(1, 11)
    # net = unet_model.UNet_twoPart(1, (6, 5))
    net = unet_model.UNet_double()

    # vgg_model = VGGNet(requires_grad = True, pretrained=False)
    # net = FCN8s(pretrained_net=vgg_model, n_class=11)

    net.float().cuda()
    net.eval()
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:8000', rank=0, world_size=1)
    # net = nn.parallel.DistributedDataParallel(net.float().cuda())
    if config_test['checkout'] != "":
        net.load_state_dict(torch.load(config_test['checkout']))
    else:
        print("choice the mode file please!")

    testDataset = myTestAdata(config_test['test_jsondir'], config_test['test_img_dir'], transforms=submitTestTransform)
    testDataLoader = DataLoader(testDataset, config_test['batch_size'], False)
    Loader_num = len(testDataLoader)
    all_pred_point = []
    # all res contain: keypoint,type_pred,instanceUid, seriesUid, studyUid
    all_pred_res = []
    all_pred_res_point = []
    all_pred_res_type = []
    all_pred_res_instanceUid = []
    all_pred_res_seriesUid = []
    all_pred_res_studyUid = []
    all_pred_res_zindx = []
    # random_type = ['v1', 'v2', 'v2', 'v2', 'v3', 'v2', 'v4', 'v2', 'v5', 'v2', 'v2']
    # make the type based on probability on statistics info
    typeList_disc = ['v1', 'v2',  'v3', 'v4',  'v5']
    prob_disc = [[30/53, 5/53, 2/53, 1/53, 15/53], [30/53, 5/53, 2/53, 1/53, 15/53], [23/53, 15/53, 8/53, 2/53, 5/53], [7/53, 19/53, 12/53, 2/53, 13/53], [5/53, 18/53, 19/53, 8/53, 3/53], [15/53, 15/53, 17/53, 4/53, 2/53]]
    typeList_vert = ['v1', 'v2']
    prob_vert = [0.1, 0.9]
    with torch.no_grad():
        for i, (origi_shape, img_aug, instanceUid, seriesUid, studyUid, zindx) in enumerate(testDataLoader):
            print("test batch ===== ", i)
            if config_test['tranform'] == 0:
                img_aug = img_aug[:, np.newaxis, :, :]
            img_aug = Variable(img_aug).float().cuda()
            # heatmaps_targets = Variable(distance_maps_normalized, requires_grad=True).float().cuda()
            # keyPsoi= Variable(keyPsoi).float().cuda()
            pred_heatmaps = net(img_aug)
            cur_shape = (256, 256)
            pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy())  # (N,15,2)
            batch_id = 0
            project_batch_point = []
            # (width × hieght)
            for batchPoint in pred_points:
                project_point_img = []
                # using the .shape to get the (hieght * width)
                orgi_img_shape = (origi_shape[batch_id][1], origi_shape[batch_id][0])
                # orgi_img_shape is （width , hight)
                # make the  tensor to ndarray ,and change(c,h,w) to (h,w,c)
                img = img_aug[batch_id].cpu().numpy().transpose(1, 2, 0)
                img = cv2.resize(img, orgi_img_shape)
                for point in batchPoint:
                    # point[0] is width
                    point_newWidth, point_newHight = resize_pos(point[0], point[1], cur_shape, orgi_img_shape)
                    project_point_img.append([point_newWidth, point_newHight])
                project_batch_point.append(project_point_img)
                batch_id += 1
                if config_test['showFlag'] == 1:
                    plotPre_GT(255*img, [], project_point_img)

            # all_pred_res.append([project_batch_point, random_type, instanceUid, seriesUid, studyUid])
            random_type = []
            for _ in range(len(studyUid)):
                temp_random_type = []
                for k in range(11):
                    if k&1:
                        # temp_random_type.append(random_pick(typeList_disc, prob_disc[i//2]))
                        temp_random_type.append(random.choice(typeList_disc))
                    else:
                        # temp_random_type.append(random_pick(typeList_vert, prob_vert))
                        temp_random_type.append(random.choice(typeList_vert))
                random_type.append(temp_random_type)

            all_pred_res_point.extend(project_batch_point)
            all_pred_res_type.extend(random_type)
            all_pred_res_instanceUid.extend(instanceUid)
            all_pred_res_seriesUid.extend(seriesUid)
            all_pred_res_studyUid.extend(studyUid)
            all_pred_res_zindx.extend(zindx)
            if config_test['showFlag'] == 1:
                print("pred  point:", project_batch_point)
            all_pred_point.extend(project_batch_point)
    makejson(all_pred_res_point, all_pred_res_type, all_pred_res_instanceUid, all_pred_res_seriesUid, all_pred_res_studyUid, all_pred_res_zindx)

