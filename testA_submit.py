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
from classi_models.resnet import resnet18,ResNet,BasicBlock
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


# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config_test['tranform'] = 1
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
config_test['checkout'] = r"UNet_double_PosiLoss_classi_100v20_final_model.ckpt"
config_test['disc_checkout'] = "disc_resNet_final0919_model.ckpt"
config_test['vert_checkout'] = "vert_resNet_final0919_model.ckpt"
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


class diffuse2D(object):
    """diffuse2D is use the nonlinear anisotropic diffusion filter to keep the edge and brighten img ,remove the noise of img.
    ``num_iter=5, delta_t=1 / 7, kappa=20, option=2``

    .. note::
        the this function before ToTensor

    Args:

    """

    def __init__(self, num_iter=5, delta_t=1 / 7, kappa=10, option=2):
        self.num_iter = num_iter
        self.delta_t = delta_t
        self.kappa = kappa
        self.option = option
        self.hN = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        self.hS = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        self.hE = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        self.hW = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        self.hNE = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]])
        self.hSE = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.hSW = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])
        self.hNW = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

    def fit(self, img):

        diff_im = img.copy()

        dx = 1;
        dy = 1;
        dd = np.sqrt(2)

        for i in range(self.num_iter):
            nablaN = cv2.filter2D(diff_im, -1, self.hN)
            nablaS = cv2.filter2D(diff_im, -1, self.hS)
            nablaW = cv2.filter2D(diff_im, -1, self.hW)
            nablaE = cv2.filter2D(diff_im, -1, self.hE)
            nablaNE = cv2.filter2D(diff_im, -1, self.hNE)
            nablaSE = cv2.filter2D(diff_im, -1, self.hSE)
            nablaSW = cv2.filter2D(diff_im, -1, self.hSW)
            nablaNW = cv2.filter2D(diff_im, -1, self.hNW)

            cN = 0;
            cS = 0;
            cW = 0;
            cE = 0;
            cNE = 0;
            cSE = 0;
            cSW = 0;
            cNW = 0

            if self.option == 1:
                cN = np.exp(-(nablaN / self.kappa) ** 2)
                cS = np.exp(-(nablaS / self.kappa) ** 2)
                cW = np.exp(-(nablaW / self.kappa) ** 2)
                cE = np.exp(-(nablaE / self.kappa) ** 2)
                cNE = np.exp(-(nablaNE / self.kappa) ** 2)
                cSE = np.exp(-(nablaSE / self.kappa) ** 2)
                cSW = np.exp(-(nablaSW / self.kappa) ** 2)
                cNW = np.exp(-(nablaNW / self.kappa) ** 2)
            elif self.option == 2:
                cN = 1 / (1 + (nablaN / self.kappa) ** 2)
                cS = 1 / (1 + (nablaS / self.kappa) ** 2)
                cW = 1 / (1 + (nablaW / self.kappa) ** 2)
                cE = 1 / (1 + (nablaE / self.kappa) ** 2)
                cNE = 1 / (1 + (nablaNE / self.kappa) ** 2)
                cSE = 1 / (1 + (nablaSE / self.kappa) ** 2)
                cSW = 1 / (1 + (nablaSW / self.kappa) ** 2)
                cNW = 1 / (1 + (nablaNW / self.kappa) ** 2)

            diff_im = diff_im + self.delta_t * (

                (1 / dy ** 2) * cN * nablaN +
                (1 / dy ** 2) * cS * nablaS +
                (1 / dx ** 2) * cW * nablaW +
                (1 / dx ** 2) * cE * nablaE +

                (1 / dd ** 2) * cNE * nablaNE +
                (1 / dd ** 2) * cSE * nablaSE +
                (1 / dd ** 2) * cSW * nablaSW +
                (1 / dd ** 2) * cNW * nablaNW
            )
        return diff_im

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return [self.fit(tensor[0]) for tensor in tensors]

    def __repr__(self):
        return self.__class__.__name__ + '(delta_t={0}, kappa={1})'.format(self.delta_t, self.kappa)

class myTestAdata(Dataset):
    def __init__(self, jsondir, img_dir, transforms=None):
        self.transform = transforms
        self.testImg = pickeT2fromtestdata(jsondir, img_dir)
        self.diffuse2D = diffuse2D()
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


        img_aug = self.diffuse2D.fit(img_aug)

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


def corpRectangele(img, point, CropSz):
    """

    Args:
        img:
        point:
        CropSz: wdith * hight

    Returns:

    """
    cropIMG = []
    for idx in range(len(point)):
        p = point[idx]
        w_corp, h_corp = CropSz
        w_center, h_center = p
        img_sz_h, img_sz_w = img.shape
        # the vert smaller 4 pixel in hight.
        # if idx & 1:
        #     h_corp = h_corp-4
        h_start = h_center - h_corp // 2
        h_end = h_center + h_corp // 2
        if h_start <= 0 or h_end >= img_sz_h:
            h_start = 0
            h_end = h_corp

        w_start = w_center - w_corp // 2
        w_end = w_center + w_corp // 2
        if w_start < 0 or w_end >= img_sz_w:
            w_start = 0
            w_end = w_corp
        new_img = img[h_start:h_end, w_start:w_end].copy()
        # new_img = new_img[:]
        # print(new_img)
        cropIMG.append(new_img)
        # cv2.rectangle(img,(w_center-w_corp//2,h_center-h_corp//2),(w_center+w_corp//2,h_center+h_corp//2),color=(0,0,255),thickness=2)
    # plt.imshow(255*img)
    # plt.show()
    # for i in range(len(cropIMG)):
    #     plt.imshow(cropIMG[i])
    #     plt.show()
    # cropIMG = np.array(cropIMG)
    return cropIMG

if __name__ == '__main__':
    # load model from

    # net = unet_model.UNet(1, 11)
    # net = unet_model.UNet_twoPart(1, (6, 5))
    net = unet_model.UNet_double()
    vert_net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2)
    disc_net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=5)


    # vgg_model = VGGNet(requires_grad = True, pretrained=False)
    # net = FCN8s(pretrained_net=vgg_model, n_class=11)

    net.float().cuda()
    net.eval()
    vert_net = vert_net.float().cuda()
    vert_net.eval()
    disc_net = disc_net.float().cuda()
    disc_net.eval()
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:8000', rank=0, world_size=1)
    # net = nn.parallel.DistributedDataParallel(net.float().cuda())
    if config_test['checkout'] != "":
        net.load_state_dict(torch.load(config_test['checkout']))
        vert_net.load_state_dict(torch.load(config_test['vert_checkout']))
        disc_net.load_state_dict(torch.load(config_test['disc_checkout']))
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
    idx2vertType = {'0': "v1", "1": "v2"}
    idx2discType = {'0': "v1", "1": "v2", "2": "v3", "3": "v4", "4": "v5"}
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
            pred_type = []
            for pred_point_idx in range(len(pred_points)):
                batchPoint = pred_points[pred_point_idx]
                temp_type = []
                project_point_img = []
                # using the .shape to get the (hieght * width)
                orgi_img_shape = (origi_shape[batch_id][1], origi_shape[batch_id][0])
                # orgi_img_shape is （width , hight)
                # make the  tensor to ndarray ,and change(c,h,w) to (h,w,c)
                img_original = img_aug[batch_id].cpu().numpy().transpose(1, 2, 0)
                img_fixSz = cv2.resize(img_original, (256, 256))
                img = cv2.resize(img_original, orgi_img_shape)
                # print("img shape:", img_fixSz.shape)
                cropIMG = corpRectangele(img_fixSz, batchPoint, [48, 30])
                for point_idx in range(len(batchPoint)):
                    point = batchPoint[point_idx]
                    # point[0] is width
                    point_newWidth, point_newHight = resize_pos(point[0], point[1], cur_shape, orgi_img_shape)
                    project_point_img.append([point_newWidth, point_newHight])
                    corpImg_input = cropIMG[point_idx]
                    if config_test['showFlag'] == 1:
                        plt.imshow(corpImg_input * 255.0)
                        plt.show()
                    corpImg_input = corpImg_input[np.newaxis,np.newaxis,:,:]
                    corpImg_input = torch.Tensor(corpImg_input).float().cuda()
                    # print("corpImg_inputsahpe:", corpImg_input.shape)
                    if point_idx&1:
                        pred_vertType = vert_net(corpImg_input)[0]
                        pred_vertType_idx = torch.argmax(pred_vertType, dim=0).cpu().numpy()
                        # print("pred_vertType:", pred_vertType_idx)
                        temp_type.append(idx2vertType[str(pred_vertType_idx)])
                    else:
                        pred_discType = disc_net(corpImg_input)[0]
                        pred_discType_idx = torch.argmax(pred_discType, dim=0).cpu().numpy()
                        # print("pred_discType:", pred_discType)
                        temp_type.append(idx2discType[str(pred_discType_idx)])

                pred_type.append(temp_type)
                # print(pred_type)
                project_batch_point.append(project_point_img)
                batch_id += 1
                if config_test['showFlag'] == 1:
                    plotPre_GT(255*img, [], project_point_img)

            all_pred_res_point.extend(project_batch_point)
            all_pred_res_type.extend(pred_type)
            all_pred_res_instanceUid.extend(instanceUid)
            all_pred_res_seriesUid.extend(seriesUid)
            all_pred_res_studyUid.extend(studyUid)
            all_pred_res_zindx.extend(zindx)
            if config_test['showFlag'] == 1:
                print("pred  point:", project_batch_point)
            all_pred_point.extend(project_batch_point)
    makejson(all_pred_res_point, all_pred_res_type, all_pred_res_instanceUid, all_pred_res_seriesUid, all_pred_res_studyUid, all_pred_res_zindx)

