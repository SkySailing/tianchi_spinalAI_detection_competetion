import glob
import os
from collections import defaultdict

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from augmentDataset import makeHeatmap
from imgLabelPlot import dicom_metainfo, dicom2array



def get_info(trainPath, jsonPath):
    """
    :rturn: 图像的位置和标注信息
    """
    # path = "./train150.npy" if "150" in trainPath else "./valid50.npy"
    # path = "./train203.npy" if "203" in trainPath else "./valid150.npy"
    # global config
    # if config["valid"] == 1:
    #     if "round2train" in trainPath:
    #         path="./round2train_pc.npy"
    #     if "round2_valid53" in trainPath:
    #         path = "./round2_valid53_pc.npy"
    # else:
    path = "noSuchfile"
    if not os.path.exists(path):
        annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
        # print(jsonPath)
        # print(os.getcwd())
        json_df = pd.read_json(jsonPath)
        for idx in tqdm(json_df.index):
            studyUid = json_df.loc[idx, "studyUid"]
            seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
            instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
            annotation = json_df.loc[idx, "data"][0]['annotation']
            row = pd.Series(
                {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
            annotation_info = annotation_info.append(row, ignore_index=True)
        dcm_paths = glob.glob(os.path.join(trainPath, "**", "**.dcm"))  # 具体的图片路径
        # print(dcm_paths)
        # 'studyUid','seriesUid','instanceUid'
        tag_list = ['0020|000d', '0020|000e', '0008|0018']
        # ["检查实例号：唯一标记不同检查的号码.", "序列实例号：唯一标记不同序列的号码.", "SOP实例"]
        dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
        for dcm_path in dcm_paths:
            try:
                studyUid, seriesUid, instanceUid = dicom_metainfo(dcm_path, tag_list)
                row = pd.Series(
                    {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})

                # print("try: ", dcm_path)
                dcm_info = dcm_info.append(row, ignore_index=True)
            except:
                # print("except: ", dcm_path)
                continue
        result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])
        # result = result.set_index('dcmPath')['annotation']  # 然后把index设置为路径，值设置为annotation
        result = result[["dcmPath", "annotation"]].values
        idx_wrong = []
        print("result len:",len(result))
        for idxx in range(len(result)):
            annotation = result[idxx][1]  # 获取图片的标签
            if len(annotation[0]['data']['point']) == 11:
                continue
            else:
                idx_wrong.append(idxx)
        result_new = np.delete(result, idx_wrong, 0)
        print("result_new len:", len(result_new))
        #delete some row which label wrong
        np.save(path, result_new)
    else:
        result_new = np.load(path, allow_pickle=True)
    return result_new

# label = [{'coord': [320, 76], 'tag': {'disc': 'v1', 'identification': 'T12-L1'}, 'zIndex': 6}, {'coord': [316, 95], 'tag': {'identification': 'L1', 'vertebra': 'v1'}, 'zIndex': 6}, {'coord': [316, 119], 'tag': {'disc': 'v1', 'identification': 'L1-L2'}, 'zIndex': 6}, {'coord': [312, 143], 'tag': {'identification': 'L2', 'vertebra': 'v1'}, 'zIndex': 6}, {'coord': [309, 169], 'tag': {'disc': 'v1', 'identification': 'L2-L3'}, 'zIndex': 6}, {'coord': [307, 187], 'tag': {'identification': 'L3', 'vertebra': 'v2'}, 'zIndex': 6}, {'coord': [300, 210], 'tag': {'disc': 'v4', 'identification': 'L3-L4'}, 'zIndex': 6}, {'coord': [297, 232], 'tag': {'identification': 'L4', 'vertebra': 'v2'}, 'zIndex': 6}, {'coord': [297, 257], 'tag': {'disc': 'v2', 'identification': 'L4-L5'}, 'zIndex': 6}, {'coord': [301, 280], 'tag': {'identification': 'L5', 'vertebra': 'v2'}, 'zIndex': 6}, {'coord': [302, 303], 'tag': {'disc': 'v3', 'identification': 'L5-S1'}, 'zIndex': 6}]
def splitLabelByVertDisc(label):
    vert_label = []
    disc_label = []

    vert_posi = []
    disc_posi = []
    vert_type = []
    disc_type = []
    # sz = len(label)
    # if  sz != 11:
    #     if sz > 11:
    #         label = label[0:11]
    #     else:
    #         for _ in range(sz-11):
    #             label.append(label[-2]]
    for llabel in label:
        label_tag = llabel['tag']
        if 'vertebra' in label_tag.keys() and label_tag["vertebra"] != '':
            vert_label.append([llabel["coord"],label_tag["vertebra"]])
        elif 'disc' in label_tag.keys() and label_tag["disc"] != '':
            disc_label.append([llabel['coord'],label_tag["disc"]])
        else:
            if 'vertebra' in label_tag.keys() and 'disc' in label_tag.keys():
                if '-' in label_tag["identification"]:
                    disc_label.append([llabel['coord'], 'v1'])
                else:
                    vert_label.append([llabel["coord"], 'v1'])
    vert_label.sort(key=lambda x: x[0][1])
    disc_label.sort(key=lambda x: x[0][1])
    for vertt in vert_label:
        vert_posi.append(vertt[0])
        vert_type.append(vertt[1])
    for discc in disc_label:
        disc_posi.append(discc[0])
        disc_type.append(discc[1])
    return vert_posi, vert_type, disc_posi, disc_type
# vert_label,disc_label,vert_posi, vert_type, disc_posi, disc_type = splitLabelByVertDisc(label)
def splitLabelByVertDisc_idx(label):
    vert_label = []
    disc_label = []

    vert_posi = []
    disc_posi = []
    vert_type = []
    disc_type = []
    vert_idx={'v1': 0, "v2": 1}
    disc_idx = {'v1': 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4}
    for llabel in label:
        label_tag = llabel['tag']
        if 'vertebra' in label_tag.keys() and label_tag["vertebra"] != '':
            vert_label.append([llabel["coord"], label_tag["vertebra"]])
        elif 'disc' in label_tag.keys() and label_tag["disc"] != '':
            disc_label.append([llabel['coord'], label_tag["disc"]])
        else:
            if 'vertebra' in label_tag.keys() and 'disc' in label_tag.keys():
                if '-' in label_tag["identification"]:
                    disc_label.append([llabel['coord'], 'v1'])
                else:
                    vert_label.append([llabel["coord"], 'v1'])
    vert_label.sort(key=lambda x: x[0][1])
    disc_label.sort(key=lambda x: x[0][1])
    for vertt in vert_label:
        vert_posi.append(vertt[0])
        vert_type.append(vertt[1])
    for discc in disc_label:
        disc_posi.append(discc[0])
        disc_type.append(discc[1])
    vert_type_idx = []
    disc_type_idx = []
    for vertTypr in vert_type:
        if ',' in vertTypr:
            vert_type_idx.append(vert_idx[vertTypr.split(',')[0]])
        else:
            vert_type_idx.append(vert_idx[vertTypr])
    for discTypr in disc_type:
        if ',' in discTypr:
            disc_type_idx.append(disc_idx[discTypr.split(',')[0]])
        else:
            disc_type_idx.append(disc_idx[discTypr])
    # print(vert_type_idx)
    # print(disc_type_idx)
    one_hot_vert_type = torch.nn.functional.one_hot(torch.LongTensor(np.array(vert_type_idx)), 2)
    one_hot_disc_type = torch.nn.functional.one_hot(torch.LongTensor(np.array(disc_type_idx)), 5)
    return vert_posi, one_hot_vert_type, disc_posi, one_hot_disc_type

def splitLabelByVertDisc_old(label, shape):
    """
    :param label: 某一个图像的标注信息
    :param shape: 图像原来的尺寸
    :return:
    """
    vertebra_label = {}
    disc_label = {}
    vert_idx = {'L1': 0, 'L2': 1, 'L3': 2,  'L4': 3, 'L5': 4}
    disc_idx = {'T12-L1': 0, 'L1-L2': 1, 'L2-L3': 2, 'L3-L4': 3, 'L4-L5': 4, 'L5-S1': 5}

    idx_vert = {0: 'L1', 1: 'L2', 2: 'L3', 3: 'L4', 4: 'L5'}
    idx_disc = {0: 'T12-L1', 1: 'L1-L2', 2: 'L2-L3', 3: 'L3-L4', 4: 'L4-L5', 5: 'L5-S1'}

    vert_posi = []
    disc_posi = []

    vert_type = []
    disc_type = []

    for llabel in label:
        coord = llabel['coord']
        center_x, center_y = coord[0], coord[1]

        # normal_center_x , normal_center_y = coord[0]/shape[0], coord[1]/shape[1]
        # new_coord = [normal_center_x, normal_center_y]

        identification = llabel['tag']['identification']
        llabel_tag = llabel["tag"]
        llabel_zindex = llabel["zIndex"]
        if "vertebra" in llabel['tag'].keys() and llabel_tag["vertebra"] != '':
            vertebra_label[identification] = {"posi": coord, "type": llabel_tag["vertebra"]}
        else:
            disc_label[identification] = {"posi": coord, "type": llabel_tag["disc"]}
    # if('T12-L1' not in disc_label.keys()):
    #     print("vertebra_label:", vertebra_label)
    #     print("disc_label:", disc_label)

    for i in range(len(idx_vert)):
        vert_label = vertebra_label[idx_vert[i]]
        vert_posi.append(vert_label["posi"])
        vert_type.append(vert_label["type"])
    for j in range(len(idx_disc)):
        disc_llabel = dict()
        if j == 0:
            if 'T12-L1' not in disc_label.keys():
                disc_name = disc_label.keys()
                for name in disc_name:
                    if "T12" in name:
                        disc_llabel = disc_label[name]
            else:
                disc_llabel = disc_label[idx_disc[j]]
        else:
            disc_llabel = disc_label[idx_disc[j]]
        disc_posi.append(disc_llabel["posi"])
        disc_type.append(disc_llabel["type"])
    # # return llabel_zindex, vertebra_label, disc_label
    # print("vert_posi:",vert_posi)
    # print("disc_posi:", disc_posi)
    return vert_posi, vert_type, disc_posi, disc_type
# vert_posi2, vert_type2, disc_posi2, disc_type2 = splitLabelByVertDisc(label,256)
def makeChanelHeatmap(oldheatmap):
    newHeatmap = np.zeros((11, 256, 256))
    for c in range(11):
        for i in range(256):
            for j in range(256):
                newHeatmap[c][i][j] = oldheatmap[i][j][c]
    return newHeatmap

class diffuse2D(object):
    """diffuse2D is use the nonlinear anisotropic diffusion filter to keep the edge and brighten img ,remove the noise of img.
    ``num_iter=5, delta_t=1 / 7, kappa=20, option=2``

    .. note::
        the this function before ToTensor

    Args:

    """

    def __init__(self, num_iter=5, delta_t=1 / 7, kappa=20, option=2):
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


def corpRectangele(img, point, CropSz):
    """

    Args:
        img:
        point:
        CropSz: wdith * hight

    Returns:

    """
    cropIMG_vert = []
    cropIMG_disc = []
    for idx in range(len(point)):
        p = point[idx]
        w_corp, h_corp = CropSz
        w_center, h_center = p
        # the vert smaller 4 pixel in hight.
        # if idx & 1:
        #     h_corp = h_corp-4
        h_start = h_center - h_corp // 2
        h_end = h_center + h_corp // 2
        if h_start < 0:
            h_start = 0
            h_end = h_corp
        w_start = w_center - w_corp // 2
        w_end = w_center + w_corp // 2
        new_img = img[h_start:h_end, w_start:w_end].copy()
        # new_img = new_img[:]
        # print(new_img)
        if idx & 1:
            cropIMG_vert.append(new_img)
        else:
            cropIMG_disc.append(new_img)
        # cv2.rectangle(img,(w_center-w_corp//2,h_center-h_corp//2),(w_center+w_corp//2,h_center+h_corp//2),color=(0,0,255),thickness=2)
    # plt.imshow(255*img)
    # plt.show()
    # for i in range(len(cropIMG)):
    #     plt.imshow(cropIMG[i])
    #     plt.show()
    cropIMG_vert = np.array(cropIMG_vert)
    cropIMG_disc = np.array(cropIMG_disc)
    return cropIMG_vert, cropIMG_disc

class mydata(Dataset):
    def __init__(self, root_dir, label_dir, transforms=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.result = get_info(root_dir, label_dir)
        self.transform = transforms
        self.diffuse2D = diffuse2D()

    def __getitem__(self, idx):
        """
        :param idx:
        :return: type_label and keyPsoi are sorted by the position from up to down
        """
        img_dir = self.result[idx][0]         # 获取图片的地址
        img_arr = dicom2array(img_dir)        # 获取具体的图片数据，二维数据
        #  (hieght * width)
        origi_shape = np.array(img_arr.shape)
        # if len(img_arr.shape) == 2:
        #     img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        annotation = self.result[idx][1]   # 获取图片的标签
        label = annotation[0]['data']['point']
        # print(len(label))
        # print(img_dir)
        # print(label)
        vert_posi, vert_type, disc_posi, disc_type = splitLabelByVertDisc(label)
        # origi_keyposi = np.array(vert_posi + disc_posi)
        origi_keyposi = []
        sz = len(vert_posi)
        # print("vert_posi len", len(vert_posi))
        for i in range(sz):
            origi_keyposi.append(np.array(disc_posi[i]))
            origi_keyposi.append(np.array(vert_posi[i]))
        origi_keyposi.append(disc_posi[-1])
        # print("origi_keyposi",len(origi_keyposi))
        origi_keyposi= np.array(origi_keyposi)
        # print("origi_keyposi:", origi_keyposi)

        # zidex, vert_label, disc_label = splitLabelByVertDisc(label, img_arr.shape)
        # targetSize = (256,256)
        # if img_arr.shape != targetSize:
        #     img_arr = cv2.resize(img_arr, targetSize, interpolation=cv2.INTER_LINEAR)
        # return img_arr, zidex, vert_label, disc_label
        # return img_arr, vert_posi, vert_type, disc_posi, disc_type
        img_aug, keyPsoi, distance_maps_normalized = makeHeatmap(img_arr, origi_keyposi)
        # distance_maps_normalized = makeChanelHeatmap(distance_maps_normalized)
        # type_label = vert_type + disc_type
        type_label = []
        sz = len(vert_type)
        for i in range(sz):
            type_label.append(disc_type[i])
            type_label.append(vert_type[i])
        type_label.append(disc_type[-1])

        ### diffuse the img

        # img_aug = self.diffuse2D.fit(img_aug)

        if self.transform is not None:
            img_aug = self.transform(img_aug)   # 在这里做transform，转为tensor等等
        # print(img_aug.shape)
        # print(img_aug)

        return origi_shape, img_aug, keyPsoi, distance_maps_normalized, type_label, origi_keyposi

    def __len__(self):
        return len(self.result)


class mydata_classifi(Dataset):
    def __init__(self, root_dir, label_dir, transforms=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.result = get_info(root_dir, label_dir)
        self.transform = transforms
        self.diffuse2D = diffuse2D()

    def __getitem__(self, idx):
        """
        :param idx:
        :return: type_label and keyPsoi are sorted by the position from up to down
        """
        img_dir = self.result[idx][0]  # 获取图片的地址
        img_arr = dicom2array(img_dir)  # 获取具体的图片数据，二维数据
        #  (hieght * width)
        origi_shape = np.array(img_arr.shape)
        # if len(img_arr.shape) == 2:
        #     img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        annotation = self.result[idx][1]  # 获取图片的标签
        label = annotation[0]['data']['point']
        # print(len(label))
        # print(img_dir)
        # print(label)
        vert_posi, vert_type, disc_posi, disc_type = splitLabelByVertDisc_idx(label)
        # origi_keyposi = np.array(vert_posi + disc_posi)
        origi_keyposi = []
        sz = len(vert_posi)
        # print("vert_posi len", len(vert_posi))
        for i in range(sz):
            origi_keyposi.append(np.array(disc_posi[i]))
            origi_keyposi.append(np.array(vert_posi[i]))
        origi_keyposi.append(disc_posi[-1])
        # print("origi_keyposi",len(origi_keyposi))
        origi_keyposi = np.array(origi_keyposi)
        # print("origi_keyposi:", origi_keyposi)

        img_aug, keyPsoi, distance_maps_normalized = makeHeatmap(img_arr, origi_keyposi)

        type_label = []
        sz = len(vert_type)
        for i in range(sz):
            type_label.append(disc_type[i])
            type_label.append(vert_type[i])
        type_label.append(disc_type[-1])

        ### diffuse the img

        img_aug = self.diffuse2D.fit(img_aug)

        cropIMG_vert, cropIMG_disc = corpRectangele(img_aug, keyPsoi, [48, 30])

        if self.transform is not None:
            img_aug = self.transform(img_aug)  # 在这里做transform，转为tensor等等
            cropIMG_vert = self.transform(cropIMG_vert.transpose((1, 2, 0)))
            cropIMG_disc = self.transform(cropIMG_disc.transpose((1, 2, 0)))
        # print(img_aug.shape)
        # print(img_aug)

        return origi_shape, img_aug, keyPsoi, distance_maps_normalized, origi_keyposi, vert_type, cropIMG_vert, disc_type, cropIMG_disc

    def __len__(self):
        return len(self.result)



if __name__ == '__main__':
    root_dir = r"../tcdata/round2train"
        # r"data/lumbar_train53/train"　data/lumbar_train150/
    label_dir = r"../tcdata/round2train_checked.json"
        # r"data/lumbar_train53/lumbar_train51_annotation.json"data/lumbar_train150/lumbar_train150_annotation.json
    sipnalDataset = mydata_classifi(root_dir, label_dir)
    data = sipnalDataset[0]
    print(data)
    #cv2.circle(data[0], (center_x, center_y), 8, (127, 127, 255))
    cv2.imshow("Image", 255.0*data[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 编号 index = 1 的标注

# result len: 150
# result_new len: 149
# [0, 0, 1, 1, 1]
# [0, 0, 0, 3, 1, 2]
# (array([512, 512]), array([[8.8761166 , 8.90754472, 8.93847574, ..., 8.49526563, 8.34375348,
#         8.30196242],
#        [8.80645185, 8.84754931, 8.90263018, ..., 8.55999132, 8.3872831 ,
#         8.33665583],
#        [8.67709609, 8.738119  , 8.84379844, ..., 8.70999341, 8.47785966,
#         8.4024143 ],
#        ...,
#        [9.57987726, 9.53767959, 9.42998785, ..., 9.42958115, 9.64383006,
#         9.72436747],
#        [9.506685  , 9.45933537, 9.35697839, ..., 9.30692638, 9.44674203,
#         9.498218  ],
#        [9.49249771, 9.44162195, 9.33751972, ..., 9.26283451, 9.3656628 ,
#         9.40278041]]), array([[160,  38],
#        [158,  48],
#        [158,  60],
#        [156,  72],
#        [154,  84],
#        [154,  94],
#        [150, 105],
#        [148, 116],
#        [148, 128],
#        [150, 140],
#        [151, 152]]), array([[[0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.]],
#
#        [[0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.]],
#
#        [[0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.]],
#
#        ...,
#
#        [[0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.]],
#
#        [[0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.]],
#
#        [[0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.],
#         [0., 0., 0., ..., 0., 0., 0.]]]), array([[320,  76],
#        [316,  95],
#        [316, 119],
#        [312, 143],
#        [309, 169],
#        [307, 187],
#        [300, 210],
#        [297, 232],
#        [297, 257],
#        [301, 280],
#        [302, 303]]), tensor([[1, 0],
#         [1, 0],
#         [0, 1],
#         [0, 1],
#         [0, 1]]), array([[[ 23.16898467,  19.08279581,  17.16063331, ...,  47.01030707,
#           42.84412133,  42.30461873],
#         [ 23.44650576,  19.10456395,  17.0030614 , ...,  48.90810147,
#           44.79430972,  44.43456315],
#         [ 23.37992334,  18.86989598,  16.50606987, ...,  49.31899741,
#           45.73036473,  45.47148668],
#         ...,
#         [ 40.54207937,  31.90022181,  23.09032658, ...,  39.33368371,
#           35.27104112,  37.09731238],
#         [ 36.14608484,  29.0297468 ,  21.91153977, ...,  37.59735128,
#           31.57099731,  31.66953767],
#         [ 32.36746587,  26.63996248,  21.01034755, ...,  37.021154  ,
#           29.02944594,  27.37159467]],
#
#        [[ 60.5545378 ,  57.59954897,  48.98704671, ...,  53.20130095,
#           50.6804553 ,  47.92925172],
#         [ 58.89757814,  55.39100061,  46.70547637, ...,  52.91568307,
#           49.9419826 ,  45.52361616],
#         [ 56.48441565,  52.32213323,  44.11999714, ...,  52.63619306,
#           49.28186009,  42.29896039],
#         ...,
#         [ 29.87770016,  24.55358225,  21.88499016, ...,  30.80269241,
#           26.75338174,  27.59126006],
#         [ 27.85822819,  23.80553685,  22.00228247, ...,  27.59215596,
#           23.32907843,  23.44629434],
#         [ 26.33956417,  23.25032736,  21.93080309, ...,  25.39656419,
#           21.33550054,  21.06841005]],
#
#        [[ 61.19273955,  53.83861027,  37.1655627 , ...,  46.46630707,
#           44.69348065,  44.53260119],
#         [ 60.82641643,  52.95384768,  36.41728708, ...,  46.40252804,
#           43.79512544,  42.93940574],
#         [ 60.22515606,  51.50462941,  35.26830937, ...,  46.62616188,
#           42.94110921,  40.77004348],
#         ...,
#         [ 57.91809174,  61.08316312,  64.39521683, ...,  26.90323801,
#           24.07421174,  25.210863  ],
#         [ 61.43440067,  64.96660103,  68.6035611 , ...,  24.38943957,
#           21.75765866,  22.29336803],
#         [ 64.39584609,  67.79302241,  70.7311821 , ...,  22.65635928,
#           20.67525443,  21.10745378]],
#
#        [[ 47.33957634,  48.37499308,  46.09867532, ...,  67.70971788,
#           60.10895119,  54.08904889],
#         [ 46.51922011,  48.51563488,  46.84374856, ...,  66.07751168,
#           59.42120534,  54.24899966],
#         [ 46.4954102 ,  49.11393043,  47.7508755 , ...,  64.48767824,
#           59.00395923,  55.01290572],
#         ...,
#         [106.76813589, 105.77730654,  62.27654234, ...,  49.92862319,
#           47.19259275,  42.75388815],
#         [105.98762724, 104.27009979,  78.35089111, ...,  45.71595761,
#           42.76318801,  38.56242904],
#         [104.97151114, 103.60515564,  97.97399164, ...,  41.68602798,
#           38.37206241,  34.11681914]],
#
#        [[ 72.33134209,  40.03866107,  30.0823539 , ...,  51.30892868,
#           45.63376683,  44.46922226],
#         [ 64.25882807,  35.45551919,  26.45234198, ...,  49.24464587,
#           43.27494762,  40.81692744],
#         [ 59.93116534,  32.81279073,  24.16733131, ...,  46.32114696,
#           40.87986193,  37.90459021],
#         ...,
#         [103.45145181, 101.64092746,  99.37768073, ..., 102.02566792,
#           67.66518898,  32.84094969],
#         [104.43424338, 102.55111942, 100.01671692, ..., 119.96276565,
#           95.00044935,  38.87800801],
#         [105.18666459, 103.23685717, 100.46835154, ..., 133.64390539,
#          117.83803855,  59.36105573]]]), tensor([[1, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0]]), array([[[ 15.69326234,  14.89900467,  13.69920336, ...,  28.52546527,
#           27.17303591,  28.77736938],
#         [ 14.97911569,  14.00709143,  12.87099021, ...,  27.61216111,
#           25.81543061,  26.69072726],
#         [ 14.54953799,  13.3782244 ,  12.29542959, ...,  26.51300459,
#           24.33387565,  24.3967908 ],
#         ...,
#         [ 24.00442127,  18.12617907,  14.66613031, ...,  27.81174901,
#           27.23363635,  28.26403276],
#         [ 24.83998585,  18.57158156,  14.98863753, ...,  31.24483581,
#           31.08110763,  32.03487419],
#         [ 25.17371438,  18.680711  ,  15.06018674, ...,  36.91270894,
#           36.21231643,  36.1543525 ]],
#
#        [[ 26.57936979,  20.1009387 ,  16.51155719, ...,  46.72012643,
#           39.86093388,  32.4691937 ],
#         [ 29.3921175 ,  22.02617787,  17.65057973, ...,  47.57453123,
#           39.88308983,  32.06776076],
#         [ 33.06041884,  24.59153436,  19.20300908, ...,  46.94166494,
#           38.4583755 ,  30.77618463],
#         ...,
#         [ 24.62803582,  20.01113691,  17.74121023, ...,  26.25486719,
#           26.80277966,  28.16453417],
#         [ 24.26630853,  19.29581787,  16.87527663, ...,  30.29208236,
#           30.92181964,  32.04594211],
#         [ 23.38161994,  18.25838673,  15.76047398, ...,  35.51367676,
#           35.44736993,  35.84740997]],
#
#        [[ 50.478738  ,  49.18544277,  43.42187975, ...,  51.63264576,
#           48.96824902,  44.86710358],
#         [ 51.83094918,  50.06617516,  44.01356997, ...,  48.87250887,
#           45.23829025,  40.05544976],
#         [ 53.29979434,  51.09928219,  44.72258607, ...,  44.99101372,
#           39.54240978,  32.92917562],
#         ...,
#         [ 19.78030164,  16.07091976,  14.13881035, ...,  39.10905261,
#           39.63276907,  39.80529159],
#         [ 19.86204048,  16.06123211,  14.12042788, ...,  44.55179318,
#           44.34756769,  44.06707779],
#         [ 20.81524752,  16.87995377,  14.90219827, ...,  47.62212441,
#           47.17487443,  46.78046334]],
#
#        [[ 59.90082346,  51.04686465,  42.69782107, ...,  55.37131536,
#           51.89154602,  49.74922519],
#         [ 56.58494906,  46.65717846,  38.7416087 , ...,  54.21660769,
#           50.50568453,  47.89414392],
#         [ 53.09452624,  43.3601174 ,  35.94479583, ...,  52.15569529,
#           47.46741406,  43.82811447],
#         ...,
#         [ 88.01350084,  87.89608999,  86.03163485, ...,  51.72351007,
#           50.18774821,  49.19457566],
#         [ 90.13414785,  88.75757105,  84.59657152, ...,  60.72525   ,
#           57.20825869,  54.0013552 ],
#         [ 90.59390153,  87.56700944,  79.38259939, ...,  63.1001827 ,
#           59.94606481,  56.67024792]],
#
#        [[ 76.06824643,  74.7597229 ,  71.49672361, ...,  48.18303108,
#           41.06015141,  33.56733947],
#         [ 80.47698371,  78.18268547,  75.62033488, ...,  44.5778936 ,
#           37.49531948,  32.54032061],
#         [ 83.49920534,  81.41271477,  79.89215915, ...,  43.25910937,
#           37.2084636 ,  34.29058207],
#         ...,
#         [ 98.4272274 ,  97.6270097 ,  96.14788588, ...,  57.33909924,
#           55.11710919,  60.8721554 ],
#         [ 98.03604337,  97.43864703,  96.05787807, ...,  62.31263799,
#           53.21074111,  50.85156721],
#         [ 97.90773321,  97.44362431,  96.11522753, ...,  72.03252774,
#           54.81166511,  47.84590066]],
#
#        [[ 91.33898855,  82.91019095,  51.30518585, ...,  63.72635346,
#           65.16530957,  66.34100934],
#         [ 91.74787869,  83.013791  ,  52.26673492, ...,  68.17816843,
#           69.42492661,  70.00707754],
#         [ 91.9498921 ,  82.79251171,  52.49276053, ...,  71.33299789,
#           73.84472207,  74.56540647],
#         ...,
#         [ 68.85888238,  69.72481488,  70.1825059 , ...,  97.68676611,
#          113.63396689, 129.10763427],
#         [ 69.14970747,  70.29751201,  70.69533405, ..., 100.38853977,
#          110.51878194, 122.72023802],
#         [ 70.94611203,  72.53590912,  72.82121884, ..., 102.04514764,
#          109.14573429, 117.33450419]]]))
