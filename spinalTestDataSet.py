# author: 
# contact: onlydgi@foxmail.com
# datetime:2020/7/20 下午10:29
# software: PyCharm
"""
文件说明：test the data
"""
import  os
from skimage.measure import compare_ssim
import cv2
import tqdm
import numpy as np
from imgLabelPlot import dicom2array
from imgLabelPlot import dicom_metainfo


def dicom2Tlist(Testimg_root):
    """
    输入：存放测试集的地址，在每一个病人文件下面
    return:用户的ＵＩＤ和中间的地址
    """
    subfile = os.listdir(Testimg_root)
    subfile.sort(key=lambda x: int(x[5:]))
    testList = []
    for sub_file_name in subfile:
        subfile_path = os.path.join(Testimg_root, sub_file_name)
        # temp_res have three item: {"studyUid":001,"SOPInstaceUID":4.286.2,"img_dir":"./data/test.dic"}
        temp_res = dicoｍ2TestIdx(subfile_path)
        testList.append(temp_res)
    return testList


def dicoｍ2TestIdx(img_file):
    """
    input:some ill people's folder which contain the dicom img
    return:{"studyUid":001,"SOPInstaceUID":4.286.2,"img_dir":"./data/test.dic"}
    """
    img_fileName = os.listdir(img_file)
    # make the img list sort by id,like [img1.dcm,img2.dcm,img3.dcm,……]
    img_fileName.sort(key=lambda x: int(x[5:-4]))
    # split the file by dfferent seriesDescription
    # seriesNumber ,InstanceNumber, StudyInstanceUID,  other info tag :'0008|103e' '0002|0003'
    GroupElement=['0020|0011', '0020|0013', '0020|000d']
    res = []
    for i in range(1, len(img_fileName)+1):
        dicImg_path = os.path.join(img_file, img_fileName[i-1])
        # img_info have three item[]
        img_info = dicom_metainfo(dicImg_path, GroupElement)
        img_info.append(dicImg_path)
        print(img_info)
        res.append(img_info)
    return res



# def eucliDist(A, B):
#     return np.sqrt(sum(np.power((A - B), 2)[:][:]))
# a = eucliDist(np.array([[1,2],[3,4]]),np.array([1,1]))

def checkScore(img_arr_list, model_img):
    """

    :param img_arr_list: a image list of mid img
    :param model_img: template img for T2 key image
    :return: a list of score of image
    """
    scoreimg=[]
    for img in img_arr_list:
        new_img_arr = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        ssim = compare_ssim(new_img_arr, model_img)
        # eucliDist(new_img_arr, model_img)
        scoreimg.append(ssim)
    return scoreimg
# a = eucliDist([[1,2],[3,4]],[1,1])


def findT2keyFrame(imgList):
    """
    :param imgList: have three item: {"studyUid":001,"SOPInstaceUID":4.286.2,"img_dir":"./data/test.dic"}
    :return : each ill people have a list of mid key T2 infomation
    """
    #split the group by the
    GrouPeople = []
    for people in imgList:
        SOPIuid_0 = people[0][0]
        new_p = []
        sopGroup = []
        for p in people:
            SOPIuid = p[0]
            if SOPIuid != SOPIuid_0:
                new_p.append(sopGroup)
                sopGroup = []
                SOPIuid_0 = SOPIuid
            sopGroup.append(p)
        new_p.append(sopGroup)
        GrouPeople.append(new_p)
    Group_mid_people=[]
    # pick the mid img
    for newP in GrouPeople:
        mid_frame = []
        for grouInfo in newP:
            grouplength = len(grouInfo)
            # if grouplength < 3:
            #     continue
            mid_idx = grouplength // 2
            mid_frame.append(grouInfo[mid_idx])
        Group_mid_people.append(mid_frame)
    print(Group_mid_people)
    #  choice the T2 img
    model_img = cv2.imread("./data/oneExample/8.png", 0)
    socreList = []
    for midpeople in Group_mid_people:
        midimglist=[]
        for midimg in midpeople:
            fileroot = midimg[3]
            imgarr = dicom2array(fileroot)
            midimglist.append(imgarr)
        socre = checkScore(midimglist,model_img)
        socreList.append(socre)
    res=[]
    for score , imgList in zip(socreList, Group_mid_people):
        idx = score.index(min(score))
        res.append(imgList[idx])
    # return socreList,Group_mid_people
    return  res
# new_mid_ll２,mid_people = findT2keyFrame(ll2)


if __name__ == '__main__':
    ll2 = dicom2Tlist("./data/lumbar_train53/train")
    finalmid = findT2keyFrame(ll2)
    # finalmid.sort()
    # 测试准确率
    gt_inflist2 = np.load("./valid50_sort.npy", allow_pickle=True)
    count = 0
    compare_t2imgPath=[]
    for a, b in zip(finalmid, gt_inflist2):
        # compare_t2imgPath have the gt_t2img and pred_t2img
        flag = bool(a[3][2:] == b[0])
        compare_t2imgPath.append([flag, b[0], a[3]])
        if a[3][2:] == b[0]:
            print(a[3])
            count += 1
    print(count / 53)

