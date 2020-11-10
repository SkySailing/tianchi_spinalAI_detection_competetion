# author: 
# contact: onlydgi@foxmail.com
# datetime:2020/7/24 下午11:15
# software: PyCharm
"""
文件说明：制作一个病人的，符合官方要求的提交数据集JSON格式
"""
import json
import numpy as np
from collections import defaultdict, OrderedDict


def oneUid(posi_res, type_res,instanceUid_res, seriesUid_res, studyUid_res, zindx):
    onestudent = OrderedDict()
    onestudent["studyUid"] = studyUid_res
    onestudent["version"] = "v0.1"


    # make the data value
    student_data = OrderedDict()
    student_data["seriesUid"] = seriesUid_res
    student_data["instanceUid"] = instanceUid_res


    #make the annotation value
    annotation_data = OrderedDict()
    annotation_data["annotator"] = 72


    #make the data value
    data_point = OrderedDict()
    point_data = []

    #make the point data
    idx_identification = {0: 'T12-L1', 1: 'L1', 2: 'L1-L2', 3: 'L2', 4: 'L2-L3', 5:'L3', 6: 'L3-L4', 7: 'L4', 8: 'L4-L5', 9: 'L5', 10: 'L5-S1'}

    for i in range(len(posi_res)):
        point = OrderedDict()
        point["coord"] = posi_res[i]
        tag = OrderedDict()
        if i in [0, 2, 4, 6, 8]:
            tag['disc'] = type_res[i]
        else:
            tag["vertebra"] = type_res[i]
        tag["identification"] = idx_identification[i]
        point["tag"] = tag
        point["zIndex"] = zindx
        point_data.append(point)
    data_point['point'] = point_data
    annotation_data["data"] = data_point
    student_data["annotation"] = annotation_data
    onestudent["data"] = student_data
    return onestudent



