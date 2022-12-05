# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:12:25 2022

@author: Cheng Yuxuan Original, Disjoint Object Analyze of YoloXRe-DST-DataPool
"""
import os
import cv2
import csv
import copy
import torch
# import torchvision
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from datapool_analysis_disjoint import check_direction1, check_direction2
from datapool_analysis_overlap import situation_2, situation_3, overlap


def disjoint(img, res, label_name, save_path):
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(res.shape[0]): # 外层for循环,选定一个目标作为基准
        inner_bool = []
        for j in range(res.shape[0]): # 内层for循环,选定另一个目标与基准对比
            if res[i].tolist() == res[j].tolist(): # 当检测到两目标相同时跳过本轮次
                continue
            else: # 判断目标是否存在于本目标四个对面中,返回列表,其中为四个布尔变量
                inner_bool.append(check_direction1(res[i], res[j]))
        outer_bool.append(inner_bool) # 储存该基准目标与其他所有目标的对比结果
    
    # print(outer_bool)
    
    del inner_bool
    
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    four_side = []
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        top, bottom, left, right = False, False, False, False # 初始化四个方向的布尔变量
        for j in range(len(outer_bool[i])): # 对每个目标的对比结果进行或操作
            top    = top    or outer_bool[i][j][0]
            bottom = bottom or outer_bool[i][j][1]
            left   = left   or outer_bool[i][j][2]
            right  = right  or outer_bool[i][j][3]
        four_side.append([top, bottom, left, right]) # 储存最终对比结果,即该目标四面是否存在空面
    
    # print(four_side)
    
    # 取得每个目标的[上距 上左距 上右距], [下距 下左距 下右距], [左距 左上距 左下距], [右距 右上距 右下距]
    expandinfo = []
    for i in range(len(four_side)):
        list1, list2, list3, list4 = check_direction2(four_side[i], res[i], res, height, width)
        expandinfo.append([list1, list2, list3, list4])
        
    """
    print("\nexpandinfo: ")
    for i in range(len(expandinfo)):
        print(expandinfo[i]) # top bottom left right
    """

    # 分析目标数据得到多种可能的裁剪方式, 可以适当向外扩展几个像素
    cutinfo = []
    for i in range(len(expandinfo)):
        inner_info = []
        for j in range(len(expandinfo[i])):
            if len(expandinfo[i][j]) == 0:
                continue
            else:
                xcutmin = res[i][0] - expandinfo[i][j][2] # x1 - left
                ycutmin = res[i][1] - expandinfo[i][j][0] # y1 - top
                xcutmax = res[i][2] + expandinfo[i][j][3] # x2 + right
                ycutmax = res[i][3] + expandinfo[i][j][1] # y2 + bottom
                
                # print("\n", i, res[i], expandinfo[i][j], xcutmin, ycutmin, xcutmax, ycutmax)
                
                if xcutmin < 0:
                    xcutmin = 0
                if ycutmin < 0:
                    ycutmin = 0
                if xcutmax > width:
                    xcutmax = width
                if ycutmax > height:
                    ycutmax = height
                
                inner_info.append([xcutmin, ycutmin, xcutmax, ycutmax])
        cutinfo.append(inner_info)
    
    """
    print("\ncutinfo: ")
    for i in range(len(cutinfo)):
        print(cutinfo[i])
    """
    
    # 计算图块/目标比例方便检索
    ratio = []
    for i in range(res.shape[0]):
        inner_info = []
        for j in range(len(cutinfo[i])):
            target_area = (res[i][2] - res[i][0]) * (res[i][3] - res[i][1])
            cutout_area = (cutinfo[i][j][2] - cutinfo[i][j][0]) * (cutinfo[i][j][3] - cutinfo[i][j][1])
            ratio_area = cutout_area / target_area
            inner_info.append([target_area, cutout_area, ratio_area])
        ratio.append(inner_info)

    del inner_info
    
    """
    # 逐个目标地可视化图块计算是否正确
    _COLORS = np.array([0.000, 0.447, 0.741,
                        0.850, 0.325, 0.098,
                        0.929, 0.694, 0.125,
                        0.494, 0.184, 0.556,
                        0.466, 0.674, 0.188,
                        0.301, 0.745, 0.933,
                        0.635, 0.078, 0.184,
                        0.300, 0.300, 0.300,
                        0.600, 0.600, 0.600,
                        1.000, 0.000, 0.000,
                        1.000, 0.500, 0.000,
                        0.749, 0.749, 0.000,
                        0.000, 1.000, 0.000,
                        0.000, 0.000, 1.000,
                        0.667, 0.000, 1.000,
                        0.333, 0.333, 0.000,
                        0.333, 0.667, 0.000,
                        0.333, 1.000, 0.000,]).astype(np.float32).reshape(-1, 3)
    # label_tensor = torch.from_numpy(res[:, :4])
    for i in range(res.shape[0]):
        for j in range(len(cutinfo[i])):
            img_temp = copy.deepcopy(img)
            # label_temp = torch.cat((label_tensor, torch.from_numpy(np.array([cutinfo[i][j]]))), 0)
            label_temp = torch.from_numpy(np.array([cutinfo[i][j]]))
            for k in range(len(label_temp)):
                box = label_temp[k]
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                color = (_COLORS[k] * 255).astype(np.uint8).tolist()
                cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
            cv2.imwrite(save_path + str(label_name[:-4]) + '_' + str(i) + '_' + str(j) + '.jpg', img_temp)
    """
    
    
    # 将res内所有坐标从np.array转换成list
    if isinstance(res, list) != True:
        res = res.tolist()
    for i in range(len(res)):
        if isinstance(res[i], list) != True:
            res[i] = res[i].tolist()
    
    # 写入csv文件,图片名 目标坐标(xyxycls) 裁剪坐标(xyxy) 目标面积 裁剪面积 图块/目标比例
    for i in range(len(res)):
        for j in range(len(cutinfo[i])):
            list_info = [label_name,
                         res[i][0], res[i][1], res[i][2], res[i][3], res[i][4], 
                         cutinfo[i][j][0], cutinfo[i][j][1], cutinfo[i][j][2], cutinfo[i][j][3], 
                         ratio[i][j][0], ratio[i][j][1], ratio[i][j][2],
                         res[i]]
            with open(save_path, 'a', newline="") as csvfile:
                writer0 = csv.writer(csvfile)
                writer0.writerow(list_info)
    


def onedisjoint(target, res): # 处理存在相交目标的图片中的没有相交的目标
    # 该目标对比其他目标测试两者是否相对,并保存布尔列表
    inner_bool = []
    for i in range(res.shape[0]): # 内层for循环,选定另一个目标与基准对比
        if target.tolist() == res[i].tolist(): # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断目标是否存在于本目标四个对面中,返回列表,其中为四个布尔变量
            inner_bool.append(check_direction1(target, res[i]))
    
    # 该目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    top, bottom, left, right = False, False, False, False # 初始化四个方向的布尔变量
    for i in range(len(inner_bool)): # 对该目标的对比结果进行或操作
        top = top or inner_bool[i][0]
        bottom = bottom or inner_bool[i][1]
        left = left or inner_bool[i][2]
        right = right or inner_bool[i][3]
    
    list1, list2, list3, list4 = check_direction2([top, bottom, left, right], target, res, height, width)
    
    return [list1, list2, list3, list4]
    

def intersect(img, res, label_name, save_path):
    list_obj, results_whole, results_multi = [], copy.deepcopy(res).tolist(), copy.deepcopy(res).tolist() # 所有目标的重叠情况列表, 被看作为一个整体的目标坐标, 这个整体包含的目标的坐标
    
    # 分析每个目标与其他目标的模式, 完全无相交, 存在相交但不多, 严重相交
    for i in range(res.shape[0]):
        list1, list2, list3 = [], [], [] # 完全无相交, 小于0.1, 大于0.1
        for j in range(res.shape[0]):
            if res[i].tolist() == res[j].tolist():
                continue
            else:
                if overlap(res[i][:4], res[j][:4]) == 0: # torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) == 0:
                    list1.append(res[j]) # 无相交
                elif threshold >= overlap(res[i][:4], res[j][:4]) >= 0: # 0.1 >= torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0:
                    list2.append(res[j]) # 存在相交
                elif overlap(res[i][:4], res[j][:4]) > threshold: # torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0.1:
                    list3.append(res[j]) # 严重相交
        list_obj.append([list1, list2, list3])
    
    expandinfo = []
    # 分别对每个目标进行处理, 即处理其与其他目标的相交信息
    for i in range(len(list_obj)):
        # 当该目标与其他目标无重叠时
        if len(list_obj[i][0]) == len(res) - 1:
            # print("no intersect", i)
            expandinfo.append(onedisjoint(res[i], res)) # 只需要返回单个目标的扩展信息即可
        
        # 当该目标与其他目标重叠均不大于0.1时
        if len(list_obj[i][2]) == 0 and len(list_obj[i][1]) != 0:
            # print("some intersect", i)
            expandinfo.append(situation_2(res[i], list_obj[i], height, width))
        
        # 当该目标存在重叠大于0.1的其他目标时
        if len(list_obj[i][2]) != 0:
            # print("sever intersect", i)
            dense_obj, exinfo = situation_3(res[i], list_obj[i], res, height, width)
            expandinfo.append(exinfo)
            results_whole[i][:4] = np.array(dense_obj)
            if list_obj[i][2] != 1:
                list_obj[i][2].append(res[i]) # 该目标自己也算被大目标包含的目标之一
                results_multi[i] = list_obj[i][2] # 将这个大目标包含的所有目标全部赋予results_multi[i]
            else:
                if list_obj[i][2][0][0] <= res[i][0] < res[i][2] <= list_obj[i][2][0][2] and  list_obj[i][2][0][1] <= res[i][1] < res[i][3] <= list_obj[i][2][0][3]:
                    pass # 当该目标被其他目标包含且没有其他目标存在时, 该目标向外扩展10个像素, 因此只包括自己
                else:
                    list_obj[i][2].append(res[i]) # 该目标自己也算被大目标包含的目标之一
                    results_multi[i] = list_obj[i][2] # 将这个大目标包含的所有目标全部赋予results_multi[i]
    
    """
    print("\nresults_whole: ")
    for i in range(len(results_whole)):
        print(len(results_whole[i]) # 被看作为一个整体的目标坐标
    
    print("\nresults_multi: ")
    for i in range(len(results_multi)):
        print(results_multi[i]) # 组成大目标的所有目标的坐标
    """
    """
    print("\nexpandinfo: ")
    for i in range(len(expandinfo)):
        print(expandinfo[i])
    """
    
    # 分析目标数据得到多种可能的裁剪方式, 每边适当向外扩展10个像素--此处获得扩展信息准备切割--
    cutinfo = []
    for i in range(len(res)):
        if isinstance(expandinfo[i][0], list) == True: # 当该目标扩展信息有多组时
            cut_temp = []
            for j in range(len(expandinfo[i])): # 拆解这个有多组扩展参数的列表
                if len(expandinfo[i][j]) == 0: # 当然也可能里面存在空列表
                    continue
                else: # 的确存在扩展参数而不是空列表时
                    cut_temp.append(computecoordinate(results_whole[i], expandinfo[i][j]))
            if len(cut_temp) == 1:
                cut_temp = cut_temp[0]
            cutinfo.append(cut_temp)
        else: # 扩展信息只有一组时
            cutinfo.append(computecoordinate(results_whole[i], expandinfo[i]))
    
    assert len(results_whole) == len(expandinfo), "exist object missing"
    
    """
    print("cutinfo b4: ")
    for i in range(len(cutinfo)):
        print(cutinfo[i])
    """
    
    # 筛选出切割坐标列表中相同的坐标
    temp_list, pop_list = [], []
    for i in range(len(cutinfo)):
        if cutinfo[i] not in temp_list:
            temp_list.append(cutinfo[i])
        else:
            pop_list.append(i)
    
    # 排除切割坐标列表中相同的坐标
    if len(pop_list) != 0:
        pop_list.reverse()
        for i in pop_list:
            cutinfo.pop(i)
            results_whole.pop(i)
            results_multi.pop(i)
    
    """
    print("cutinfo af: ")
    for i in range(len(cutinfo)):
        print(cutinfo[i])
    """
    
    assert len(cutinfo) == len(results_whole), "something is wrong"
    
    # 计算图块/目标比例方便检索
    ratio, ratio_temp = [], []
    for i in range(len(cutinfo)):
        if isinstance(cutinfo[i][0], float) == True:
            target_area = (results_whole[i][2] - results_whole[i][0]) * (results_whole[i][3] - results_whole[i][1]) # 目标区域坐标
            cutout_area = (cutinfo[i][2] - cutinfo[i][0]) * (cutinfo[i][3] - cutinfo[i][1]) # 图块区域坐标
            ratio_area = cutout_area / target_area # 目标区域与图块区域的面积比例
            ratio.append([target_area, cutout_area, ratio_area])
        else:
            for j in range(len(cutinfo[i])):
                target_area = (results_whole[i][2] - results_whole[i][0]) * (results_whole[i][3] - results_whole[i][1]) # 目标区域坐标
                cutout_area = (cutinfo[i][j][2] - cutinfo[i][j][0]) * (cutinfo[i][j][3] - cutinfo[i][j][1]) # 图块区域坐标
                ratio_area = cutout_area / target_area # 目标区域与图块区域的面积比例
                ratio_temp.append([target_area, cutout_area, ratio_area])
            ratio.append(ratio_temp)
            ratio_temp = []
    
    """
    print("\nratio: ")
    for i in range(len(ratio)):
        print(ratio[i])
    """
    
    """
    # 逐个目标地可视化图块计算是否正确
    _COLORS = np.array([0.000, 0.447, 0.741,
                        0.850, 0.325, 0.098,
                        0.929, 0.694, 0.125,
                        0.494, 0.184, 0.556,
                        0.466, 0.674, 0.188,
                        0.301, 0.745, 0.933,
                        0.635, 0.078, 0.184,
                        0.300, 0.300, 0.300,
                        0.600, 0.600, 0.600,
                        1.000, 0.000, 0.000,
                        1.000, 0.500, 0.000,
                        0.749, 0.749, 0.000,
                        0.000, 1.000, 0.000,
                        0.000, 0.000, 1.000,
                        0.667, 0.000, 1.000,
                        0.333, 0.333, 0.000,
                        0.333, 0.667, 0.000,
                        0.333, 1.000, 0.000,]).astype(np.float32).reshape(-1, 3)
    # label_tensor = torch.from_numpy(res[:, :4]) # 取得所有目标坐标
    for i in range(len(cutinfo)):
        # label_temp = torch.cat((label_tensor, torch.from_numpy(np.array([cutinfo[i]]))), 0) # 在所有目标基础上可视化当前目标, 但需要将本句插入下面if条件句后
        if isinstance(cutinfo[i][0], float) == True:
            img_temp = copy.deepcopy(img)
            label_temp = torch.from_numpy(np.array([cutinfo[i]]))
            for k in range(len(label_temp)):
                box = label_temp[k]
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                color = (_COLORS[k] * 255).astype(np.uint8).tolist()
                cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
            cv2.imwrite(save_path + str(label_name[0:-4]) + '_' + str(i) + '.jpg', img_temp)
        else:
            for j in range(len(cutinfo[i])):
                img_temp = copy.deepcopy(img)
                label_temp = torch.from_numpy(np.array([cutinfo[i][j]]))
                for k in range(len(label_temp)):
                    box = label_temp[k]
                    x0 = int(box[0])
                    y0 = int(box[1])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    color = (_COLORS[k] * 255).astype(np.uint8).tolist()
                    cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
                cv2.imwrite(save_path + str(label_name[0:-4]) + '_' + str(i) + '_' + str(j) + '.jpg', img_temp)
    """
    
    
    # 将results_multi的所有元素全部转成list
    for i in range(len(results_multi)):
        for j in range(len(results_multi[i])):
            if isinstance(results_multi[i][j], float) == True:
                pass
            elif isinstance(results_multi[i][j], list) != True:
                results_multi[i][j] = results_multi[i][j].tolist()
    
    # 写入csv文件: 图片名, 目标坐标(xyxycls), 裁剪坐标(xyxy), 目标面积, 裁剪面积, 图块/目标比例
    for i in range(len(cutinfo)):
        if isinstance(cutinfo[i][0], float) == True:
            list_info = [label_name, # 图片名称
                         results_whole[i][0], results_whole[i][1], results_whole[i][2], results_whole[i][3], results_whole[i][4], # 目标整体坐标
                         cutinfo[i][0], cutinfo[i][1], cutinfo[i][2], cutinfo[i][3], # 图块切割坐标
                         ratio[i][0], ratio[i][1], ratio[i][2], # 目标面积, 图块面积, 两者比例
                         results_multi[i]] # 本图块中包含的目标的坐标, 可能存在多个目标
            with open(save_path, 'a', newline="") as csvfile:
                writer0 = csv.writer(csvfile)
                writer0.writerow(list_info)
        else:
            for j in range(len(cutinfo[i])):
                list_info = [label_name, # 图片名称
                             results_whole[i][0], results_whole[i][1], results_whole[i][2], results_whole[i][3], results_whole[i][4], # 目标整体坐标
                             cutinfo[i][j][0], cutinfo[i][j][1], cutinfo[i][j][2], cutinfo[i][j][3], # 图块切割坐标
                             ratio[i][j][0], ratio[i][j][1], ratio[i][j][2], # 目标面积, 图块面积, 两者比例
                             results_multi[i]] # 本图块中包含的目标的坐标, 可能存在多个目标
                with open(save_path, 'a', newline="") as csvfile:
                    writer0 = csv.writer(csvfile)
                    writer0.writerow(list_info)
    

# -----------------------------------------------------------------------------
def computecoordinate(coor, info):
    xcutmin = coor[0] - info[2] # x1 - left
    ycutmin = coor[1] - info[0] # y1 - top
    xcutmax = coor[2] + info[3] # x2 + right
    ycutmax = coor[3] + info[1] # y2 + bottom
    
    if xcutmin < 0:
        xcutmin = 0
    if ycutmin < 0:
        ycutmin = 0
    if xcutmax > width:
        xcutmax = width
    if ycutmax > height:
        ycutmax = height

    return [xcutmin, ycutmin, xcutmax, ycutmax]


def xywh_to_xyxy(tensor): # [x1, y1, x2, y2, cls]
    for i in range(tensor.shape[0]):
        x1 = copy.deepcopy(tensor[i][0] - tensor[i][2] / 2)
        y1 = copy.deepcopy(tensor[i][1] - tensor[i][3] / 2)
        x2 = copy.deepcopy(tensor[i][0] + tensor[i][2] / 2)
        y2 = copy.deepcopy(tensor[i][1] + tensor[i][3] / 2)
        tensor[i][0] = x1
        tensor[i][1] = y1
        tensor[i][2] = x2
        tensor[i][3] = y2
        
    return tensor


def xml_target(xml_path, class_to_ind):
    # 读取xml文件中所有目标
    target = ET.parse(xml_path).getroot()
    res = np.empty((0, 5))
    for obj in target.iter("object"):
        name = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        pts = ["xmin", "ymin", "xmax", "ymax"]
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(float(bbox.find(pt).text)) - 1
            bndbox.append(cur_pt)
        label_idx = class_to_ind[name]
        bndbox.append(label_idx)
        res = np.vstack((res, bndbox)) # [0xmin, 1ymin, 2xmax, 3ymax, 4label_ind]
    
    # 取得图片宽高
    width = int(target.find("size").find("width").text)
    height = int(target.find("size").find("height").text)
    
    return res, height, width


def txt_target(txt_path, height, width): # COCO txt = xywh
    res = np.empty((0, 5))
    df = pd.read_table(txt_path, header = None)
    list1 = df.values.tolist()
    for i in range(len(list1)):
        list2 = list1[i][0].split()
        list2[0] = int(list2[0])# list2[0]=cls
        list2[1] = float(list2[1])# list2[1]=Xcen
        list2[2] = float(list2[2])# list2[2]=Ycen
        list2[3] = float(list2[3])# list2[3]=W
        list2[4] = float(list2[4])# list2[4]=H
        res = np.vstack((res, [list2[1]*width, list2[2]*height, list2[3]*width, list2[4]*height, list2[0]]))
        # [0xcen, 1ycen, 2w, 3h, 4label_ind]
    
    return res


if __name__ == "__main__":
    # 将科学计数法转换为数字
    np.set_printoptions(suppress=True)

    # 设定并封装类别 classes
    VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
                   "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    
    # 设定单张图片与标签的DeBug测试文件夹路径 Single image test
    # img_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/singleimg/'
    # label_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/singlelab/'
    # test_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/singleres/'
    
    # 设定多张图片与标签的批量测试文件夹路径 Multi images test
    # img_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/vocimages/'
    # label_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/voclabels/'
    # test_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/vocresults/'
    
    # 设定正式计算目标边界的数据集与csv文件路径 Formal Comuptation for Patches
    img_path = 'F:/VOCtrainval_images/' # D:/AICV-DSTRethink/DataPool/vocdataset/images/
    label_path = 'F:/VOCtrainval_xml/' # D:/AICV-DSTRethink/DataPool/vocdataset/labels/
    csv_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/VOC_0712trainval_noandoverlapmix.csv'
    
    # 测试路径与csv文件保存路径不能同时存在
    if 'test_path' in dir():
        save_path = locals()['test_path']
    if 'csv_path' in dir():
        save_path = locals()['csv_path']
    
    global threshold
    threshold = 0.2
    
    for label_name in os.listdir(label_path):
        # print("+++++++++++++++++++++++++++++")
        # print(label_name)
        
        # 设定图片路径并读取图片宽高
        img = cv2.imread(img_path + label_name[0:-4] + '.jpg')
        height, width = img.shape[0], img.shape[1]
        
        # 读取txt文件中的目标坐标
        # res = txt_target(label_path + label_name, height, width)
        
        # txt [xcen, ycen, w, h] to [xmin, ymin, xmax, ymax]
        # res = xywh_to_xyxy(res)
        
        # 读取xml文件中的目标坐标
        res, height, width = xml_target(label_path + label_name, class_to_ind)
        
        # print("------------------------")
        # print(res)
        
        label_flag = 0 # 初始化label_flag
        
        # 如果直接将只有一个目标的图片送入后续会产生问题
        if len(res) == 1:
            disjoint(img, res, label_name, save_path)
        else:
            # 检查是否存在相交区域,如果存在相交区域则无法使用本算法
            for i in range(res.shape[0]):
                for j in range(res.shape[0]):
                    if res[i].tolist() == res[j].tolist():
                        continue
                    else:
                        if overlap(res[i][:4], res[j][:4]) > 0: # torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0:
                            break
                else: # for-else逻辑为执行完for就会执行else, 当for无法正常执行完毕时不会执行else
                    continue
                label_flag = 1 # label_flag为1则该图片存在重叠目标
                break
        
        # 检测该图片是否存在重叠, 需要调用哪套方案
        if label_flag == 1:
            # print(label_name[0:-4], "intersect")
            intersect(img, res, label_name, save_path)
        else:
            # print(label_name[0:-4], "disjoint")
            disjoint(img, res, label_name, save_path)
            