# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:46:53 2022

@author: Cheng Yuxuan Original, Disjoint Object Analyze of YoloXRe-DST-DataPool
"""
import os
import cv2
import csv
import copy
import torch
# import shutil
import torchvision
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


# 以1个目标为中心,判断另一个目标是否在其对面
def check_direction1(tensor1, tensor2): # True=存在目标 False=不存在目标, top bottom left right
    if tensor1[2] >= tensor2[0] and tensor1[0] <= tensor2[2] and tensor1[1] >= tensor2[3]: # x2 > x'1 & x1 < x'2 & y1 > y'2
        top = True
    else:
        top = False
    if tensor1[2] >= tensor2[0] and tensor1[0] <= tensor2[2] and tensor1[3] <= tensor2[1]: # x2 > x'1 & x1 < x'2 & y2 < y'1
        bottom = True
    else:
        bottom = False
    if tensor1[3] >= tensor2[1] and tensor1[1] <= tensor2[3] and tensor1[0] >= tensor2[2]: # y2 > y'1 & y1 < y'2 & x1 > x'2
        left = True
    else:
        left = False
    if tensor1[3] >= tensor2[1] and tensor1[1] <= tensor2[3] and tensor1[2] <= tensor2[0]: # y2 > y'1 & y1 < y'2 & x2 < x'1
        right = True
    else:
        right = False
    
    return [top, bottom, left, right] # 四个值只有一个值为True,或者四个值均不为True


# 以1个目标为中心,判断其四个对面是否存在其他目标
def check_direction2(flag, tensor1, res, height, width): # True=存在目标, False=不存在目标
    if flag == [False, False, False, False]: # 四个对面都没有目标=================
        list1, list2, list3, list4 = four_side_no_target(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, False, False, False]: # 顶面存在目标,其他面没有目标1111111111
        list1, list2, list3, list4 = top_side_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [False, True, False, False]: # 底面存在目标,其他面没有目标
        list1, list2, list3, list4 = bottom_side_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [False, False, True, False]: # 左面存在目标,其他面没有目标
        list1, list2, list3, list4 = left_side_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [False, False, False, True]: # 右面存在目标,其他面没有目标1111111111
        list1, list2, list3, list4 = right_side_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, True, False, False]: # 顶面和底面存在目标,其他面没有目标------
        list1, list2, list3, list4 = top_bottom_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, False, True, False]: # 顶面和左面存在目标,其他面没有目标
        list1, list2, list3, list4 = top_left_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, False, False, True]: # 顶面和右面存在目标,其他面没有目标
        list1, list2, list3, list4 = top_right_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [False, True, True, False]: # 底面和左面存在目标,其他面没有目标
        list1, list2, list3, list4 = bottom_left_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [False, True, False, True]: # 底面和右面存在目标,其他面没有目标
        list1, list2, list3, list4 = bottom_right_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [False, False, True, True]: # 左面和右面存在目标,其他面没有目标------
        list1, list2, list3, list4 = left_right_only(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, True, True, False]: # 顶面和底面和左面存在目标,其他面没有目标22
        list1, list2, list3, list4 = top_bottom_left(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, True, False, True]: # 顶面和底面和右面存在目标,其他面没有目标
        list1, list2, list3, list4 = top_bottom_right(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, False, True, True]: # 顶面和左面和右面存在目标,其他面没有目标
        list1, list2, list3, list4 = top_left_right(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [False, True, True, True]: # 底面和左面和右面存在目标,其他面没有目标22
        list1, list2, list3, list4 = bottom_left_right(tensor1, res, height, width)
        return list1, list2, list3, list4
    if flag == [True, True, True, True]: # 四个面全部存在目标++++++++++++++++++++
        list1, list2, list3, list4 = four_side_has_target(tensor1, res, height, width)
        return list1, list2, list3, list4


def top_side_no_target(tensor1, res, height, width): # 该目标上方无其他目标,检查左上右上是否存在目标并计算距离
    # 初始化左右距离列表
    left_len, right_len = [], []
    
    # tensor1可能为np.array也可能为list
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
    
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断是否在选中目标上方,并计算其与选中目标距离
    for i in range(len(res)):
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 计算两目标之间的距离
            if tensor1[1] > res[i][3]: # y1 > y'2 存在目标在本目标上方
                if tensor1[0] > res[i][2]: # x1 > x'2 存在目标在本目标左上方
                    left_len.append(tensor1[0] - res[i][2])
                if tensor1[2] < res[i][0]: # x2 < x'1 存在目标在本目标右上方
                    right_len.append(res[i][0] - tensor1[2])
    
    # 取得右上目标与本目标的距离
    if len(right_len) == 0:
        right_len = width - tensor1[2]
    else:
        right_len = min(right_len)
    
    # 取得左上目标与本目标的距离
    if len(left_len) == 0:
        left_len = tensor1[0]
    else:
        left_len = min(left_len)
    
    return left_len, right_len


def top_side_has_target(tensor1, res, height, width): # 该目标上方存在目标,检查左上右上是否存在目标并计算距离
    # 初始化左右距离列表
    top, left_len, right_len = [], [], []
    
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
    
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断选中目标上方最短距离
    for i in range(len(res)):
        if isinstance(res[i], list):
            pass
        else:
            res[i] = res[i].tolist()
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次的目标是否在基准目标正上方
            if tensor1[1] >= res[i][3] and tensor1[2] >= res[i][0] and tensor1[0] <= res[i][2]: # y1 > y'2 & x2 > x'1 & x1 < x'2
                top.append(tensor1[1] - res[i][3]) # y1 - y'2, 可能存在多个目标在基准目标正上方
    
    if len(top) != 0:
        top = min(top) # 取得最小上距
    else:
        top = tensor1[1]
    
    # 遍历所有目标,判断目标是否在基准目标的左上和右上
    for i in range(len(res)):
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次目标是否在基准目标的左上方和右上方
            if tensor1[1] > res[i][3] > tensor1[1] - top and tensor1[0] > res[i][2]: # y1 > y'2 > y1 - top & x1 > x'2
                left_len.append(tensor1[0] - res[i][2])
            if tensor1[1] > res[i][3] > tensor1[1] - top and tensor1[2] < res[i][0]: # y1 > y'2 > y1 - top & x2 < x'1
                right_len.append(res[i][0] - tensor1[2])

    # 取得右上目标与本目标的距离
    if len(right_len) == 0:
        right_len = width - tensor1[2]
    else:
        right_len = min(right_len)
    
    # 取得左上目标与本目标的距离
    if len(left_len) == 0:
        left_len = tensor1[0]
    else:
        left_len = min(left_len)

    return top, left_len, right_len


def bottom_side_no_target(tensor1, res, height, width): # 该目标下方无其他目标,检查左下右下是否存在目标并计算距离
    # 初始化左右距离列表
    left_len, right_len = [], []
    
    # 总是出现tensor1和res要么是np.array要么是list的情况, 烦得很
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
    
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断是否在选中目标上方,并计算其与选中目标距离
    for i in range(len(res)):
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 计算两目标之间的距离
            if tensor1[3] < res[i][1]: # y2 < y'1 存在目标在本目标下方
                if tensor1[0] > res[i][2]: # x1 > x'2 存在目标在本目标左下方
                    left_len.append(tensor1[0] - res[i][2])
                if tensor1[2] < res[i][0]: # x2 < x'1 存在目标在本目标右下方
                    right_len.append(res[i][0] - tensor1[2])
    
    # 取得右下目标与本目标的距离
    if len(right_len) == 0:
        right_len = width - tensor1[2]
    else:
        right_len = min(right_len)
    
    # 取得左下目标与本目标的距离
    if len(left_len) == 0:
        left_len = tensor1[0]
    else:
        left_len = min(left_len)
    
    return left_len, right_len


def bottom_side_has_target(tensor1, res, height, width): # 该目标下方存在目标,检查左下右下是否存在目标并计算距离
    # 初始化左右距离列表
    bottom, left_len, right_len = [], [], []
    
    # tensor1可能为np.array也可能为list
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
    
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断选中目标下方最短距离
    for i in range(len(res)):
        if isinstance(res[i], list):
            pass
        else:
            res[i] = res[i].tolist()
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次的目标是否在基准目标正下方
            if tensor1[3] <= res[i][1] and tensor1[2] >= res[i][0] and tensor1[0] <= res[i][2]: # y2 < y'1 & x2 > x'1 & x1 < x'2
                bottom.append(res[i][1] - tensor1[3]) # y'1 - y2
    
    if len(bottom) != 0:
        bottom = min(bottom) # 取得最小下距
    else:
        bottom = height - tensor1[3]
    
    # 遍历所有目标,判断目标是否在基准目标的左下和右下
    for i in range(len(res)):
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次目标是否在基准目标的左下方和右下方
            if tensor1[3] + bottom > res[i][1] > tensor1[3] and tensor1[0] > res[i][2]: # y2 + bot > y'1 > y2 & x1 > x'2
                left_len.append(tensor1[0] - res[i][2])
            if tensor1[3] + bottom > res[i][1] > tensor1[3] and tensor1[2] < res[i][0]: # y2 + bot > y'1 > y2 & x2 < x'1
                right_len.append(res[i][0] - tensor1[2])
                
    # 取得右下目标与本目标的距离
    if len(right_len) == 0:
        right_len = width - tensor1[2]
    else:
        right_len = min(right_len)
    
    # 取得左下目标与本目标的距离
    if len(left_len) == 0:
        left_len = tensor1[0]
    else:
        left_len = min(left_len)
    
    return bottom, left_len, right_len


def left_side_no_target(tensor1, res, height, width): # 该目标左侧无其他目标,检查左上左下是否存在目标并计算距离
    # 初始化左右距离列表
    top_len, bottom_len = [], []
    
    # 总是出现tensor1和res要么是np.array要么是list的情况, 烦得很
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
    
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断是否在选中目标上方,并计算其与选中目标距离
    for i in range(len(res)):
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 计算两目标之间的距离
            if tensor1[0] > res[i][2]: # x1 > x'2 存在目标在本目标左侧
                if tensor1[1] > res[i][3]: # y1 > y'2 存在目标在本目标左上方
                    top_len.append(tensor1[1] - res[i][3])
                if tensor1[3] < res[i][1]: # y2 < y'1 存在目标在本目标左下方
                    bottom_len.append(res[i][1] - tensor1[3])
    
    # 取得左上目标与本目标的距离
    if len(top_len) == 0:
        top_len = tensor1[1]
    else:
        top_len = min(top_len)
    
    # 取得左下目标与本目标的距离
    if len(bottom_len) == 0:
        bottom_len = height - tensor1[3]
    else:
        bottom_len = min(bottom_len)
    
    return top_len, bottom_len


def left_side_has_target(tensor1, res, height, width): # 该目标左侧存在目标,检查左上左下是否存在目标并计算距离
    # 初始化左右距离列表
    left, top_len, bottom_len = [], [], []
    
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
    
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断是否在选中目标左侧,并计算其与选中目标距离
    for i in range(len(res)):
        if isinstance(res[i], list):
            pass
        else:
            res[i] = res[i].tolist()
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次目标是否在基准目标正左侧
            if tensor1[0] >= res[i][2] and tensor1[3] >= res[i][1] and tensor1[1] <= res[i][3]: # x1 > x'2 & y2 > y'1 & y1 < y'2
                left.append(tensor1[0] - res[i][2]) # x1 - x'2
    
    if len(left) != 0:
        left = min(left) # 取得最小左距
    else:
        left = tensor1[0]
    
    # 遍历所有目标,判断目标是否在基准目标的左上和左下
    for i in range(len(res)):
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次目标是否在基准目标的左上方和左下方
            if tensor1[0] > res[i][2] > tensor1[0] - left and tensor1[1] > res[i][3]: # x1 > x'2 > x1 - left & y1 > y'2
                top_len.append(tensor1[1] - res[i][3])
            if tensor1[0] > res[i][2] > tensor1[0] - left and tensor1[3] < res[i][1]: # x1 > x'2 > x1 - left & y2 < y'1
                bottom_len.append(res[i][1] - tensor1[3])
    
    # 取得左上目标与本目标的距离
    if len(top_len) == 0:
        top_len = tensor1[1]
    else:
        top_len = min(top_len)
    
    # 取得左下目标与本目标的距离
    if len(bottom_len) == 0:
        bottom_len = height - tensor1[3]
    else:
        bottom_len = min(bottom_len)
    
    return left, top_len, bottom_len


def right_side_no_target(tensor1, res, height, width): # 该目标右侧无其他目标,检查右上右下是否存在目标并计算距离
    # 初始化左右距离列表
    top_len, bottom_len = [], []
    
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
        
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断是否在选中目标上方,并计算其与选中目标距离
    for i in range(len(res)):
        if isinstance(res[i], list):
            pass
        else:
            res[i] = res[i].tolist()
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 计算两目标之间的距离
            if tensor1[2] < res[i][0]: # x2 < x'1 存在目标在本目标右侧
                if tensor1[1] > res[i][3]: # y1 > y'2 存在目标在本目标右上方
                    top_len.append(tensor1[1] - res[i][3])
                if tensor1[3] < res[i][1]: # y2 < y'1 存在目标在本目标右下方
                    bottom_len.append(res[i][1] - tensor1[3])
    
    # 取得右下目标与本目标的距离
    if len(top_len) == 0:
        top_len = tensor1[1]
    else:
        top_len = min(top_len)
    
    # 取得左下目标与本目标的距离
    if len(bottom_len) == 0:
        bottom_len = height - tensor1[3]
    else:
        bottom_len = min(bottom_len)
    
    return top_len, bottom_len


def right_side_has_target(tensor1, res, height, width): # 该目标左侧存在目标,检查左上左下是否存在目标并计算距离
    # 初始化左右距离列表
    right, top_len, bottom_len = [], [], []
    
    # 总是出现tensor1和res要么是np.array要么是list的情况, 烦得很
    if isinstance(tensor1, list):
        pass
    else:
        tensor1 = tensor1.tolist()
    
    if isinstance(res, list):
        pass
    else:
        res = res.tolist()
    
    # 遍历所有目标,判断是否在选中目标右侧,并计算其与选中目标距离
    for i in range(len(res)):
        if isinstance(res[i], list):
            pass
        else:
            res[i] = res[i].tolist()
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次目标是否在基准目标正右侧
            if tensor1[2] <= res[i][0] and tensor1[3] >= res[i][1] and tensor1[1] <= res[i][3]: # x2 < x'1 & y2 > y'1 & y1 < y'2
                right.append(res[i][0] - tensor1[2]) # x'1 - x2
    
    if len(right) != 0:
        right = min(right) # 取得最小左距
    else:
        right = width - tensor1[2]
    
    # 遍历所有目标,判断目标是否在基准目标的右上和右下
    for i in range(len(res)):
        if tensor1 == res[i]: # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断本轮次目标是否在基准目标的右上方和右下方
            if tensor1[2] + right > res[i][0] > tensor1[2] and tensor1[1] > res[i][3]: # x2 + right > x'1 > x2 & y1 > y'2
                top_len.append(tensor1[1] - res[i][3])
            if tensor1[2] + right > res[i][0] > tensor1[2] and tensor1[3] < res[i][1]: # x2 + right > x'1 > x2 & y2 < y'1
                bottom_len.append(res[i][1] - tensor1[3])
    
    # 取得右下目标与本目标的距离
    if len(top_len) == 0:
        top_len = tensor1[1]
    else:
        top_len = min(top_len)
    
    # 取得左下目标与本目标的距离
    if len(bottom_len) == 0:
        bottom_len = height - tensor1[3]
    else:
        bottom_len = min(bottom_len)
    
    return right, top_len, bottom_len
    
# -----------------------------------------------------------------------------
    
def four_side_no_target(tensor1, res, height, width): # 四面均不存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3, list4 = [], [], [], []
    
    # 左上&右上&左下&右下全部没有目标
    if top_left_len == tensor1[0] and top_right_len == width - tensor1[2] and bottom_left_len == tensor1[0] and bottom_right_len == width - tensor1[2]:
        list1 = [tensor1[1], height - tensor1[3], tensor1[0], width - tensor1[2]]
    
    # 左上存在目标,其他方向无目标
    if top_left_len != tensor1[0] and top_right_len == width - tensor1[2] and bottom_left_len == tensor1[0] and bottom_right_len == width - tensor1[2]:
        list1 = [tensor1[1], height - tensor1[3], top_left_len, width - tensor1[2]]
        
    # 右上存在目标,其他方向无目标
    if top_left_len == tensor1[0] and top_right_len != width - tensor1[2] and bottom_left_len == tensor1[0] and bottom_right_len == width - tensor1[2]:
        list1 = [tensor1[1], height - tensor1[3], tensor1[0], top_right_len]
    
    # 左下存在目标,其他方向无目标
    if top_left_len == tensor1[0] and top_right_len == width - tensor1[2] and bottom_left_len != tensor1[0] and bottom_right_len == width - tensor1[2]:
        list1 = [tensor1[1], height - tensor1[3], bottom_left_len, width - tensor1[2]]
    
    # 右下存在目标,其他方向无目标
    if top_left_len == tensor1[0] and top_right_len == width - tensor1[2] and bottom_left_len == tensor1[0] and bottom_right_len != width - tensor1[2]:
        list1 = [tensor1[1], height - tensor1[3], tensor1[0], bottom_right_len]
    
    # 左上存在目标&右上存在目标,直上下式切图
    if top_left_len != tensor1[0] and top_right_len != width - tensor1[2]:
        list1 = [tensor1[1], height - tensor1[3], min(top_left_len, bottom_left_len), min(top_right_len, bottom_right_len)]
    
    # 左下存在目标&右下存在目标,直上下式切图
    if bottom_left_len != tensor1[0] and bottom_right_len != width - tensor1[2]:
        list1 = [tensor1[1], height - tensor1[3], min(top_left_len, bottom_left_len), min(top_right_len, bottom_right_len)]
    
    # 左上存在目标&左下存在目标,直左右式切图
    if top_left_len != tensor1[0] and bottom_left_len != tensor1[0]:
        list2 = [min(left_top_len, right_top_len), min(left_bottom_len, right_bottom_len), tensor1[0], width - tensor1[2]]
    
    # 右上存在目标&右下存在目标,直左右式切图
    if top_right_len != width - tensor1[2] and bottom_right_len != width - tensor1[2]:
        list2 = [min(left_top_len, right_top_len), min(left_bottom_len, right_bottom_len), tensor1[0], width - tensor1[2]]
    
    # 左上不存在目标,其他方向存在目标,对角不确定
    if top_left_len == tensor1[0] and top_right_len != width - tensor1[2] and bottom_left_len != tensor1[0]:
        if bottom_right_len != width - tensor1[2]: # 对角存在目标
            list3 = [tensor1[1], left_bottom_len, tensor1[0], min(top_right_len, bottom_right_len)]
        else:
            list3 = [tensor1[1], left_bottom_len, tensor1[0], min(top_right_len, bottom_right_len)]
            list4 = [left_top_len, height - tensor1[3], bottom_left_len, width - tensor1[2]]
    
    # 右上不存在目标,其他方向存在目标,对角不确定
    if top_right_len == width - tensor1[2] and top_left_len != tensor1[0] and bottom_right_len != width - tensor1[2]:
        if bottom_left_len != tensor1[0]: # 对角存在目标
            list3 = [tensor1[1], right_bottom_len, min(top_left_len, bottom_left_len), width - tensor1[2]]
        else:
            list3 = [tensor1[1], right_bottom_len, min(top_left_len, bottom_left_len), width - tensor1[2]]
            list4 = [left_top_len, height - tensor1[3], tensor1[0], bottom_right_len]
    
    # 左下不存在目标,其他方向存在目标,对角不确定
    if bottom_left_len == tensor1[0] and top_left_len != tensor1[0] and bottom_right_len != width - tensor1[2]:
        if top_right_len != width - tensor1[2]: # 对角存在目标
            list3 = [left_top_len, height - tensor1[3], tensor1[0], min(top_right_len, bottom_right_len)]
        else:
            list3 = [left_top_len, height - tensor1[3], tensor1[0], min(top_right_len, bottom_right_len)]
            list4 = [tensor1[1], right_bottom_len, top_left_len, width - tensor1[2]]
    
    # 右下不存在目标,其他方向存在目标,对角不确定
    if bottom_right_len == width - tensor1[2] and top_right_len != width - tensor1[2] and bottom_left_len != tensor1[0]:
        if top_left_len != tensor1[0]: # 对角存在目标
            list3 = [right_top_len, height - tensor1[3], min(top_left_len, bottom_left_len), width - tensor1[2]]
        else:
            list3 = [right_top_len, height - tensor1[3], min(top_left_len, bottom_left_len), width - tensor1[2]]
            list4 = [tensor1[1], left_bottom_len, tensor1[0], top_right_len]
    
    return list1, list2, list3, list4
    
    # return tensor1[1], top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def top_side_only(tensor1, res, height, width): # 只有上方存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直左右式切图距离(虽然左右无目标但不代表左上右上左下右下无目标)
    list1 = [min(top, left_top_len, right_top_len), min(left_bottom_len, right_bottom_len), tensor1[0], width - tensor1[2]]
    
    # 排除左上或右上无目标&上方目标距基准目标最近的情况
    if top != min(top, left_top_len, right_top_len):
        # 判断上方开口方向
        if left_top_len > right_top_len: # 上方开口朝左
            top_orient = 'lef'
        else: # left_top_len < right_top_len 上方开口朝右
            top_orient = 'rig'
    else:
        top_orient = 'Non'
        
    # 当上方开口朝左时计算开口式切图四周距离
    if top_orient == 'lef':
        # 计算上方开口距离
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算右侧最短距离
        close_rig = min(top_right_len, bottom_right_len)
        
        # 组装列表
        list2 = [open_top, height - tensor1[3], tensor1[0], close_rig]
    
    # 当上方开口朝右时计算开口式切图四周距离
    if top_orient == 'rig':
        # 计算上方开口距离
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算左侧最短距离
        close_lef = min(top_left_len, bottom_left_len)
        
        # 组装列表
        list2 = [open_top, height - tensor1[3], close_lef, width - tensor1[2]]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def bottom_side_only(tensor1, res, height, width): # 只有下方存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直左右式切图距离(虽然左右无目标但不代表左上右上左下右下无目标)
    list1 = [min(left_top_len, right_top_len), min(bottom, left_bottom_len, right_bottom_len), tensor1[0], width - tensor1[2]]
    
    # 排除左下或右下无目标&下方目标距基准目标最近的情况
    if bottom != min(bottom, left_bottom_len, right_bottom_len):
        # 判断下方开口方向
        if left_bottom_len > right_bottom_len: # 下方开口朝左
            bot_orient = 'lef'
        else:
            bot_orient = 'rig'
    else:
        bot_orient = 'Non'
    
    # 当下方开口朝左时计算开口式切图四周距离
    if bot_orient == 'lef':
        # 计算下方开口距离
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bot = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bot = left_bottom_len
        else: # left_top_len < right_top_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bot = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bot = right_bottom_len
        
        # 计算右侧最短距离
        close_rig = min(top_right_len, bottom_right_len)
        
        # 组装列表
        list2 = [tensor1[1], open_bot, tensor1[0], close_rig]
    
    # 当下方开口朝右时计算开口式切图四周距离
    if bot_orient == 'rig':
        # 计算下方开口距离
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bot = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bot = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bot = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bot = right_bottom_len
        
        # 计算左侧最短距离
        close_lef = min(top_left_len, bottom_left_len)
        
        # 组装列表
        list2 = [tensor1[1], open_bot, close_lef, width - tensor1[2]]
    
    return list1, list2, list3, []
    
    # return tensor1[1], top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def left_side_only(tensor1, res, height, width): # 只有左侧存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直上下式切图距离(虽然上下无目标但不代表左上右上左下右下无目标)
    list1 = [tensor1[1], height - tensor1[3], min(left, top_left_len, bottom_left_len), min(top_right_len, bottom_right_len)]
    
    # 排除左上或左下无目标&左侧目标距基准目标最近的情况
    if left != min(left, top_left_len, bottom_left_len):
        # 判断左侧开口方向
        if top_left_len > bottom_left_len: # 下方开口朝左
            lef_orient = 'top'
        else:
            lef_orient = 'bot'
    else:
        lef_orient = 'Non'
        
    # 当左侧开口朝上时计算开口式切图四周距离
    if lef_orient == 'top':
        # 计算左侧开口距离
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_lef = left
            else:
                open_lef = top_left_len
        else:
            if bottom_left_len > left:
                open_lef = left
            else:
                open_lef = bottom_left_len
        
        # 计算下方最短距离
        close_bot = min(left_bottom_len, right_bottom_len)
        
        # 组装列表
        list2 = [tensor1[1], close_bot, open_lef, width - tensor1[2]]
    
    # 当左侧开口朝下时计算开口式切图四周距离
    if lef_orient == 'bot':
        # 计算左侧开口距离
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_lef = left
            else:
                open_lef = top_left_len
        else:
            if bottom_left_len > left:
                open_lef = left
            else:
                open_lef = bottom_left_len
        
        # 计算上方最短距离
        close_top = min(left_top_len, right_top_len)
        
        # 组装列表
        list2 = [close_top, height - tensor1[3], open_lef, width - tensor1[2]]
        
    return list1, list2, list3, []
    
    # return tensor1[1], top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def right_side_only(tensor1, res, height, width): # 只有右侧存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直上下式切图距离(虽然上下无目标但不代表左上右上左下右下无目标)
    list1 = [tensor1[1], height - tensor1[3], min(top_left_len, bottom_left_len), min(right, top_right_len, bottom_right_len)]
    
    # 排除右上或右下无目标&左侧目标距基准目标最近的情况
    if right != min(right, top_right_len, bottom_right_len):
        # 判断右侧开口方向
        if top_right_len > bottom_right_len:
            rig_orient = 'top'
        else:
            rig_orient = 'bot'
    else:
        rig_orient = 'Non'
        
    # 当右侧开口朝上时计算开口式切图四周距离
    if rig_orient == 'top':
        # 计算右侧开口距离
        if top_right_len > bottom_right_len:
            if top_right_len > right:
                open_rig = right
            else:
                open_rig = top_right_len
        else:
            if bottom_right_len > right:
                open_rig = right
            else:
                open_rig = bottom_right_len
        
        # 计算下方最短距离
        close_bot = min(left_bottom_len, right_bottom_len)
        
        # 组装列表
        list2 = [tensor1[1], close_bot, tensor1[0], open_rig]
    
    # 当右侧开口朝下时计算开口式切图四周距离
    if rig_orient == 'bot':
        # 计算右侧开口距离
        if top_right_len > bottom_right_len:
            if top_right_len > right:
                open_rig = right
            else:
                open_rig = top_right_len
        else:
            if bottom_right_len > right:
                open_rig = right
            else:
                open_rig = bottom_right_len
        
        # 计算上方最短距离
        close_top = min(left_top_len, right_top_len)
        
        # 组装列表
        list2 = [close_top, height - tensor1[3], tensor1[0], open_rig]
    
    return list1, list2, list3, []
    
    # return tensor1[1], top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def top_bottom_only(tensor1, res, height, width): # 上方和下方存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直左右式切图距离
    list1 = [min(top, left_top_len, right_top_len), min(bottom, left_bottom_len, right_bottom_len), tensor1[0], width - tensor1[2]]
    
    # 排除左上或右上无目标&上方目标距基准目标最近的情况
    if top != min(top, left_top_len, right_top_len):
        # 判断单开口式切图上方的朝向
        if left_top_len > right_top_len: # 上方开口朝左
            top_orient = 'lef'
        else: # left_top_len < right_top_len 上方开口朝右
            top_orient = 'rig'
    else:
        top_orient = 'Non'
    
    # 排除左下或右下无目标&上方目标距基准目标最近的情况
    if bottom != min(bottom, left_bottom_len, right_bottom_len):
        # 判断单开口式切图下方的朝向
        if left_bottom_len > right_bottom_len: # 下方开口朝左
            bot_orient = 'lef'
        else: # left_bottom_len < right_bottom_len 下方开口朝右
            bot_orient = 'rig'
    else:
        bot_orient = 'Non'
    
    # 当上方开口朝左时计算开口式切图的四周距离
    if top_orient != bot_orient and top_orient == 'lef':
        # 计算上方开口距离 - 上方开口向左切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方闭口距离
        close_bottom = min(bottom, left_bottom_len, right_bottom_len)
        
        # 计算右侧最短距离
        close_right = min(top_right_len, bottom_right_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, close_bottom, tensor1[0], close_right]
    
    # 当上方开口朝右时计算开口式切图的四周距离
    if top_orient != bot_orient and top_orient == 'rig':
        # 计算上方开口距离 - 上方开口向右切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方闭口距离
        close_bottom = min(bottom, left_bottom_len, right_bottom_len)
        
        # 计算左侧最短距离
        close_left = min(top_left_len, bottom_left_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, close_bottom, close_left, width - tensor1[2]]
    
    # 当下方开口朝左时计算开口式切图的四周距离
    if bot_orient != top_orient and bot_orient == 'lef':
        # 计算下方开口距离 - 下方开口向左切图
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算上方闭口距离
        close_top = min(top, left_top_len, right_top_len)
        
        # 计算右侧最短距离
        close_right = min(top_right_len, bottom_right_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, open_bottom, tensor1[0], close_right]
    
    # 当下方开口朝右时计算开口式切图的四周距离
    if bot_orient != top_orient and bot_orient == 'rig':
        # 计算下方开口距离 - 下方开口向右切图
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算上方闭口距离
        close_top = min(top, left_top_len, right_top_len)
        
        # 计算左侧最短距离
        close_left = min(top_left_len, bottom_left_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, open_bottom, close_left, width - tensor1[2]]
    
    # 当上方开口朝左&下方开口朝左时,计算单开口式切图的四周距离
    if top_orient == bot_orient == 'lef':
        # 计算上方开口距离 - 两方开口向左切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方开口距离
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算右侧最短距离
        close_right = min(top_right_len, bottom_right_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, open_bottom, tensor1[0], close_right]
        
    # 当上方开口朝右&下方开口朝右时,计算单开口式切图的四周距离
    if top_orient == bot_orient == 'rig':
        # 计算上方开口距离 - 两方开口向右切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方开口距离
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算左侧最短距离
        close_left = min(top_left_len, bottom_left_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, open_bottom, close_left, width - tensor1[2]]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def top_left_only(tensor1, res, height, width): # 上方和左侧存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 判断右下方是否存在目标
    if bottom_right_len == width - tensor1[2] and right_bottom_len == height - tensor1[3]: # 当右下方不存在目标时
        # 组装需要返回的直下右式切图距离
        list1 = [min(top, right_top_len), height - tensor1[3], min(left, top_left_len, bottom_left_len), width - tensor1[2]]
    else: # bottom_right_len != width - tensor1[2] or right_bottom_len != height - tensor1[3] 当右下方存在目标时
        # 组装需要返回的直下式切图距离
        list2 = [min(top, left_top_len, right_top_len), height - tensor1[3], min(left, top_left_len, bottom_left_len), bottom_right_len]
        # 组装需要返回的直右式切图距离
        list3 = [min(top, left_top_len, right_top_len), right_bottom_len, min(left, top_left_len, bottom_left_len), width - tensor1[2]]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def top_right_only(tensor1, res, height, width): # 上方和右侧存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 判断左下方是否存在目标
    if left_bottom_len == height - tensor1[3] and bottom_left_len == tensor1[0]: # 当左下方不存在目标时
        # 组装需要返回的直下左式切图距离
        list1 = [min(top, left_top_len), height - tensor1[3], tensor1[0], min(right, bottom_right_len, top_right_len)]
    else: # left_bottom_len != height - tensor1[3] or bottom_left_len != tensor1[0] 当左下方不存在目标时
        # 组装需要返回的直下式切图距离
        list2 = [min(top, left_top_len), height - tensor1[3], bottom_left_len, min(right, bottom_right_len, top_right_len)]
        # 组装需要返回的直左式切图距离
        list3 = [min(top, left_top_len), left_bottom_len, tensor1[0], min(right, bottom_right_len, top_right_len)]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def bottom_left_only(tensor1, res, height, width): # 下方和左侧存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 判断右上方是否存在目标
    if top_right_len == width - tensor1[2] and right_top_len == tensor1[1]: # 当右上方不存在目标时
        # 组装需要返回的直上右式切图距离
        list1 = [tensor1[1], min(bottom, right_bottom_len), min(left, top_left_len, bottom_left_len), (width - tensor1[2])]
    else: # top_right_len != width - tensor1[2] or right_top_len != tensor1[1] 当右上方存在目标时
        # 组装需要返回的直右式切图距离
        list2 = [right_top_len, min(bottom, right_bottom_len), min(left, top_left_len, bottom_left_len), (width - tensor1[2])]
        # 组装需要返回的直上式切图距离
        list3 = [tensor1[1], min(bottom, right_bottom_len), min(left, top_left_len, bottom_left_len), top_right_len]
    
    return list1, list2, list3, []
    
    # return tensor1[1], top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def bottom_right_only(tensor1, res, height, width): # 下方和右方存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 判断左上方是否存在目标
    if top_left_len == tensor1[0] and left_top_len == tensor1[1]: # 当左上方不存在目标时
        # 组装需要返回的直上左式切图距离
        list1 = [tensor1[1], min(bottom, left_bottom_len), tensor1[0], min(right, top_right_len, bottom_right_len)]
    else: # top_left_len != tensor1[0] or left_top_len != tensor1[1] 当左上方存在目标时
        # 组装需要返回的直左式切图距离
        list2 = [left_top_len, min(bottom, left_bottom_len), tensor1[0], min(right, top_right_len, bottom_right_len)]
        # 组装需要返回的直上式切图距离
        list3 = [tensor1[1], min(bottom, left_bottom_len, right_bottom_len), top_left_len, min(right, top_right_len)]
    
    return list1, list2, list3, []
    
    # return tensor1[1], top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def left_right_only(tensor1, res, height, width): # 左侧和右侧存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直上下式切图距离
    list1 = [tensor1[1], height - tensor1[3], min(left, top_left_len, bottom_left_len), min(right, top_right_len, bottom_right_len)]
    
    # 排除左上或左下无目标&左侧目标距基准目标最近的情况
    if left != min(left, top_left_len, bottom_left_len):
        # 判断单开口式切图左侧的朝向
        if top_left_len > bottom_left_len: # 左侧开口朝上
            left_orient = 'top'
        else: # top_left_len < bottom_left_len 左侧开口朝下
            left_orient = 'bot'
    else:
        left_orient = 'Non'
    
    # 排除右上或右下无目标&右侧目标距基准目标最近的情况
    if right != min(right, top_right_len, bottom_right_len):
        # 判断单开口式切图右侧的朝向
        if top_right_len > bottom_right_len: # 右侧开口朝上
            right_orient = 'top'
        else: # top_right_len < bottom_right_len 右侧开口朝下
            right_orient = 'bot'
    else:
        right_orient = 'Non'
    
    # 当左侧开口朝上时计算开口式切图的四周距离
    if left_orient != right_orient and left_orient == 'top': 
        # 计算左侧开口距离 - 左侧开口向上切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧闭口距离
        close_right = min(right, top_right_len, bottom_right_len)
        
        # 计算底面最短距离
        close_bottom = min(left_bottom_len, right_bottom_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [tensor1[1], close_bottom, open_left, close_right]
    
    # 当左侧开口朝下时计算单开口式切图的四周距离
    if left_orient != right_orient and left_orient == 'bot':
        # 计算左侧开口距离 - 左侧开口向下切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧闭口距离
        close_right = min(right, top_right_len, bottom_right_len) # 右方放松,因为选择上方收紧
        
        # 计算顶面最短距离
        close_top = min(left_top_len, right_top_len) # 上方收紧
        
        # 组装需要返回的开口式切图距离
        list2 = [close_top, height - tensor1[3], open_left, close_right]
    
    # 当右侧开口朝上时计算单开口式切图的四周距离
    if right_orient != left_orient and right_orient == 'top':
        # 计算左侧闭口距离 - 右侧开口向上切图
        close_left = min(left, top_left_len, bottom_left_len)
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算底面最短距离
        close_bottom = min(left_bottom_len, right_bottom_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [tensor1[1], close_bottom, close_left, open_right]
    
    # 当右侧开口朝下时计算开口式切图的四周距离
    if right_orient != left_orient and right_orient == 'bot':
        # 计算左侧闭口距离 - 右侧开口向下切图
        close_left = min(left, bottom_left_len, top_left_len)
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算顶面最短距离
        close_top = min(left_top_len, right_top_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, height - tensor1[3], close_left, open_right]
    
    # 当左右开口方向均向上时计算开口式切图的四周距离
    if left_orient == right_orient == 'top':
        # 计算左侧开口距离 - 左侧开口向上切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算底面最短距离
        close_bottom = min(left_bottom_len, right_bottom_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [tensor1[1], close_bottom, open_left, open_right]
    
    # 当左右开口方向均向下时计算开口式切图的四周距离
    if left_orient == right_orient == 'bot':
        # 计算左侧开口距离 - 左侧开口向下切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算顶面最短距离
        close_top = min(left_top_len, right_top_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, height - tensor1[3], open_left, open_right]
    
    return list1, list2, list3, []
    
    # return tensor1[1], top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def top_bottom_left(tensor1, res, height, width): # 上方和下方和左侧存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方无障碍时,计算该目标右上右下各个目标与其距离
    right_top_len, right_bottom_len = right_side_no_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直左右式切图距离
    list1 = [min(top, right_top_len), min(bottom, right_bottom_len), min(left, top_left_len, bottom_left_len), width - tensor1[2]]
    
    # 排除左上或右上无目标&上方目标距基准目标最近的情况
    if top != min(top, left_top_len, right_top_len):
        # 判断单开口式切图上方的朝向
        if left_top_len > right_top_len: # 上方开口朝左
            top_orient = 'lef'
        else: # left_top_len < right_top_len 上方开口朝右
            top_orient = 'rig'
    else:
        top_orient = 'Non'
    
    # 排除左下或右下无目标&上方目标距基准目标最近的情况
    if bottom != min(bottom, left_bottom_len, right_bottom_len):
        # 判断单开口式切图下方的朝向
        if left_bottom_len > right_bottom_len: # 下方开口朝左
            bot_orient = 'lef'
        else: # left_bottom_len < right_bottom_len 下方开口朝右
            bot_orient = 'rig'
    else:
        bot_orient = 'Non'
    
    # 当上方开口朝左时计算开口式切图的四周距离
    if top_orient != bot_orient and top_orient == 'lef':
        # 计算上方开口距离 - 上方开口向左切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方闭口距离
        close_bottom = min(bottom, left_bottom_len, right_bottom_len)
        
        # 计算右侧最短距离
        close_right = min(top_right_len, bottom_right_len)
        
        # 计算左侧最短距离
        close_left = min(left, top_left_len, bottom_left_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, close_bottom, close_left, close_right]
    
    # 当下方开口朝左时计算开口式切图的四周距离
    if bot_orient != top_orient and bot_orient == 'lef':
        # 计算下方开口距离 - 下方开口向左切图
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算上方闭口距离
        close_top = min(top, left_top_len, right_top_len)
        
        # 计算右侧最短距离
        close_right = min(top_right_len, bottom_right_len)
        
        # 计算左侧最短距离
        close_left = min(left, top_left_len, bottom_left_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, open_bottom, close_left, close_right]
    
    # 当上方开口朝左&下方开口朝左时,计算单开口式切图的四周距离
    if top_orient == bot_orient == 'lef':
        # 计算上方开口距离 - 两方开口向左切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方开口距离
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算右侧最短距离
        close_right = min(top_right_len, bottom_right_len)
        
        # 计算左侧最短距离
        close_left = min(left, top_left_len, bottom_left_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, open_bottom, close_left, close_right]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, width - tensor1[2], right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def top_bottom_right(tensor1, res, height, width): # 上方和下方和右侧存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左方无障碍时,计算该目标左上左下各个目标与其距离
    left_top_len, left_bottom_len = left_side_no_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直左右式切图距离
    list1 = [min(top, left_top_len), min(bottom, left_bottom_len), tensor1[0], min(right, top_right_len, bottom_right_len)]
    
    # 排除左上或右上无目标&上方目标距基准目标最近的情况
    if top != min(top, left_top_len, right_top_len):
        # 判断单开口式切图上方的朝向
        if left_top_len > right_top_len: # 上方开口朝左
            top_orient = 'lef'
        else: # left_top_len < right_top_len 上方开口朝右
            top_orient = 'rig'
    else:
        top_orient = 'Non'
    
    # 排除左下或右下无目标&上方目标距基准目标最近的情况
    if bottom != min(bottom, left_bottom_len, right_bottom_len):
        # 判断单开口式切图下方的朝向
        if left_bottom_len > right_bottom_len: # 下方开口朝左
            bot_orient = 'lef'
        else: # left_bottom_len < right_bottom_len 下方开口朝右
            bot_orient = 'rig'
    else:
        bot_orient = 'Non'
    
    # 当上方开口朝右时计算开口式切图的四周距离
    if top_orient != bot_orient and top_orient == 'rig':
        # 计算上方开口距离 - 上方开口向右切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方闭口距离
        close_bottom = min(bottom, left_bottom_len, right_bottom_len)
        
        # 计算左侧最短距离
        close_left = min(top_left_len, bottom_left_len)
        
        # 计算右侧最短距离
        close_right = min(right, top_right_len, bottom_right_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, close_bottom, close_left, close_right]
    
    # 当下方开口朝右时计算开口式切图的四周距离
    if bot_orient != top_orient and bot_orient == 'rig':
        # 计算下方开口距离 - 下方开口向右切图
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算上方闭口距离
        close_top = min(top, left_top_len, right_top_len)
        
        # 计算左侧最短距离
        close_left = min(top_left_len, bottom_left_len)
        
        # 计算右侧最短距离
        close_right = min(right, top_right_len, bottom_right_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, open_bottom, close_left, close_right]
    
    # 当上方开口朝右&下方开口朝右时,计算单开口式切图的四周距离
    if top_orient == bot_orient == 'rig':
        # 计算上方开口距离 - 两方开口向右切图
        if left_top_len > right_top_len: # 返回max值
            if left_top_len > top: # 返回min值
                open_top = top
            else: # left_top_len < top 返回min值
                open_top = left_top_len
        else: # left_top_len < right_top_len 返回max值
            if right_top_len > top: # 返回min值
                open_top = top
            else: # right_top_len < top 返回min值
                open_top = right_top_len
        
        # 计算下方开口距离
        if left_bottom_len > right_bottom_len: # 返回max值
            if left_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # left_bottom_len < bottom 返回min值
                open_bottom = left_bottom_len
        else: # left_bottom_len < right_bottom_len 返回max值
            if right_bottom_len > bottom: # 返回min值
                open_bottom = bottom
            else: # right_bottom_len < bottom 返回min值
                open_bottom = right_bottom_len
        
        # 计算左侧最短距离
        close_left = min(top_left_len, bottom_left_len)
        
        # 计算右侧最短距离
        close_right = min(right, top_right_len, bottom_right_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [open_top, open_bottom, close_left, close_right]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, tensor1[0], left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def top_left_right(tensor1, res, height, width): # 上方和左侧和右侧存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方无障碍时,计算该目标左下右下各个目标与其距离
    bottom_left_len, bottom_right_len = bottom_side_no_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直上式切图距离
    list1 = [min(top, left_top_len, right_top_len), height - tensor1[3], min(left, bottom_left_len), min(right, bottom_right_len)]
    
    # 排除左上或左下无目标&左侧目标距基准目标最近的情况
    if left != min(left, top_left_len, bottom_left_len):
        # 判断单开口式切图左侧的朝向
        if top_left_len > bottom_left_len: # 左侧开口朝上
            left_orient = 'top'
        else: # top_left_len < bottom_left_len 左侧开口朝下
            left_orient = 'bot'
    else:
        left_orient = 'Non'
    
    # 排除右上或右下无目标&右侧目标距基准目标最近的情况
    if right != min(right, top_right_len, bottom_right_len):
        # 判断单开口式切图右侧的朝向
        if top_right_len > bottom_right_len: # 右侧开口朝上
            right_orient = 'top'
        else: # top_right_len < bottom_right_len 右侧开口朝下
            right_orient = 'bot'
    else:
        right_orient = 'Non'
    
    # 当左侧开口朝上时计算开口式切图的四周距离
    if left_orient != right_orient and left_orient == 'top': 
        # 计算左侧开口距离 - 左侧开口向上切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧闭口距离
        close_right = min(right, top_right_len, bottom_right_len)
        
        # 计算底面最短距离
        close_bottom = min(left_bottom_len, right_bottom_len)
        
        # 计算顶面最短距离
        close_top = min(top, left_top_len, right_top_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [close_top, close_bottom, open_left, close_right]
    
    # 当右侧开口朝上时计算单开口式切图的四周距离
    if right_orient != left_orient and right_orient == 'top':
        # 计算左侧闭口距离 - 右侧开口向上切图
        close_left = min(left, top_left_len, bottom_left_len)
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算底面最短距离
        close_bottom = min(left_bottom_len, right_bottom_len)
        
        # 计算顶面最短距离
        close_top = min(top, left_top_len, right_top_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, close_bottom, close_left, open_right]
    
    # 当左右开口方向均向上时计算开口式切图的四周距离
    if left_orient == right_orient == 'top':
        # 计算左侧开口距离 - 左侧开口向上切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算底面最短距离
        close_bottom = min(left_bottom_len, right_bottom_len)
        
        # 计算顶面最短距离
        close_top = min(top, left_top_len, right_top_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, close_bottom, open_left, open_right]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, height - tensor1[3], bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def bottom_left_right(tensor1, res, height, width): # 下方和左侧和右侧存在障碍
    # 当该目标上方无障碍时,计算该目标左上右上各个目标与其距离
    top_left_len, top_right_len = top_side_no_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 组装需要返回的直上式切图距离
    list1 = [tensor1[1], min(bottom, left_bottom_len, right_bottom_len), min(left, top_left_len), min(right, top_right_len)]
    
    # 排除左上或左下无目标&左侧目标距基准目标最近的情况
    if left != min(left, top_left_len, bottom_left_len):
        # 判断单开口式切图左侧的朝向
        if top_left_len > bottom_left_len: # 左侧开口朝上
            left_orient = 'top'
        else: # top_left_len < bottom_left_len 左侧开口朝下
            left_orient = 'bot'
    else:
        left_orient = 'Non'
    
    # 排除右上或右下无目标&右侧目标距基准目标最近的情况
    if right != min(right, top_right_len, bottom_right_len):
        # 判断单开口式切图右侧的朝向
        if top_right_len > bottom_right_len: # 右侧开口朝上
            right_orient = 'top'
        else: # top_right_len < bottom_right_len 右侧开口朝下
            right_orient = 'bot'
    else:
        right_orient = 'Non'

    # 当左侧开口朝下时计算单开口式切图的四周距离
    if left_orient != right_orient and left_orient == 'bot':
        # 计算左侧开口距离 - 左侧开口向下切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧闭口距离
        close_right = min(right, top_right_len, bottom_right_len) # 右方放松,因为选择上方收紧
        
        # 计算顶面最短距离
        close_top = min(left_top_len, right_top_len) # 上方收紧
        
        # 计算底面最短距离
        close_bottom = min(bottom, left_bottom_len, right_bottom_len)
        
        # 组装需要返回的开口式切图距离
        list2 = [close_top, close_bottom, open_left, close_right]

    # 当右侧开口朝下时计算开口式切图的四周距离
    if right_orient != left_orient and right_orient == 'bot':
        # 计算左侧闭口距离 - 右侧开口向下切图
        close_left = min(left, bottom_left_len, top_left_len)
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算顶面最短距离
        close_top = min(left_top_len, right_top_len)
        
        # 计算底面最短距离
        close_bottom = min(bottom, left_bottom_len, right_bottom_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, close_bottom, close_left, open_right]
    
    # 当左右开口方向均向下时计算开口式切图的四周距离
    if left_orient == right_orient == 'bot':
        # 计算左侧开口距离 - 左侧开口向下切图
        if top_left_len > bottom_left_len: # 返回max值
            if top_left_len > left: # 返回min值
                open_left = left
            else: # top_left_len < left 返回min值
                open_left = top_left_len
        else: # top_left_len < bottom_left_len 返回max值
            if bottom_left_len > left: # 返回min值
                open_left = left
            else: # bottom_left_len < left 返回min值
                open_left = bottom_left_len
        
        # 计算右侧开口距离
        if top_right_len > bottom_right_len: # 返回max值
            if top_right_len > right: # 返回min值
                open_right = right
            else: # top_right_len < right 返回min值
                open_right = top_right_len
        else: # top_right_len < bottom_right_len 返回max值
            if bottom_right_len > right: # 返回min值
                open_right = right
            else: # bottom_right_len < right 返回min值
                open_right = bottom_right_len
        
        # 计算顶面最短距离
        close_top = min(left_top_len, right_top_len)
        
        # 计算底面最短距离
        close_bottom = min(bottom, left_bottom_len, right_bottom_len)
        
        # 组装需要返回的开口式切图距离
        list3 = [close_top, close_bottom, open_left, open_right]
    
    return list1, list2, list3, []
    
    # return tensor1[1], top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距


def four_side_has_target(tensor1, res, height, width): # 四面均存在障碍
    # 当该目标上方存在障碍时,计算该目标与障碍的距离,计算该目标左上到xmin、xmax到右上的距离
    top, top_left_len, top_right_len = top_side_has_target(tensor1, res, height, width)
    
    # 当该目标下方存在障碍时,计算该目标与障碍的距离,计算该目标左下到xmin、xmax到右下的距离
    bottom, bottom_left_len, bottom_right_len = bottom_side_has_target(tensor1, res, height, width)
    
    # 当该目标左侧存在障碍时,计算该目标与障碍的距离,计算该目标左上到ymin、ymax到左下的距离
    left, left_top_len, left_bottom_len = left_side_has_target(tensor1, res, height, width)
    
    # 当该目标右方存在障碍时,计算该目标与障碍的距离,计算该目标右上到ymin、ymax到右下的距离
    right, right_top_len, right_bottom_len = right_side_has_target(tensor1, res, height, width)
    
    # 初始化三个列表防止返回时某个列表不存在导致错误
    list1, list2, list3 = [], [], []
    
    # 计算上方最短距离
    close_top = min(top, left_top_len, right_top_len)
    
    # 计算下方最短距离
    close_bot = min(bottom, left_bottom_len, right_bottom_len)
    
    # 计算左侧最短距离
    close_lef = min(left, top_left_len, bottom_left_len)
    
    # 计算右侧最短距离
    close_rig = min(right, top_right_len, bottom_right_len)
    
    list1 = [close_top, close_bot, close_lef, close_rig]
    
    return list1, list2, list3, []
    
    # return top, top_left_len, top_right_len, bottom, bottom_left_len, bottom_right_len, left, left_top_len, left_bottom_len, right, right_top_len, right_bottom_len
    # 上距 上左距 上右距 下距 下左距 下右距 左距 左上距 左下距 右距 右上距 右下距
    
# -----------------------------------------------------------------------------
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

    # 设定类别
    VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
                   "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

    class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    
    # 设定并读取xml文件路径, 单图测试
    # label_path = 'D:/AICV-YoloXReDST-SGD/chub_lianyu_845.xml'
    # res, height, width = xml_target(label_path, class_to_ind)
    
    # 设定标签文件夹路径, 多图测试
    label_path = 'F:/VOCtrainval_xml/'
    # inter_path = 'D:/AICV-YoloXReDST-SGD/datasets_sample/Intersection/'
    
    for label_name in os.listdir(label_path):
        # 设定图片路径并读取图片宽高
        img = cv2.imread('F:/VOCtrainval_images/' + label_name[0:-4] + '.jpg')
        height, width = img.shape[0], img.shape[1]
        
        # 读取txt文件中的目标坐标, 并将[xcen, ycen, w, h]转换为[xmin, ymin, xmax, ymax]
        # res = xywh_to_xyxy(txt_target(label_path + label_name, height, width))
        
        # 读取xml文件中的目标坐标
        res, height, width = xml_target(label_path + label_name, class_to_ind)
        
        # print(res)
        
        label_flag = 0 # 初始化label_flag
        
        # 检查是否存在相交区域,如果存在相交区域则无法使用本算法
        for i in range(res.shape[0]):
            for j in range(res.shape[0]):
                if res[i].tolist() == res[j].tolist():
                    continue
                else:
                    if torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0:
                        print("Intersection area: ", label_name)
                        # shutil.move(label_path + label_name, inter_path + label_name) # old, new
                        break
            else: # for-else逻辑为执行完for就会执行else,当for无法正常执行完毕时不会执行else
                continue
            label_flag = 1 # label_flag为1则该图片存在重叠目标,需要放弃该图片
            break
        
        # 检测该图片是否需要被放弃
        if label_flag == 1:
            continue
        
        del label_flag
    
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
    
        del inner_bool
        
        # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
        four_side = []
        for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
            top, bottom, left, right = False, False, False, False # 初始化四个方向的布尔变量
            for j in range(len(outer_bool[i])): # 对每个目标的对比结果进行或操作
                top = top or outer_bool[i][j][0]
                bottom = bottom or outer_bool[i][j][1]
                left = left or outer_bool[i][j][2]
                right = right or outer_bool[i][j][3]
            four_side.append([top, bottom, left, right]) # 储存最终对比结果,即该目标四面是否存在空面
            
        """
        # 逐行打印目标的布尔列表
        for i in range(len(four_side)):
            print(four_side[i])
        """
    
        # 取得每个目标的[上距 上左距 上右距], [下距 下左距 下右距], [左距 左上距 左下距], [右距 右上距 右下距]
        expandinfo = []
        for i in range(len(four_side)):
            list1, list2, list3, list4 = check_direction2(four_side[i], res[i], res, height, width)
            expandinfo.append([list1, list2, list3, list4])

        # 分析目标数据得到多种可能的裁剪方式, 每边适当向外扩展10个像素
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
    
        # print(cutinfo)
    
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
        label_tensor = torch.from_numpy(res[:, :4])
        for i in range(res.shape[0]):
            for j in range(len(cutinfo[i])):
                img_temp = copy.deepcopy(img)
                label_temp = torch.cat((label_tensor, torch.from_numpy(np.array([cutinfo[i][j]]))), 0)
                for k in range(len(label_temp)):
                    box = label_temp[k]
                    x0 = int(box[0])
                    y0 = int(box[1])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    color = (_COLORS[k] * 255).astype(np.uint8).tolist()
                    cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
                cv2.imwrite('D:/AICV-DSTRethink/DataPoolBasicCode/Submit/' + str(label_name) + '_' + str(i) + '_' + str(j) + '.jpg', img_temp)
        
        """
        # 写入csv文件: 图片名, 目标坐标(xyxycls), 裁剪坐标(xyxy), 目标面积, 裁剪面积, 图块/目标比例
        csv_path = 'D:/AICV-DSTRethink/DataPoolBasicCode/Submit/VOC_0712trainval_ObjectInfo.csv'
        for i in range(res.shape[0]):
            for j in range(len(cutinfo[i])):
                list_info = [label_name, # os.path.basename(label_path), 
                             res[i][0], res[i][1], res[i][2], res[i][3], res[i][4], 
                             cutinfo[i][j][0], cutinfo[i][j][1], cutinfo[i][j][2], cutinfo[i][j][3], 
                             ratio[i][j][0], ratio[i][j][1], ratio[i][j][2]]
                with open(csv_path, 'a', newline="") as csvfile:
                    writer0 = csv.writer(csvfile)
                    writer0.writerow(list_info)
        