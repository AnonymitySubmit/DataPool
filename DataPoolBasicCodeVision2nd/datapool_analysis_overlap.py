# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 21:04:47 2022

@author: Cheng Yuxuan Original, Overlap Object Analyze of YoloXRe-DST-DataPool
"""

import os
import cv2
# import csv
import copy
import torch
# import torchvision
import numpy as np
# import pandas as pd
import xml.etree.ElementTree as ET

from datapool_analysis_disjoint import (check_direction1, check_direction2,
                                        top_side_has_target, bottom_side_has_target, 
                                        left_side_has_target, right_side_has_target)


def situation_2(target, list_target, height, width): # 输入该目标与该目标和其他目标的相交信息, 返回单个目标的扩展信息   
    target_bool = [] # 初始化判断该目标四边是否可扩展的矩阵, 准备做and操作
    
    # print("\ntarget: ", target)
    # print("overlap objects", list_target[1])
    
    # 判断该目标与每个目标的相交模式, 筛选是否有某条边可以扩展切割
    for i in range(len(list_target[1])): # 存在相交不等于0但小于0.1的其他目标
        target_bool.append(estimate_point(target, list_target[1][i])) # 计算该目标与每个相交目标的相交形式
    
    top, bot, lef, rig = True, True, True, True # 初始化四边布尔变量以进行和操作
    
    # print(target_bool)
    
    # 判断该目标四边是否可扩展, True为可扩展, False不可扩展
    for i in range(len(target_bool)):
        top = top and target_bool[i][0]
        bot = bot and target_bool[i][1]
        lef = lef and target_bool[i][2]
        rig = rig and target_bool[i][3]
    
    # print("Dose four side can expand? ", top, bot, lef, rig)
    # print("list_target[0]: ", list_target[0]) # []即完全没有无相交的目标
    
    if len(list_target[0]) != 0: # 还存在其他无相交目标
        return allocate_situation(target, list_target[0], top, bot, lef, rig, height, width) # 主要与list_target[0]的目标对比, 即完全无相交的其他目标
    else:
        return special_case(target, top, bot, lef, rig, height, width)


def situation_3(target, list_target, res, height, width): # 输入该目标的三个列表, 返回单个目标的扩展信息
    # 特殊情况: 与该目标严重相交的目标只有一个, 并且另一目标包含该目标, 则该目标只向外扩展10像素
    if len(list_target[2]) == 1:
        if list_target[2][0][0] <= target[0] < target[2] <= list_target[2][0][2] and  list_target[2][0][1] <= target[1] < target[3] <= list_target[2][0][3]:
            if target[0] >= 10:
                left = 10
            else:
                left = target[0]
            if target[1] >= 10:
                top = 10
            else:
                top = target[1]
            if width - target[2] >= 10:
                right = 10
            else:
                right = width - target[2]
            if height - target[3] >= 10:
                bottom = 10
            else:
                bottom = height - target[3]
            
            return target[:4], [top, bottom, left, right]

    # 总体思路是将严重相交的几个目标作为一个大目标, 然后计算这个大目标在该图片上的扩展切割信息, 首先就要取得这个大目标的坐标
    xmin, ymin, xmax, ymax = target[0], target[1], target[2], target[3] # xmin与ymin越小越好, xmax与ymax越大越好
    for i in range(len(list_target[2])):
        if xmin > list_target[2][i][0]:
            xmin = list_target[2][i][0]
        if ymin > list_target[2][i][1]:
            ymin = list_target[2][i][1]
        if xmax < list_target[2][i][2]:
            xmax = list_target[2][i][2]
        if ymax < list_target[2][i][3]:
            ymax = list_target[2][i][3]
    
    dense_obj = [xmin, ymin, xmax, ymax]
    # print("dense: ", dense_obj)
    
    # 然后再将这个大目标作为整体来与其他目标计算扩展距离, 将整张图片的目标分为两份, 一份是密集目标块, 另一份是其他目标
    other = copy.deepcopy(res.tolist())
    for i in range(len(list_target[2])):
        if list_target[2][i].tolist() in other:
            other.remove(list_target[2][i].tolist())
    
    # 把target从other中去除, 上文没有去除掉
    for i in range(len(other)):
        if target[0] == other[i][0] and target[1] == other[i][1] and target[2] == other[i][2] and target[3] == other[i][3]:
            other.remove(other[i])
            break
    
    # 此时可能出现这个大目标与其他目标严重相交, 但是没有与选定的基础目标相交, 该情况则当作轻微相交处理, 相交边不扩展
    list1, list2 = [], [] # 目标列表: 完全无相交, 存在相交
    for i in range(len(other)): # 分析大目标与每个其他目标的位置信息, 完全无相交, 存在相交
        if overlap(dense_obj, other[i][:4]) == 0: # torchvision.ops.box_iou(torch.from_numpy(np.array(dense_obj)).unsqueeze(0), torch.from_numpy(np.array(other[i][:4])).unsqueeze(0)) == 0:
            list1.append(other[i]) # 完全无相交的目标
        else: # torchvision.ops.box_iou(torch.from_numpy(np.array(dense_obj)).unsqueeze(0), torch.from_numpy(np.array(other[i][:4])).unsqueeze(0)) > 0:
            list2.append(other[i]) # 存在相交的目标
    
    return dense_obj, situation_2(dense_obj, [list1, list2], height, width)


def estimate_point(tensor1, tensor2):
    upper_left, upper_right = [tensor1[0], tensor1[1]], [tensor1[2], tensor1[1]]
    bottom_left, bottom_right = [tensor1[0], tensor1[3]], [tensor1[2], tensor1[3]]
    
    # 检查左上点是否在tensor2中, 即左上点是否处于tensor2宽高坐标范围
    if tensor2[0] < upper_left[0] < tensor2[2] and tensor2[1] < upper_left[1] < tensor2[3]:
        upper_lef = True # 左上点在其他目标内则取True
    else:
        upper_lef = False
    
    # 检查右上点是否在tensor2中, 即右上点是否处于tensor2宽高坐标范围
    if tensor2[0] < upper_right[0] < tensor2[2] and tensor2[1] < upper_right[1] < tensor2[3]:
        upper_rig = True # 右上点在其他目标内则取True
    else:
        upper_rig = False
    
    # 检查左下点是否在tensor2中, 即左下点是否处于tensor2宽高坐标范围
    if tensor2[0] < bottom_left[0] < tensor2[2] and tensor2[1] < bottom_left[1] < tensor2[3]:
        bottom_lef = True # 左下点在其他目标内则取True
    else:
        bottom_lef = False
    
    # 检查右下点是否在tensor2中, 即右下点是否处于tensor2宽高坐标范围
    if tensor2[0] < bottom_right[0] < tensor2[2] and tensor2[1] < bottom_right[1] < tensor2[3]:
        bottom_rig = True # 右下点在其他目标内则取True
    else:
        bottom_rig = False
    
    # print(upper_lef, upper_rig, bottom_lef, bottom_rig)
    
    # 分析该目标四边是否可扩展, 返回值为四边布尔变量, [top, bot, lef, rig], True则可扩展, False不可扩展
    if upper_lef == True and upper_rig == True and bottom_lef == True and bottom_rig == True:
        return [False, False, False, False] # 四个点均在其他目标内, 四边均不可扩展, 这种情况应该不会出现
    
    if upper_lef == True and upper_rig == False and bottom_lef == False and bottom_rig == False:
        return [False, True, False, True] # 左上点在其他目标内, 其他点不在其他目标内, 下边、右边无相交
    
    if upper_lef == False and upper_rig == True and bottom_lef == False and bottom_rig == False:
        return [False, True, True, False] # 右上点在其他目标内, 其他点不在其他目标内, 下边、左边无相交
    
    if upper_lef == False and upper_rig == False and bottom_lef == True and bottom_rig == False:
        return [True, False, False, True] # 左下点在其他目标内, 其他点不在其他目标内, 上边、右边无相交
    
    if upper_lef == False and upper_rig == False and bottom_lef == False and bottom_rig == True:
        return [True, False, True, False] # 右下点在其他目标内, 其他点不在其他目标内, 上边、左边无相交
    
    if upper_lef == True and upper_rig == True and bottom_lef == False and bottom_rig == False:
        return [False, True, False, False] # 左上和右上点在其他目标内, 其他点不在其他目标内, 下边无相交
    
    if upper_lef == True and upper_rig == False and bottom_lef == True and bottom_rig == False:
        return [False, False, False, True] # 左上和左下点在其他目标内, 其他点不在其他目标内, 右边无相交
    
    if upper_lef == True and upper_rig == False and bottom_lef == False and bottom_rig == True:
        return [False, False, False, False] # 左上和右下点在其他目标内, 其他点不在其他目标内, 四边均相交
    
    if upper_lef == False and upper_rig == True and bottom_lef == True and bottom_rig == False:
        return [False, False, False, False] # 右上和左下点在其他目标内, 其他点不在其他目标内, 四边均相交
    
    if upper_lef == False and upper_rig == True and bottom_lef == False and bottom_rig == True:
        return [False, False, True, False] # 右上和右下点在其他目标内, 其他点不在其他目标内, 左边无相交
    
    if upper_lef == False and upper_rig == False and bottom_lef == True and bottom_rig == True:
        return [True, False, False, False] # 左下和右下点在其他目标内, 其他点不在其他目标内, 上边无相交
    
    if upper_lef == True and upper_rig == True and bottom_lef == True and bottom_rig == False:
        return [False, False, False, False] # 左上右上左下点在其他目标内, 右下点不在其他目标内, 四边均相交
    
    if upper_lef == True and upper_rig == True and bottom_lef == False and bottom_rig == True:
        return [False, False, False, False] # 左上右上右下点在其他目标内, 左下点不在其他目标内, 四边均相交
    
    if upper_lef == True and upper_rig == False and bottom_lef == True and bottom_rig == True:
        return [False, False, False, False] # 左上左下右下点在其他目标内, 右上点不在其他目标内, 四边均相交
    
    if upper_lef == False and upper_rig == True and bottom_lef == True and bottom_rig == True:
        return [False, False, False, False] # 右上左下右下点在其他目标内, 左上点不在其他目标内, 四边均相交
    
    if upper_lef == False and upper_rig == False and bottom_lef == False and bottom_rig == False:
        return estimate_side(tensor1, tensor2) # 四个点均不在其他目标内, 需要进一步分析边相交情况


def estimate_side(tensor1, tensor2): # 该目标没有角在另一个目标内, 但可能存在某条边与其他目标相交
    # print("tensor1: ", tensor1)
    # print("tensor2: ", tensor2)

    # 该目标的顶边与其他目标相交, [top, bot, lef, rig], True则可扩展, False不可扩展
    if tensor1[0] <= tensor2[0] < tensor2[2] <= tensor1[2] and tensor2[1] <= tensor1[1] < tensor2[3] <= tensor1[3]:
        # print("tensor1 top side intersect with tensor2")
        return [False, True, True, True]
    
    # 该目标的底边与其他目标相交, [top, bot, lef, rig], True则可扩展, False不可扩展
    if tensor1[0] <= tensor2[0] < tensor2[2] <= tensor1[2] and tensor1[1] < tensor2[1] < tensor1[3] < tensor2[3]:
        # print("tensor1 bottom side intersect with tensor2")
        return [True, False, True, True]
    
    # 该目标的左边与其他目标相交, [top, bot, lef, rig], True则可扩展, False不可扩展
    if tensor2[0] < tensor1[0] < tensor2[2] < tensor1[2] and tensor1[1] <= tensor2[1] < tensor2[3] <= tensor1[3]:
        # print("tensor1 left side intersect with tensor2")
        return [True, True, False, True]
    
    # 该目标的右边与其他目标相交, [top, bot, lef, rig], True则可扩展, False不可扩展
    if tensor1[0] < tensor2[0] < tensor1[2] < tensor2[2] and tensor1[1] <= tensor2[1] < tensor2[3] <= tensor1[3]:
        # print("tensor1 right side intersect with tensor2")
        return [True, True, True, False]
    
    # 该目标的顶边底边与其他目标相交, [top, bot, lef, rig], True则可扩展, False不可扩展
    if tensor1[0] <= tensor2[0] < tensor2[2] <= tensor1[2] and tensor2[1] <= tensor1[1] < tensor1[3] <= tensor2[3]:
        # print("tensor1 top & bottom side intersect with tensor2")
        return [False, False, True, True]
    
    # 该目标的左边右边与其他目标相交, [top, bot, lef, rig], True则可扩展, False不可扩展
    if tensor2[0] <= tensor1[0] < tensor1[2] <= tensor2[2] and tensor1[1] <= tensor2[1] < tensor2[3] <= tensor1[3]:
        # print("tensor1 left & right side intersect with tensor2")
        return [True, True, False, False]
    
    # 该目标完全包括另一目标, [top, bot, lef, rig], True则可扩展, False不可扩展
    if tensor1[0] <= tensor2[0] < tensor2[2] <= tensor1[2] and tensor1[1] <= tensor2[1] < tensor2[3] <= tensor1[3]:
        # print("tensor1 contain tensor2")
        return [False, False, False, False]


def special_case(target, top, bot, lef, rig, height, width):
    if top == True:
        t = target[1]
    else:
        t = 0
    
    if bot == True:
        b = height - target[3]
    else:
        b = 0
    
    if lef == True:
        l = target[0]
    else:
        l = 0
    
    if rig == True:
        r = width - target[2]
    else:
        r = 0
    
    return [t, b, l, r]

"""
def allocate_situation(target, noneoverlap, top, bot, lef, rig, height, width): # True为可扩展即无交叉, False不可扩展即有交叉
    print("fuck", top, bot, lef, rig) # False True False True
    # 先提取目标中可以扩展的面，然后对比无相交的目标，取得无相交目标的方位
    boollist_temp = []
    for i in range(len(noneoverlap)):
        boollist_temp.append(check_direction1(target, noneoverlap[i])) # check_direction1中True为存在目标, False为不存在目标
    
    top, bot, lef, rig = not top, not bot, not lef, not rig
    
    for i in range(len(boollist_temp)):
        top = top or boollist_temp[i][0]
        bot = bot or boollist_temp[i][1]
        lef = lef or boollist_temp[i][2]
        rig = rig or boollist_temp[i][3]
    
    print("\nallocate: ", top, bot, lef, rig)
    
    list1, list2, list3, list4 = check_direction2([top, bot, lef, rig], target, noneoverlap, height, width)
    
    return [list1, list2, list3, list4]
"""

def allocate_situation(target, list1, top, bot, lef, rig, height, width):
    # 四面均不可扩展, 即切割目标本身即可
    if [top, bot, lef, rig] == [False, False, False, False]:
        return [0, 0, 0, 0]
    
    # 顶面可以扩展, 其他三面不可扩展
    if [top, bot, lef, rig] == [True, False, False, False]:
        return [oneside_expand(target, list1, 'top', height, width), 0, 0, 0]
        
    # 底面可以扩展, 其他三面不可扩展
    if [top, bot, lef, rig] == [False, True, False, False]:
        return [0, oneside_expand(target, list1, 'bot', height, width), 0, 0]
        
    # 左面可以扩展, 其他三面不可扩展
    if [top, bot, lef, rig] == [False, False, True, False]:
        return [0, 0, oneside_expand(target, list1, 'lef', height, width), 0]
        
    # 右面可以扩展, 其他三面不可扩展
    if [top, bot, lef, rig] == [False, False, False, True]:
        return [0, 0, 0, oneside_expand(target, list1, 'rig', height, width)]
        
    # 顶面底面可以扩展, 其他二面不可扩展
    if [top, bot, lef, rig] == [True, True, False, False]:
        return twoside_expand(target, list1, 'top', 'bot', height, width)
        
    # 顶面左面可以扩展, 其他二面不可扩展
    if [top, bot, lef, rig] == [True, False, True, False]:
        return twoside_expand(target, list1, 'top', 'lef', height, width)
        
    # 顶面右面可以扩展, 其他二面不可扩展
    if [top, bot, lef, rig] == [True, False, False, True]:
        return twoside_expand(target, list1, 'top', 'rig', height, width)
        
    # 底面左面可以扩展, 其他二面不可扩展
    if [top, bot, lef, rig] == [False, True, True, False]:
        return twoside_expand(target, list1, 'bot', 'lef', height, width)
        
    # 底面右面可以扩展, 其他二面不可扩展
    if [top, bot, lef, rig] == [False, True, False, True]:
        return twoside_expand(target, list1, 'bot', 'rig', height, width)
        
    # 左面右面可以扩展, 其他二面不可扩展
    if [top, bot, lef, rig] == [False, False, True, True]:
        return twoside_expand(target, list1, 'lef', 'rig', height, width)
    
    # 顶面底面左面可以扩展, 右面不可扩展
    if [top, bot, lef, rig] == [True, True, True, False]:
        return threeside_expand(target, list1, 'rig', height, width)
        
    # 顶面底面右面可以扩展, 左面不可扩展
    if [top, bot, lef, rig] == [True, True, False, True]:
        return threeside_expand(target, list1, 'lef', height, width)
        
    # 顶面左面右面可以扩展, 底面不可扩展
    if [top, bot, lef, rig] == [True, False, True, True]:
        return threeside_expand(target, list1, 'bot', height, width)
        
    # 底面左面右面可以扩展, 顶面不可扩展
    if [top, bot, lef, rig] == [False, True, True, True]:
        return threeside_expand(target, list1, 'top', height, width)
        
    # 四面均可扩展, 但不太可能有这种情况
    if [top, bot, lef, rig] == [True, True, True, True]:
        return fourside_expand(target, list1, height, width)

def oneside_expand(target, list1, side, height, width): # 计算裁剪距离的主要函数之一
    if side == 'top':
        return topside_expand(target, list1, height, width)
    
    if side == 'bot':
        return botside_expand(target, list1, height, width)
    
    if side == 'lef':
        return lefside_expand(target, list1, height, width)
    
    if side == 'rig':
        return rigside_expand(target, list1, height, width)


def twoside_expand(target, list1, side1, side2, height, width): # 计算裁剪距离的主要函数之一
    if side1 == 'top' and side2 == 'bot':
        return [topside_expand(target, list1, height, width), botside_expand(target, list1, height, width), 0, 0]
        
    if side1 == 'top' and side2 == 'lef':
        return toplef_expand(target, list1, height, width)
        
    if side1 == 'top' and side2 == 'rig':
        return toprig_expand(target, list1, height, width)
        
    if side1 == 'bot' and side2 == 'lef':
        return botlef_expand(target, list1, height, width)
    
    if side1 == 'bot' and side2 == 'rig':
        return botrig_expand(target, list1, height, width)
        
    if side1 == 'lef' and side2 == 'rig':
        return [0, 0, lefside_expand(target, list1, height, width), rigside_expand(target, list1, height, width)]


def toplef_expand(target, list1, height, width): # 该函数属于twoside_expand的下属函数, 只需要考虑正上、正左、左上的目标
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[1] > list1[i][3]: # x2 > x'1 & x1 < x'2 & y1 > y'2
            top = True
        else:
            top = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[0] > list1[i][2]: # y2 > y'1 & y1 < y'2 & x1 > x'2
            left = True
        else:
            left = False
        outer_bool.append([top, left]) # 储存该基准目标与其他所有目标的对比结果
        
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    top, left = False, False # 初始化2个方向的布尔变量
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        top = top or outer_bool[i][0]
        left = left or outer_bool[i][1]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的上距, 上左距, 左距 左上距, 存在目标为True, 不存在目标为False
    if top == True and left == True:
        top_min_distence, top_left_distance = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(top_min_distence, left_top_distance), 0, min(left_min_distance, top_left_distance), 0]
        
    if top == True and left == False:
        top_min_distence, top_left_distance = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        if top_min_distence > top_left_distance:
            return [[top_min_distence, 0, left_top_distance, 0], [top_left_distance, 0, target[0], 0]]
        else:
            return [top_min_distence, 0, target[0], 0]
        
    if top == False and left == True:
        top_min_distence, top_left_distance = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        if left_min_distance > left_top_distance:
            return [[top_left_distance, 0, left_min_distance, 0], [target[1], 0, left_top_distance, 0]]
        else:
            return [target[1], 0, left_min_distance, 0]
        
    if top == False and left == False:
        top_min_distence, top_left_distance = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        return [[target[1], 0, left_top_distance, 0], [top_left_distance, 0, target[0], 0]]


def toprig_expand(target, list1, height, width): # 该函数属于twoside_expand的下属函数, 只需要考虑正上、正右、右上的目标
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[1] > list1[i][3]: # x2 > x'1 & x1 < x'2 & y1 > y'2
            top = True
        else:
            top = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[2] < list1[i][0]: # y2 > y'1 & y1 < y'2 & x2 < x'1
            right = True
        else:
            right = False
        outer_bool.append([top, right]) # 储存该基准目标与其他所有目标的对比结果
        
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    top, right = False, False
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        top = top or outer_bool[i][0]
        right = right or outer_bool[i][1]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的上距, 上右距, 右距 右上距, 存在目标为True, 不存在目标为False
    if top == True and right == True:
        top_min_distence, top_right_distance = top_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(top_min_distence, right_top_distance), 0, 0, min (right_min_distance, top_right_distance)]
        
    if top == True and right == False:
        top_min_distence, top_right_distance = top_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        if top_min_distence > top_right_distance:
            return [[top_min_distence, 0, 0, right_top_distance], [top_right_distance, 0, 0, width - target[2]]]
        else:
            return [top_min_distence, 0, 0, width - target[2]]
        
    if top == False and right == True:
        top_min_distence, top_right_distance = top_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        if right_min_distance > right_top_distance:
            return [[top_right_distance, 0, 0, right_min_distance], [target[1], 0, 0, right_top_distance]]
        else:
            return [target[1], 0, 0, right_min_distance]
        
    if top == False and right == False:
        top_min_distence, top_right_distance = top_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        return [[target[1], 0, 0, right_top_distance], [top_right_distance, 0, 0, width - target[2]]]


def botlef_expand(target, list1, height, width): # 该函数属于twoside_expand的下属函数 只需要考虑正下、正左、左下的目标
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] >= list1[i][0] and target[0] <= list1[i][2] and target[3] <= list1[i][1]: # x2 > x'1 & x1 < x'2 & y2 < y'1
            bottom = True
        else:
            bottom = False
        if target[3] >= list1[i][1] and target[1] <= list1[i][3] and target[0] >= list1[i][2]: # y2 > y'1 & y1 < y'2 & x1 > x'2
            left = True
        else:
            left = False
        outer_bool.append([bottom, left]) # 储存该基准目标与其他所有目标的对比结果
        
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    bottom, left = False, False
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        bottom = bottom or outer_bool[i][0]
        left = left or outer_bool[i][1]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的下距, 下左距, 左距, 左下距, 存在目标为True, 不存在目标为False
    if bottom == True and left == True:
        bottom_min_distance, bottom_left_distance = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, lef_bottom_distance = lef_bot_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [0, min(bottom_min_distance, lef_bottom_distance), min(left_min_distance, bottom_left_distance), 0]
    
    if bottom == True and left == False:
        bottom_min_distance, bottom_left_distance = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, lef_bottom_distance = lef_bot_has_target(target, list1, side_bool, inner_bool, height, width)
        if bottom_min_distance > bottom_left_distance:
            return [[0, bottom_min_distance, lef_bottom_distance, 0], [0, bottom_left_distance, target[0], 0]]
        else:
            return [0, bottom_min_distance, target[0], 0]
        
    if bottom == False and left == True:
        bottom_min_distance, bottom_left_distance = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, lef_bottom_distance = lef_bot_has_target(target, list1, side_bool, inner_bool, height, width)
        if left_min_distance > lef_bottom_distance:
            return [[0, bottom_left_distance, left_min_distance, 0], [0, height-target[3], lef_bottom_distance, 0]]
        else:
            return [0, height-target[3], left_min_distance, 0]
        
    if bottom == False and left == False:
        bottom_min_distance, bottom_left_distance = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_distance, lef_bottom_distance = lef_bot_has_target(target, list1, side_bool, inner_bool, height, width)
        return [[0, height-target[3], lef_bottom_distance, 0], [0, bottom_left_distance, target[0], 0]]


def botrig_expand(target, list1, height, width): # 该函数属于twoside_expand的下属函数, 只需要考虑正右、正下、右下是否存在目标即可
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] >= list1[i][0] and target[0] <= list1[i][2] and target[3] <= list1[i][1]: # x2 > x'1 & x1 < x'2 & y2 < y'1
            bottom = True
        else:
            bottom = False
        if target[3] >= list1[i][1] and target[1] <= list1[i][3] and target[2] <= list1[i][0]: # y2 > y'1 & y1 < y'2 & x2 < x'1
            right = True
        else:
            right = False
        outer_bool.append([bottom, right]) # 储存该基准目标与其他所有目标的对比结果
        
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    bottom, right = False, False
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        bottom = bottom or outer_bool[i][0]
        right = right or outer_bool[i][1]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的下距, 下右距, 右距, 右下距, 存在目标为True, 不存在目标为False
    if bottom == True and right == True:
        bottom_min_distance, bottom_right_distance = bot_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_bottom_distance = rig_bot_has_target(target, list1, side_bool, inner_bool, height, width)
    
        return [0, min(bottom_min_distance, right_bottom_distance), 0, min(right_min_distance, bottom_right_distance)]
    
    if bottom == True and right == False:
        bottom_min_distance, bottom_right_distance = bot_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_bottom_distance = rig_bot_has_target(target, list1, side_bool, inner_bool, height, width)
        if bottom_min_distance > bottom_right_distance:
            return [[0, bottom_min_distance, 0, right_bottom_distance], [0, bottom_right_distance, 0, width-target[2]]]
        else:
            return [0, bottom_min_distance, 0, width-target[2]]
        
    if bottom == False and right == True:
        bottom_min_distance, bottom_right_distance = bot_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_bottom_distance = rig_bot_has_target(target, list1, side_bool, inner_bool, height, width)
        if right_min_distance > right_bottom_distance:
            return [[0, bottom_right_distance, 0, right_min_distance], [0, height-target[3], 0, right_bottom_distance]]
        else:
            return [0, height-target[3], 0, right_min_distance]
        
    if bottom == False and right == False:
        bottom_min_distance, bottom_right_distance = bot_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_bottom_distance = rig_bot_has_target(target, list1, side_bool, inner_bool, height, width)
        return [[0, height-target[3], 0, right_bottom_distance], [0, bottom_right_distance, 0, width-target[2]]]


def threeside_expand(target, list1, unside, height, width): # 计算裁剪距离的主要函数之一
    if unside == 'rig':
        return topbotlef_expand(target, list1, height, width)
    
    if unside == 'lef':
        return topbotrig_expand(target, list1, height, width)
    
    if unside == 'bot':
        return toplefrig_expand(target, list1, height, width)
        
    if unside == 'top':
        return botlefrig_expand(target, list1, height, width)


def topbotlef_expand(target, list1, height, width): # 该函数属于threeside_expand的下属函数
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[1] > list1[i][3]: # x2 > x'1 & x1 < x'2 & y1 > y'2
            top = True
        else:
            top = False
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[3] < list1[i][1]: # x2 > x'1 & x1 < x'2 & y2 < y'1
            bottom = True
        else:
            bottom = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[0] > list1[i][2]: # y2 > y'1 & y1 < y'2 & x1 > x'2
            left = True
        else:
            left = False
        outer_bool.append([top, bottom, left])
    
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    top, bottom, left = False, False, False
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        top = top or outer_bool[i][0]
        bottom = bottom or outer_bool[i][1]
        left = left or outer_bool[i][2]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的上距, 上左距, 下距, 下左距, 左距, 左上距, 左下距
    if top == True and bottom == True and left == True:
        top_min_len, top_left_len = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        bottom_min_len, bottom_left_len = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        
        return [min(top_min_len, left_top_len), min(bottom_min_len, left_bottom_len), min(left_min_len, top_left_len, bottom_left_len), 0]
    
    if top == True and bottom == True and left == False:
        top_min_len, top_left_len = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        bottom_min_len, bottom_left_len = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [top_min_len, bottom_min_len, min(target[0], top_left_len, bottom_left_len), 0]
        
    if top == True and bottom == False and left == True:
        top_min_len, top_left_len = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        
        return [min(top_min_len, left_top_len), min(height-target[3], left_bottom_len), min(left_min_len, top_left_len), 0]
        
    if top == True and bottom == False and left == False:
        top_min_len, top_left_len = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [top_min_len, height-target[3], min(target[0], top_left_len), 0]
        
    if top == False and bottom == True and left == True:
        bottom_min_len, bottom_left_len = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        
        return [min(target[1], left_top_len), min(bottom_min_len, left_bottom_len), min(left_min_len, bottom_left_len), 0]
        
    if top == False and bottom == False and left == True:
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        
        return [min(target[1], left_top_len), min(height-target[3], left_bottom_len), left_min_len, 0]
        
    if top == False and bottom == True and left == False:
        bottom_min_len, bottom_left_len = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [target[1], bottom_min_len, min(target[0], bottom_left_len), 0]
        
    if top == False and bottom == False and left == False:
        return [target[1], height-target[3], target[0], 0]
        

def topbotrig_expand(target, list1, height, width): # 该函数属于threeside_expand的下属函数
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[1] > list1[i][3]: # x2 > x'1 & x1 < x'2 & y1 > y'2
            top = True
        else:
            top = False
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[3] < list1[i][1]: # x2 > x'1 & x1 < x'2 & y2 < y'1
            bottom = True
        else:
            bottom = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[2] < list1[i][0]: # y2 > y'1 & y1 < y'2 & x2 < x'1
            right = True
        else:
            right = False
        outer_bool.append([top, bottom, right])
    
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    top, bottom, right = False, False, False
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        top = top or outer_bool[i][0]
        bottom = bottom or outer_bool[i][1]
        right = right or outer_bool[i][2]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的上右距离, 下距, 下右距, 右距, 右上距, 右下距
    if top == True and bottom == True and right == True:
        top_min_len, top_right_len = top_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        bottom_min_len, bottom_right_len = bot_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [min(top_min_len, right_top_len), min(bottom_min_len, right_bottom_len), 0, min(right_min_len, top_right_len, bottom_right_len)]
    
    if top == True and bottom == True and right == False:
        top_min_len, top_right_len = top_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        bottom_min_len, bottom_right_len = bot_lef_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [top_min_len, bottom_min_len, 0, min(width-target[2], top_right_len, bottom_right_len)]
        
    if top == True and bottom == False and right == True:
        top_min_len, top_right_len = top_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [min(top_min_len, right_top_len), min(height-target[3], right_bottom_len), 0, min(right_min_len, top_right_len)]
        
    if top == True and bottom == False and right == False:
        top_min_len, top_right_len = top_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [top_min_len, height-target[3], 0, min(width-target[2], top_right_len)]
        
    if top == False and bottom == True and right == True:
        bottom_min_len, bottom_right_len = bot_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [min(target[1], right_top_len), min(bottom_min_len, right_bottom_len), 0, min(right_min_len, bottom_right_len)]
    
    if top == False and bottom == False and right == True:
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [min(target[1], right_top_len), min(height-target[3], right_bottom_len), 0, right_min_len]
        
    if top == False and bottom == True and right == False:
        bottom_min_len, bottom_right_len = bot_rig_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [target[1], bottom_min_len, 0, min(target[0], bottom_right_len)]
    
    if top == False and bottom == False and right == False:
        return [target[1], height-target[3], 0, width-target[2]]
        
    
def toplefrig_expand(target, list1, height, width): # 该函数属于threeside_expand的下属函数
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[1] > list1[i][3]: # x2 > x'1 & x1 < x'2 & y1 > y'2
            top = True
        else:
            top = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[0] > list1[i][2]: # y2 > y'1 & y1 < y'2 & x1 > x'2
            left = True
        else:
            left = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[2] < list1[i][0]: # y2 > y'1 & y1 < y'2 & x2 < x'1
            right = True
        else:
            right = False
        outer_bool.append([top, left, right])
    
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    top, left, right = False, False, False
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        top = top or outer_bool[i][0]
        left = left or outer_bool[i][1]
        right = right or outer_bool[i][2]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的上右距离, 下距, 下右距, 右距, 右上距, 右下距
    if top == True and left == True and right == True:
        top_min_len, top_left_len, top_right_len = top_side_has_target(target, list1, height, width)
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(top_min_len, left_top_distance, right_top_distance), 0, min(left_min_distance, top_left_len), min(right_min_distance, top_right_len)]
    
    if top == True and left == True and right == False:
        top_min_len, top_left_len, top_right_len = top_side_has_target(target, list1, height, width)
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(top_min_len, left_top_distance), 0, min(left_min_distance, top_left_len), min(width-target[2], top_right_len)]
        
    if top == True and left == False and right == True:
        top_min_len, top_left_len, top_right_len = top_side_has_target(target, list1, height, width)
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(top_min_len, right_top_distance), 0, min(target[0], top_left_len), min(right_min_distance, top_right_len)]
        
    if top == True and left == False and right == False:
        top_min_len, top_left_len, top_right_len = top_side_has_target(target, list1, height, width)
        
        return [top_min_len, 0, min(target[0], top_left_len), min(width-target[3], top_right_len)]
        
    if top == False and left == True and right == True:
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(target[1], left_top_distance, right_top_distance), 0, left_min_distance, right_min_distance]
    
    if top == False and left == False and right == True:
        right_min_distance, right_top_distance = rig_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(target[1], right_top_distance), 0, target[0], right_min_distance]
        
    if top == False and left == True and right == False:
        left_min_distance, left_top_distance = lef_top_has_target(target, list1, side_bool, inner_bool, height, width)
        
        return [min(target[1], left_top_distance), 0, left_min_distance, width-target[2]]
    
    if top == False and left == False and right == False:
        return [target[1], 0, target[0], width-target[2]]
        
    
def botlefrig_expand(target, list1, height, width): # 该函数属于threeside_expand的下属函数, 只需要考虑正左、正右、正下、左下、右下
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)):
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[3] < list1[i][1]: # x2 > x'1 & x1 < x'2 & y2 < y'1
            bottom = True
        else:
            bottom = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[0] > list1[i][2]: # y2 > y'1 & y1 < y'2 & x1 > x'2
            left = True
        else:
            left = False
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[2] < list1[i][0]: # y2 > y'1 & y1 < y'2 & x2 < x'1
            right = True
        else:
            right = False
        outer_bool.append([bottom, left, right])
    
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    bottom, left, right = False, False, False
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        bottom = bottom or outer_bool[i][0]
        left = left or outer_bool[i][1]
        right = right or outer_bool[i][2]
    
    side_bool, inner_bool = [], [] # 废物利用这两列表
    
    # 取得每个目标的上右距离, 下距, 下右距, 右距, 右上距, 右下距, True有目标, False无目标
    if bottom == True and left == True and right == True:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [0, min(bottom_min_len, left_bottom_len, right_bottom_len), min(left_min_len, bottom_left_len), min(right_min_len, bottom_right_len)]
    
    if bottom == True and left == True and right == False:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        """
        if left_bottom_len > left_min_len or bottom_left_len > bottom_min_len: # 左下距 > 左距 or 下左距 > 下距 (左下目标比左目标和下目标离中心更远)
            if bottom_min_len > bottom_right_len: # 下距 > 下右距
                return [0, bottom_min_len, left_min_len, width - target[2]]
            else: # 下距 < 下右距
                return [[0, bottom_min_len, left_min_len, right_bottom_len], [0, bottom_right_len, left_min_len, width - target[2]]]
        else: # 左下目标比左目标和下目标离中心更近
            if : # 下左距 > 下右距
                return [[0, bottom_right_len, left_min_len, ], [0, ]]
            else: # 下左距 < 下右距
                return [[0, bottom_left_len, left_min_len, width - target[2]], [0, min(bottom_left_len, bottom_right_len), left_bottom_len], width - target[2]]
        """
        return [0, min(bottom_min_len, left_bottom_len), min(left_min_len, bottom_left_len), min(width-target[2], bottom_right_len)]
    
    if bottom == True and left == False and right == True:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [0, min(bottom_min_len, right_bottom_len), min(target[0], bottom_left_len), min(right_min_len, bottom_right_len)]
    
    if bottom == True and left == False and right == False:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [0, bottom_min_len, min(target[0], bottom_left_len), min(width-target[2], bottom_right_len)]
    
    if bottom == False and left == True and right == True:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [0, min(height-target[3], left_bottom_len, right_bottom_len), left_min_len, right_min_len]
    
    if bottom == False and left == False and right == True:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [0, min(height-target[3], right_bottom_len), target[0], right_min_len]
    
    if bottom == False and left == True and right == False:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        
        return [0, min(height-target[3], left_bottom_len), left_min_len, width-target[2]]
    
    if bottom == False and left == False and right == False:
        bottom_min_len, bottom_left_len, bottom_right_len = bottom_side_has_target(target, list1, height, width)
        left_min_len, left_top_len, left_bottom_len = left_side_has_target(target, list1, height, width)
        right_min_len, right_top_len, right_bottom_len = right_side_has_target(target, list1, height, width)
        return [[0, height-target[3], left_bottom_len, right_bottom_len], [0, min(bottom_left_len, bottom_right_len), target[0], width-target[2]]]


def fourside_expand(target, list1, height, width): # 计算裁剪距离的主要函数之一
    if list1 == []:
        return [target[1], height-target[3], target[0], width-target[2]]

    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(len(list1)): # 外层for循环,选定一个目标作为基准
        outer_bool.append(check_direction1(target, list1[i])) # 储存该基准目标与其他所有目标的对比结果
    
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    four_side, top, bottom, left, right = [], False, False, False, False # 初始化四个方向的布尔变量
    for i in range(len(outer_bool)): # 对每个目标的对比结果进行或操作
        top = top or outer_bool[i][0]
        bottom = bottom or outer_bool[i][1]
        left = left or outer_bool[i][2]
        right = right or outer_bool[i][3]
    four_side.append([top, bottom, left, right]) # 储存最终对比结果,即该目标四面是否存在空面
    
    # 取得每个目标的[上距 上左距 上右距], [下距 下左距 下右距], [左距 左上距 左下距], [右距 右上距 右下距]
    expandinfo = []
    for i in range(len(four_side)):
        list1, list2, list3, list4 = check_direction2(four_side[i], target, list1, height, width)
        expandinfo.append(list1)
        expandinfo.append(list2)
        expandinfo.append(list3)
        expandinfo.append(list4)

    return expandinfo


def topside_expand(target, list1, height, width): # 该函数属于toplef_expand的下属函数
    side_bool = [] # 初始化布尔矩阵
    
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    for i in range(len(list1)): # 遍历其他目标, 判断相对位置
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[1] > list1[i][3]: # x2 > x'1 & x1 < x'2 & y1 > y'2
            side_bool.append(True) # True: 存在目标
        else:
            side_bool.append(False) # False: 不存在目标
    
    # 将目标与其他目标的布尔列表进行或操作,得到该目标对面是否存在目标
    top = False
    for i in range(len(side_bool)):
        top = top or side_bool[i]
    
    side_bool = [] # 废物利用该列表
    
    # 取得目标的上距
    if top == True:
        for i in range(len(list1)):
            if target[1] > list1[i][3] and target[2] > list1[i][0] and target[0] < list1[i][2]: # y1 > y'2 & x2 > x'1 & x1 < x'2
                side_bool.append(target[1] - list1[i][3]) # y1 - y'2, 可能存在多个目标在基准目标正上方
        return min(side_bool)
    else:
        return target[1]


def botside_expand(target, list1, height, width): # 该函数属于toplef_expand的下属函数
    side_bool = [] # 初始化布尔矩阵
    
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    for i in range(len(list1)): # 遍历其他目标, 判断相对位置
        if target[2] > list1[i][0] and target[0] < list1[i][2] and target[3] < list1[i][1]: # x2 > x'1 & x1 < x'2 & y2 < y'1
            side_bool.append(True) # True: 存在目标
        else:
            side_bool.append(False) # False: 不存在目标
    
    # 将目标与其他目标的布尔列表进行或操作,得到该目标对面是否存在目标
    bot = False
    for i in range(len(side_bool)):
        bot = bot or side_bool[i]
    
    side_bool = [] # 废物利用该列表
    
    # 取得目标的下距
    if bot == True:
        for i in range(len(list1)):
            if target[3] < list1[i][1] and target[2] > list1[i][0] and target[0] < list1[i][2]: # y2 < y'1 & x2 > x'1 & x1 < x'2
                side_bool.append(list1[i][1] - target[3]) # y'1 - y2
        return min(side_bool)
    else:
        return height - target[3]


def lefside_expand(target, list1, height, width): # 该函数属于oneside_expand的下属函数
    side_bool = [] # 初始化布尔矩阵
    
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    for i in range(len(list1)): # 遍历其他目标, 判断相对位置
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[0] > list1[i][2]: # y2 > y'1 & y1 < y'2 & x1 > x'2
            side_bool.append(True) # True: 存在目标
        else:
            side_bool.append(False) # False: 不存在目标
    
    # 将目标与其他目标的布尔列表进行或操作,得到该目标对面是否存在目标
    lef = False
    for i in range(len(side_bool)):
        lef = lef or side_bool[i]
    
    side_bool = [] # 废物利用该列表
    
    # 取得目标的左距
    if lef == True:
        for i in range(len(list1)):
            if target[0] > list1[i][2] and target[3] > list1[i][1] and target[1] < list1[i][3]: # x1 > x'2 & y2 > y'1 & y1 < y'2
                side_bool.append(target[0] - list1[i][2]) # x1 - x'2
        return min(side_bool)
    else:
        return target[0]


def rigside_expand(target, list1, height, width): # 该函数属于oneside_expand的下属函数
    side_bool = [] # 初始化布尔矩阵
    
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    for i in range(len(list1)): # 遍历其他目标, 判断相对位置
        if target[3] > list1[i][1] and target[1] < list1[i][3] and target[2] < list1[i][0]: # y2 > y'1 & y1 < y'2 & x2 < x'1
            side_bool.append(True) # True: 存在目标
        else:
            side_bool.append(False) # False: 不存在目标
    
    # 将目标与其他目标的布尔列表进行或操作,得到该目标对面是否存在目标
    rig = False
    for i in range(len(side_bool)):
        rig = rig or side_bool[i]
    
    side_bool = [] # 废物利用该列表
    
    # 取得目标的右距
    if rig == True:
        for i in range(len(list1)):
            if target[2] < list1[i][0] and target[3] > list1[i][1] and target[1] < list1[i][3]: # x2 < x'1 & y2 > y'1 & y1 < y'2
                side_bool.append(list1[i][0] - target[2]) # x'1 - x2
        return min(side_bool)
    else:
        return width - target[2]


def top_lef_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为toplef_expand函数服务
    # 遍历所有目标,判断选中目标上方最短距离
    for i in range(len(list1)):
        if target[1] > list1[i][3] and target[2] > list1[i][0] and target[0] < list1[i][2]: # y1 > y'2 & x2 > x'1 & x1 < x'2
            side_bool.append(target[1] - list1[i][3]) # y1 - y'2, 可能存在多个目标在基准目标正上方
    
    if len(side_bool) != 0:
        top_min_distence = min(side_bool)
    else:
        top_min_distence = target[1]
        
    # 遍历所有目标,判断目标是否在基准目标的左上
    for i in range(len(list1)):
        if target[1] > list1[i][3] > target[1] - top_min_distence and target[0] > list1[i][2]: # y1 > y'2 > y1 - top & x1 > x'2
            inner_bool.append(target[1] - list1[i][3])
        
    # 取得上左目标与本目标的距离
    if len(inner_bool) == 0:
        top_left_distance = target[0]
    else:
        top_left_distance = min(inner_bool)
        
    return top_min_distence, top_left_distance


def lef_top_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为toplef_expand函数服务
    # 遍历所有目标,判断是否在选中目标左侧,并计算其与选中目标距离
    for i in range(len(list1)):
        if target[0] > list1[i][2] and target[3] > list1[i][1] and target[1] < list1[i][3]: # x1 > x'2 & y2 > y'1 & y1 < y'2
            side_bool.append(target[0] - list1[i][2]) # x1 - x'2
        
    if len(side_bool) != 0:
        left_min_distance = min(side_bool)
    else:
        left_min_distance = target[0]
        
    # 遍历所有目标,判断目标是否在基准目标的左上
    for i in range(len(list1)):
        if target[0] > list1[i][2] > target[0] - left_min_distance and target[1] > list1[i][3]: # x1 > x'2 > x1 - left & y1 > y'2
            inner_bool.append(target[0] - list1[i][2])
        
    # 取得左上目标与本目标的距离
    if len(inner_bool) == 0:
        left_top_distance = target[1]
    else:
        left_top_distance = min(inner_bool)
        
    return left_min_distance, left_top_distance


def top_rig_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为toprig_expand函数服务
    # 遍历所有目标,判断选中目标上方最短距离
    for i in range(len(list1)):
        if target[1] > list1[i][3] and target[2] > list1[i][0] and target[0] < list1[i][2]: # y1 > y'2 & x2 > x'1 & x1 < x'2
            side_bool.append(target[1] - list1[i][3]) # y1 - y'2, 可能存在多个目标在基准目标正上方
        
    if len(side_bool) != 0:
        top_min_distence = min(side_bool)
    else:
        top_min_distence = target[1]
    
    # 遍历所有目标,判断目标是否在基准目标的右上
    for i in range(len(list1)):
        if target[1] > list1[i][3] > target[1] - top_min_distence and target[2] < list1[i][0]: # y1 > y'2 > y1 - top & x2 < x'1
                inner_bool.append(target[1] - list1[i][3])
                
    # 取得右上目标与本目标的距离
    if len(inner_bool) == 0:
        top_right_distance = width - target[2]
    else:
        top_right_distance = min(inner_bool)
    
    return top_min_distence, top_right_distance


def rig_top_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为toprig_expand函数服务
    # 遍历所有目标,判断是否在选中目标右侧,并计算其与选中目标距离
    for i in range(len(list1)):
        if target[2] < list1[i][0] and target[3] > list1[i][1] and target[1] < list1[i][3]: # x2 < x'1 & y2 > y'1 & y1 < y'2
                side_bool.append(list1[i][0] - target[2]) # x'1 - x2
    
    if len(side_bool) != 0:
        right_min_distance = min(side_bool) # 取得最小左距
    else:
        right_min_distance = width - target[2]
    
    # 遍历所有目标, 判断目标是否在基准目标的右上
    for i in range(len(list1)):
        if target[2] + right_min_distance > list1[i][0] > target[2] and target[1] > list1[i][3]: # x2 + right > x'1 > x2 & y1 > y'2
            inner_bool.append(list1[i][0] - target[2])
    
    # 取得右上目标与本目标的距离
    if len(inner_bool) == 0:
        right_top_distance = target[1]
    else:
        right_top_distance = min(inner_bool)
    
    return right_min_distance, right_top_distance


def bot_lef_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为botlef_expand函数服务
    # 遍历所有目标,判断选中目标下方最短距离
    for i in range(len(list1)):
        if target[3] < list1[i][1] and target[2] > list1[i][0] and target[0] < list1[i][2]: # y2 < y'1 & x2 > x'1 & x1 < x'2
                side_bool.append(list1[i][1] - target[3]) # y'1 - y2
    
    if len(side_bool) != 0:
        bottom_min_distance = min(side_bool) # 取得最小下距
    else:
        bottom_min_distance = height - target[3]
    
    # 遍历所有目标,判断目标是否在基准目标的左下
    for i in range(len(list1)):
        if target[3] + bottom_min_distance > list1[i][1] > target[3] and target[0] > list1[i][2]: # y2 + bot > y'1 > y2 & x1 > x'2
            inner_bool.append(list1[i][1] - target[3])
    
    # 取得左下目标与本目标的距离
    if len(inner_bool) == 0:
        bottom_left_distance = target[0]
    else:
        bottom_left_distance = min(inner_bool)
    
    return bottom_min_distance, bottom_left_distance


def lef_bot_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为botlef_expand函数服务
    # 遍历所有目标,判断是否在选中目标左侧,并计算其与选中目标距离
    for i in range(len(list1)):
        if target[0] > list1[i][2] and target[3] > list1[i][1] and target[1] < list1[i][3]: # x1 > x'2 & y2 > y'1 & y1 < y'2
            side_bool.append(target[0] - list1[i][2]) # x1 - x'2
    
    if len(side_bool) != 0:
        left_min_distance = min(side_bool) # 取得最小左距
    else:
        left_min_distance = target[0]
    
    # 遍历所有目标,判断目标是否在基准目标的左下
    for i in range(len(list1)):
        if target[0] > list1[i][2] > target[0] - left_min_distance and target[3] < list1[i][1]: # x1 > x'2 > x1 - left & y2 < y'1
            inner_bool.append(target[0] - list1[i][2])
    
    # 取得左下目标与本目标的距离
    if len(inner_bool) == 0:
        lef_bottom_distance = height - target[3]
    else:
        lef_bottom_distance = min(inner_bool)
    
    return left_min_distance, lef_bottom_distance


def bot_rig_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为botrig_expand函数服务
    # 遍历所有目标,判断选中目标下方最短距离
    for i in range(len(list1)):
        if target[3] < list1[i][1] and target[2] > list1[i][0] and target[0] < list1[i][2]: # y2 < y'1 & x2 > x'1 & x1 < x'2
                side_bool.append(list1[i][1] - target[3]) # y'1 - y2
    
    if len(side_bool) != 0:
        bottom_min_distance = min(side_bool) # 取得最小下距
    else:
        bottom_min_distance = height - target[3]
    
    # 遍历所有目标,判断目标是否在基准目标的右下
    for i in range(len(list1)):
        if target[3] + bottom_min_distance > list1[i][1] > target[3] and target[2] < list1[i][0]: # y2 + bot > y'1 > y2 & x2 < x'1
            inner_bool.append(list1[i][1] - target[3])
    
    # 取得右下目标与本目标的距离
    if len(inner_bool) == 0:
        bottom_right_distance = width - target[2]
    else:
        bottom_right_distance = min(inner_bool)
    
    return bottom_min_distance, bottom_right_distance


def rig_bot_has_target(target, list1, side_bool, inner_bool, height, width): # 该函数为botrig_expand函数服务
    # 遍历所有目标,判断是否在选中目标右侧,并计算其与选中目标距离
    for i in range(len(list1)):
        if target[2] < list1[i][0] and target[3] > list1[i][1] and target[1] < list1[i][3]: # x2 < x'1 & y2 > y'1 & y1 < y'2
            side_bool.append(list1[i][0] - target[2]) # x'1 - x2
    
    if len(side_bool) != 0:
        right_min_distance = min(side_bool) # 取得最小左距
    else:
        right_min_distance = width - target[2]
    
    # 遍历所有目标, 判断目标是否在基准目标的右下
    for i in range(len(list1)):
        if target[2] + right_min_distance > list1[i][0] > target[2] and target[3] < list1[i][1]: # x2 + right > x'1 > x2 & y2 < y'1
            inner_bool.append(list1[i][1] - target[3])
    
    # 取得右下目标与本目标的距离
    if len(inner_bool) == 0:
        right_bottom_distance = height - target[3]
    else:
        right_bottom_distance = min(inner_bool)
    
    return right_min_distance, right_bottom_distance


def overlap(tensor1, tensor2):
    area1 = (tensor1[2] - tensor1[0]) * (tensor1[3] - tensor1[1])
    area2 = (tensor2[2] - tensor2[0]) * (tensor2[3] - tensor2[1])
    temp_area = min(area1, area2)
    
    inter_toplef_x = max(tensor1[0], tensor2[0])
    inter_toplef_y = max(tensor1[1], tensor2[1])
    inter_botrig_x = min(tensor1[2], tensor2[2])
    inter_botrig_y = min(tensor1[3], tensor2[3])
    
    if inter_botrig_x - inter_toplef_x <=0 or inter_botrig_y - inter_toplef_y <= 0:
        return 0
    else:
        return (inter_botrig_x - inter_toplef_x) * (inter_botrig_y - inter_toplef_y) / temp_area


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


if __name__ == "__main__":
    # 标签名称列表, 并将列表编码
    VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
                   "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    
    # 测试图片与标签路径
    img_path = 'D:/AICV-DSTRethink/DataPool/000023.jpg'
    xml_path = 'D:/AICV-DSTRethink/DataPool/000023.xml'
    
    img = cv2.imread(img_path)
    res, height, width = xml_target(xml_path, class_to_ind)
    
    list_obj = [] # 所有目标的重叠情况列表
    global threshold
    threshold = 0.2
    
    # 分析每个目标与其他目标的模式, 完全无相交, 存在相交但不多, 严重相交
    for i in range(res.shape[0]):
        list1, list2, list3 = [], [], [] # 完全无相交, 小于0.1, 大于0.1
        for j in range(res.shape[0]):
            if res[i].tolist() == res[j].tolist():
                continue
            else:
                if overlap(res[i][:4], res[j][:4]) == 0: # torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) == 0:
                    list1.append(res[j])
                elif threshold >= overlap(res[i][:4], res[j][:4]) >= 0: # 0.1 >= torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0:
                    list2.append(res[j])
                elif overlap(res[i][:4], res[j][:4]) > threshold: # torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0.1:
                    list3.append(res[j])
        list_obj.append([list1, list2, list3])
    
    expandinfo = []
    # 分别对每个目标进行处理, 即处理其与其他目标的相交信息
    for i in range(len(list_obj)):
        # 当该目标与其他目标无重叠时
        if len(list_obj[i][0]) == len(res) - 1:
            expandinfo.append(check_direction1(res[i], list_obj[i], height, width))
        
        # 当该目标与其他目标重叠均不大于0.1时
        if len(list_obj[i][2]) == 0:
            if len(list_obj[i][1]) == 0: # 无相交不等于0但小于0.1的其他目标
                expandinfo.append(check_direction1(res[i], list_obj[i], height, width))
            else:                        # 有相交不等于0但小于0.1的其他目标
                expandinfo.append(situation_2(res[i], list_obj[i], height, width))
        
        # 当该目标存在重叠大于0.1的其他目标时
        if len(list_obj[i][2]) != 0:
            dense_obj, exinfo = situation_3(res[i], list_obj[i], res, height, width)
            expandinfo.append(exinfo)
            res[i][:4] = np.array(dense_obj)
    
    # 分析目标数据得到多种可能的裁剪方式, 每边适当向外扩展10个像素--此处获得扩展信息准备切割--
    cutinfo = []
    for i in range(len(expandinfo)):
        xcutmin = res[i][0] - expandinfo[i][2] # x1 - left
        ycutmin = res[i][1] - expandinfo[i][0] # y1 - top
        xcutmax = res[i][2] + expandinfo[i][3] # x2 + right
        ycutmax = res[i][3] + expandinfo[i][1] # y2 + bottom
        
        if xcutmin < 0:
            xcutmin = 0
        if ycutmin < 0:
            ycutmin = 0
        if xcutmax > width:
            xcutmax = width
        if ycutmax > height:
            ycutmax = height
                
        cutinfo.append([xcutmin, ycutmin, xcutmax, ycutmax])
    
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
    for i in range(len(cutinfo)):
        img_temp = copy.deepcopy(img)
        # label_temp = torch.cat((label_tensor, torch.from_numpy(np.array([cutinfo[i]]))), 0)
        label_temp = torch.from_numpy(np.array([cutinfo[i]]))
        for k in range(len(label_temp)):
            box = label_temp[k]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (_COLORS[3] * 255).astype(np.uint8).tolist()
            cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
        cv2.imwrite('D:/AICV-DSTRethink/DataPool/test/' + str(os.path.basename(img_path)) + '_' + str(i) + '_' + str(j) + '.jpg', img_temp)
    """
    # 写入csv文件,图片名 目标坐标(xyxycls) 裁剪坐标(xyxy) 目标面积 裁剪面积 图块/目标比例
    csv_path = 'D:/AICV-YoloXReDST-ADP/datapool/VOC_0712trainval_ObjectInfo.csv'
    for i in range(res.shape[0]):
        for j in range(len(cutinfo[i])):
            list_info = [os.path.basename(img_path), # , 
                         res[i][0], res[i][1], res[i][2], res[i][3], res[i][4], 
                         cutinfo[i][j][0], cutinfo[i][j][1], cutinfo[i][j][2], cutinfo[i][j][3], 
                         ratio[i][j][0], ratio[i][j][1], ratio[i][j][2]]
            with open(csv_path, 'a', newline="") as csvfile:
                writer0 = csv.writer(csvfile)
                writer0.writerow(list_info)
    """
