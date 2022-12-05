# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:18:34 2022

@author: Cheng Yuxuan Original
"""
import cv2
import copy
import torch
import numpy as np

xml_head = '''<annotation>
    <folder>VOC2022</folder>
    <filename>{}</filename>.
    <source>
        <database>The VOC2022 Datasets</database>
        <annotation>PASCAL VOC2022</annotation>
        <image>flickr</image>
        <flickrid>325991873</flickrid>
    </source>
    <owner>
        <flickrid>null</flickrid>
        <name>null</name>
    </owner>    
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''

xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Rear</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''

xml_end = '''
</annotation>'''


def splicing_str(string):
    # 列表化字符串, 方便后续通过枚举的方式处理字符串
    string = list(string)                                             
    
    if string[0] == "[" and string[1] == "[":                         # 当字符串有双层[时, 去除外层[
        string.pop(0)
    if string[len(string)-1] == "]" and string[len(string)-2] == "]": # 当字符串有双层]时, 去除外层]
        string.pop(len(string)-1)
    
    # coor_list外层列表, list_temp内层列表, num_temp数字首字符, count计数器
    coor_list, list_temp, num_temp, count = [], [], 0, 0              
    
    for i in range(len(string)):
        if string[i] == '[':                          # 当字符为[时, 初始化内层列表, 数字首字符, 计数器
            list_temp, num_temp, count = [], string[i+1], i+1
        elif string[i] != ' ' and string[i] != ']':   # 字符不为空格同时前字符不为]时, 拼接数字
            if count == i:
                pass
            else:
                num_temp = num_temp + string[i]
        elif string[i] == ' ' and string[i-1] != ']': # 字符为空格且前字符不为]时, 拼接组装好的数字成列表, 并重新初始化
            list_temp.append(float(num_temp))
            num_temp, count = string[i+1], i+1
        elif string[i] == ' ' and string[i-1] == ']': # 字符为空格且前字符为]时, 跳过本轮次, 因为是两个内层列表中的空格
            pass
        elif string[i] == ']':                        # 字符为]时, 拼接组装好的列表进外层列表
            list_temp.append(float(num_temp))
            coor_list.append(list_temp)
    
    return coor_list


def correct_distortion_patch1(width, height, img, lab_whole, lab_multi, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1") 
        pad_top, pad_lef = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, pad_top, 0, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2], lab_whole[1], lab_whole[3] = lab_whole[0] + pad_lef, lab_whole[2] + pad_lef, lab_whole[1] + pad_top, lab_whole[3] + pad_top
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] + pad_lef, lab_multi[i][2] + pad_lef
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] + pad_top, lab_multi[i][3] + pad_top
            lab_whole[0], lab_whole[2], lab_whole[1], lab_whole[3] = lab_whole[0] + pad_lef, lab_whole[2] + pad_lef, lab_whole[1] + pad_top, lab_whole[3] + pad_top
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab_whole[1], img.shape[0] - lab_whole[3], lab_whole[0], img.shape[1] - lab_whole[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab_whole[1], img.shape[0] - lab_whole[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi[1], lab_multi[3] = lab_whole[1], lab_whole[3]
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            
        pad_lef = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] + pad_lef, lab_whole[2] + pad_lef
            lab_multi[0], lab_multi[2] = lab_whole[0], lab_whole[2]
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_ymin, lab_multi[i][2] - new_img_ymin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_ymin, lab_whole[2] - new_img_ymin
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab_whole[0], img.shape[1] - lab_whole[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_multi[0], lab_multi[2] = lab_whole[0], lab_whole[2]
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            
        pad_top = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top
            lab_multi[1], lab_multi[3] = lab_whole[1], lab_whole[3]
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] + pad_top, lab_multi[i][3] + pad_top
            lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top
        
    # print("----------")
    return img, lab_whole, lab_multi


def correct_distortion_patch2(width, height, img, lab_whole, lab_multi, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1") 
        pad_top, pad_rig = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, pad_top, 0, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top
            lab_multi[1], lab_multi[3] = lab_whole[1], lab_whole[3]
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] + pad_top, lab_multi[i][3] + pad_top
            lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab_whole[1], img.shape[0] - lab_whole[3], lab_whole[0], img.shape[1] - lab_whole[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab_whole[1], img.shape[0] - lab_whole[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
        
        pad_rig = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab_whole[0], img.shape[1] - lab_whole[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
        
        pad_top = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] + pad_top, lab_multi[i][3] + pad_top
            lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top
        
    # print("----------")
    return img, lab_whole, lab_multi


def correct_distortion_patch3(width, height, img, lab_whole, lab_multi, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1")
        pad_bot, pad_lef = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, pad_bot, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] + pad_lef, lab_whole[2] + pad_lef
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] + pad_lef, lab_multi[i][2] + pad_lef
            lab_whole[0], lab_whole[2] = lab_whole[0] + pad_lef, lab_whole[2] + pad_lef
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab_whole[1], img.shape[0] - lab_whole[3], lab_whole[0], img.shape[1] - lab_whole[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab_whole[1], img.shape[0] - lab_whole[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
        
        pad_lef = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] + pad_lef, lab_whole[2] + pad_lef
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] + pad_lef, lab_multi[i][2] + pad_lef
            lab_whole[0], lab_whole[2] = lab_whole[0] + pad_lef, lab_whole[2] + pad_lef
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab_whole[0], img.shape[1] - lab_whole[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
        
        pad_bot = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, 0, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
    # print("----------")
    return img, lab_whole, lab_multi


def correct_distortion_patch4(width, height, img, lab_whole, lab_multi, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1")
        pad_bot, pad_rig = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, pad_bot, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab_whole[1], img.shape[0] - lab_whole[3], lab_whole[0], img.shape[1] - lab_whole[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab_whole[1], img.shape[0] - lab_whole[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] - new_img_ymin, lab_multi[i][3] - new_img_ymin
            lab_whole[1], lab_whole[3] = lab_whole[1] - new_img_ymin, lab_whole[3] - new_img_ymin
                
        pad_rig = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab_whole[0], img.shape[1] - lab_whole[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
            lab_multi = lab_whole
        else:
            for i in range(len(lab_multi)):
                lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] - new_img_xmin, lab_multi[i][2] - new_img_xmin
            lab_whole[0], lab_whole[2] = lab_whole[0] - new_img_xmin, lab_whole[2] - new_img_xmin
                
        pad_bot = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, 0, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
    # print("----------")
    return img, lab_whole, lab_multi


def analyze_pattern(lab_multi):
    # 确定该的确有多个目标
    assert len(np.array(copy.deepcopy(lab_multi)).shape) > 1, "there are only one object inside this obj group"
    
    # 得到该目标群包含的目标数量
    obj_num, max_lab, above_thr, below_thr = len(lab_multi), [], [], []
    
    # 筛选出最大面积目标
    max_area = (lab_multi[0][2] - lab_multi[0][0]) * (lab_multi[0][3] - lab_multi[0][1])
    for i in range(1, len(lab_multi)):
        if max_area < (lab_multi[i][2] - lab_multi[i][0]) * (lab_multi[i][3] - lab_multi[i][1]):
            max_area = (lab_multi[i][2] - lab_multi[i][0]) * (lab_multi[i][3] - lab_multi[i][1])
            max_lab = lab_multi[i]
    
    """
    # 将所有目标进行分级
    second_area, third_area, fourth_area = [], [], []
    for i in range(len(lab_multi)):
        if (lab_multi[i][2] - lab_multi[i][0]) * (lab_multi[i][3] - lab_multi[i][1]) == max_area:
            pass
        if max_area > (lab_multi[i][2] - lab_multi[i][0]) * (lab_multi[i][3] - lab_multi[i][1]) >= 0.5 * max_area:
            second_area.append(lab_multi[i])
        if 0.5 * max_area > (lab_multi[i][2] - lab_multi[i][0]) * (lab_multi[i][3] - lab_multi[i][1]) >= 0.25 * max_area:
            third_area.append(lab_multi[i])
        if 0.25 * max_area > (lab_multi[i][2] - lab_multi[i][0]) * (lab_multi[i][3] - lab_multi[i][1]):
            fourth_area.append(lab_multi[i])
    
    # 判断目标群模式, 方便后续处理
    if len(second_area) > len(third_area) and len(second_area) > len(fourth_area): # 大目标占据主导
        
    if len(third_area) > len(second_area) and len(third_area) > len(fourth_area): # 中目标占据主导
        
    if len(fourth_area) > len(second_area) and len(fourth_area) > len(third_area): # 小目标占据主导
    """
    
    # 先采用最简方法节约时间, 日后再慢慢思考复杂分析方式------------------------------
    
    # 抛弃最大目标, 因为最大目标经常远大于其他目标
    lab_multi_other = copy.deepcopy(lab_multi)
    if len(max_lab) != 0:
        lab_multi_other.remove(max_lab)

    # 抛弃最大目标后将其他目标根据阈值进行分类
    for i in range(len(lab_multi_other)):
        if (lab_multi_other[i][2] - lab_multi_other[i][0]) * (lab_multi_other[i][3] - lab_multi_other[i][1]) > 32 * 32:
            above_thr.append(lab_multi_other[i])
        else:
            below_thr.append(lab_multi_other[i])
    
    # 分类后比较两者数量, 以占主导地位的目标的均值缩放至阈值处, 过小目标则抛弃
    if len(above_thr) > 0.5 * (obj_num - 1):
        return above_thr
    elif len(below_thr) > 0.5 * (obj_num - 1):
        return below_thr


def shrink_object(img, lab_whole, lab_multi, domain_obj): # img.shape=[h, w, c]
    # 当为单个目标图块时
    if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
        # 取得目标宽高
        w, h = lab_whole[2] - lab_whole[0], lab_whole[3] - lab_whole[1]
        
        # new / ori 取得缩放系数
        ratio_shrink = pow(32 * 32 * w / h, 0.5) / w
        
        # 缩放图片与目标
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img_temp = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                                   size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                                   mode='nearest').squeeze(0).numpy()
        img_temp = np.transpose(img_temp, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        lab_whole[0], lab_whole[1] = lab_whole[0] * ratio_shrink, lab_whole[1] * ratio_shrink
        lab_whole[2], lab_whole[3] = lab_whole[2] * ratio_shrink, lab_whole[3] * ratio_shrink
        lab_multi = lab_whole # 更新lab_multi
    
        return img_temp, lab_whole, lab_multi
    else:
        # 计算主导目标列表的目标面积均值
        mean_area = 0
        for i in range(len(domain_obj)):
            mean_area = mean_area + (domain_obj[i][2] - domain_obj[i][0]) * (domain_obj[i][3] - domain_obj[i][1])
        mean_area = mean_area / len(domain_obj)
        
        # 计算缩放系数
        ratio_shrink = 32 * 32 / mean_area
        
        # 缩放图片
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img_temp = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                                   size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                                   mode='nearest').squeeze(0).numpy()
        img_temp = np.transpose(img_temp, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
        # 缩放整体目标与目标群
        for i in range(len(lab_multi)):
            lab_multi[i][0], lab_multi[i][1] = lab_multi[i][0] * ratio_shrink, lab_multi[i][1] * ratio_shrink
            lab_multi[i][2], lab_multi[i][3] = lab_multi[i][2] * ratio_shrink, lab_multi[i][3] * ratio_shrink
        lab_whole[0], lab_whole[1] = lab_whole[0] * ratio_shrink, lab_whole[1] * ratio_shrink
        lab_whole[2], lab_whole[3] = lab_whole[2] * ratio_shrink, lab_whole[3] * ratio_shrink
        
        return img_temp, lab_whole, lab_multi


def check_objectsize(width, height, img, lab_whole, lab_multi): # [h, w, c]
    if (lab_whole[3] - lab_whole[1]) > height and (lab_whole[2] - lab_whole[0]) > width: # h > height, w > width
        x, y = lab_whole[2] - lab_whole[0], lab_whole[3] - lab_whole[1]
        ratio_shrink = min(width / x, height / y)
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        
        # 排除单个目标图片的图块后, 将多目标列表中每套坐标都进行缩放
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            pass
        else:
            for i in range(len(lab_multi)): # 内层列表
                lab_multi[i][0], lab_multi[i][1] = lab_multi[i][0] * ratio_shrink, lab_multi[i][1] * ratio_shrink
                lab_multi[i][2], lab_multi[i][3] = lab_multi[i][2] * ratio_shrink, lab_multi[i][3] * ratio_shrink
        
        lab_whole[0], lab_whole[1], lab_whole[2], lab_whole[3] = lab_whole[0] * ratio_shrink, lab_whole[1] * ratio_shrink, lab_whole[2] * ratio_shrink, lab_whole[3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab_whole[3] - lab_whole[1]) <= height and (lab_whole[2] - lab_whole[0]) > width: # h <= height, w > width
        x = lab_whole[2] - lab_whole[0]
        ratio_shrink = width / x
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        
        # 排除单个目标图片的图块后, 将多目标列表中每套坐标都进行缩放
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            pass
        else:
            for i in range(len(lab_multi)): # 内层列表
                lab_multi[i][0], lab_multi[i][1] = lab_multi[i][0] * ratio_shrink, lab_multi[i][1] * ratio_shrink
                lab_multi[i][2], lab_multi[i][3] = lab_multi[i][2] * ratio_shrink, lab_multi[i][3] * ratio_shrink
        
        lab_whole[0], lab_whole[1], lab_whole[2], lab_whole[3] = lab_whole[0] * ratio_shrink, lab_whole[1] * ratio_shrink, lab_whole[2] * ratio_shrink, lab_whole[3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab_whole[3] - lab_whole[1]) > height and (lab_whole[2] - lab_whole[0]) <= width: # h > height, w <= width
        y = lab_whole[3] - lab_whole[1]
        ratio_shrink = height / y
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        
        # 排除单个目标图片的图块后, 将多目标列表中每套坐标都进行缩放
        if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
            pass
        else:
            for i in range(len(lab_multi)): # 内层列表
                lab_multi[i][0], lab_multi[i][1] = lab_multi[i][0] * ratio_shrink, lab_multi[i][1] * ratio_shrink
                lab_multi[i][2], lab_multi[i][3] = lab_multi[i][2] * ratio_shrink, lab_multi[i][3] * ratio_shrink
        
        lab_whole[0], lab_whole[1], lab_whole[2], lab_whole[3] = lab_whole[0] * ratio_shrink, lab_whole[1] * ratio_shrink, lab_whole[2] * ratio_shrink, lab_whole[3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab_whole[3] - lab_whole[1]) <= height and (lab_whole[2] - lab_whole[0]) <= width: # h <= height, w<= width
        pass

    return img, lab_whole, lab_multi


def check_objectorient(width, height, img, lab_whole, lab_multi):
    if int(width) > int(height): # require w > h
        # if top-left patch w < h then rotate 90°
        if img.shape[1] < img.shape[0]: # img.shape == [h, w]
            # 首先将标签翻转90°
            xmin_new = lab_whole[1]                # xmin_new = ymin
            ymin_new = img.shape[1] - lab_whole[2] # ymin_new = width - xmax
            xmax_new = lab_whole[3]                # xmax_new = ymin + target_height
            ymax_new = img.shape[1] - lab_whole[0] # ymax_new = width - xmax + target_width
            lab_whole[0] = xmin_new
            lab_whole[1] = ymin_new
            lab_whole[2] = xmax_new
            lab_whole[3] = ymax_new
            
            # 当列表shape大于1时, 注意只有1个元素的二维列表shape也大于1, 将原本为float坐标的变为int
            if len(np.array(lab_multi).shape) > 1:
                for i in range(len(lab_multi)): # 内层列表
                    for j in range(len(lab_multi[i])): # 内层坐标
                        lab_multi[i][j] = int(lab_multi[i][j])
            
            # 排除了单个目标图片的图块后, 将多目标图块中的多个目标均旋转90°
            if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
                pass
            else:
                for i in range(len(lab_multi)): # 内层列表
                    xmin_new = lab_multi[i][1]             # xmin_new = ymin
                    ymin_new = img.shape[1] - lab_multi[i][2] # ymin_new = width - xmax
                    xmax_new = lab_multi[i][3]                # xmax_new = ymin + target_height
                    ymax_new = img.shape[1] - lab_multi[i][0] # ymax_new = width - xmax + target_width
                    lab_multi[i][0] = xmin_new
                    lab_multi[i][1] = ymin_new
                    lab_multi[i][2] = xmax_new
                    lab_multi[i][3] = ymax_new
                
            # 再把图片翻转90°
            img = np.rot90(img)
    else: # require w < h
        # if top-left patch w > h then rotate 90°
        if img.shape[1] > img.shape[0]:
            # 首先将标签翻转90°
            xmin_new = lab_whole[1]                # xmin_new = ymin
            ymin_new = img.shape[1] - lab_whole[2] # ymin_new = width - xmax
            xmax_new = lab_whole[3]                # xmax_new = ymin + target_height
            ymax_new = img.shape[1] - lab_whole[0] # ymax_new = width - xmax + target_width
            lab_whole[0] = xmin_new
            lab_whole[1] = ymin_new
            lab_whole[2] = xmax_new
            lab_whole[3] = ymax_new
            
            # 当列表shape大于1时, 注意只有1个元素的二维列表shape也大于1, 将原本为float坐标的变为int
            if len(np.array(lab_multi).shape) > 1:
                for i in range(len(lab_multi)): # 内层列表
                    for j in range(len(lab_multi[i])): # 内层坐标
                        lab_multi[i][j] = int(lab_multi[i][j])
            
            # 排除了单个目标图片的图块后, 将多目标图块中的多个目标均旋转90°
            if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
                pass
            else:
                for i in range(len(lab_multi)): # 内层列表
                    xmin_new = lab_multi[i][1]                # xmin_new = ymin
                    ymin_new = img.shape[1] - lab_multi[i][2] # ymin_new = width - xmax
                    xmax_new = lab_multi[i][3]                # xmax_new = ymin + target_height
                    ymax_new = img.shape[1] - lab_multi[i][0] # ymax_new = width - xmax + target_width
                    lab_multi[i][0] = xmin_new
                    lab_multi[i][1] = ymin_new
                    lab_multi[i][2] = xmax_new
                    lab_multi[i][3] = ymax_new
            
            # 再把图片翻转90°
            img = np.rot90(img)
    
    return img, lab_whole, lab_multi


def xml_label_generate(height, width, label_num, xml_path):
    label_str = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                 "cat", "chair", "cow", "diningtable", "dog", "horse",  "motorbike", 
                 "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    
    # 创建xml的head与目标格
    head, obj = xml_head.format(str(os.path.basename(xml_path[:-4])), str(width), str(height), str(3)), ''
    
    for i in range(len(label_num)):
        obj += xml_obj.format(label_str[int(label_num[i][4])], 
                              int(label_num[i][0]), 
                              int(label_num[i][1]), 
                              int(label_num[i][2]), 
                              int(label_num[i][3]))

    # 打开xml文件,写入各种信息
    with open(xml_path, 'w') as f_xml:
        f_xml.write(head + obj + xml_end)


def multiply_ratio_to_list(lab_whole, lab_multi, ratio):
    if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
        lab_whole[0] = lab_whole[0] * ratio[0] # xmin * ratio_width
        lab_whole[2] = lab_whole[2] * ratio[0] # xmax * ratio_width
        lab_whole[1] = lab_whole[1] * ratio[1] # ymin * ratio_height
        lab_whole[3] = lab_whole[3] * ratio[1] # ymax * ratio_height
        lab_multi = lab_whole
    else:
        for i in range(len(lab_multi)):
            lab_multi[i][0] = lab_multi[i][0] * ratio[0] # xmin * ratio_width
            lab_multi[i][2] = lab_multi[i][2] * ratio[0] # xmax * ratio_width
            lab_multi[i][1] = lab_multi[i][1] * ratio[1] # ymin * ratio_height
            lab_multi[i][3] = lab_multi[i][3] * ratio[1] # ymax * ratio_height
        lab_whole[0] = lab_whole[0] * ratio[0] # xmin * ratio_width
        lab_whole[2] = lab_whole[2] * ratio[0] # xmax * ratio_width
        lab_whole[1] = lab_whole[1] * ratio[1] # ymin * ratio_height
        lab_whole[3] = lab_whole[3] * ratio[1] # ymax * ratio_height
    
    return lab_whole, lab_multi

def add_offset_to_list(lab_whole, lab_multi, offset):
    if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
        lab_whole[0] = lab_whole[0] + offset[0]
        lab_whole[1] = lab_whole[1] + offset[1]
        lab_whole[2] = lab_whole[2] + offset[2]
        lab_whole[3] = lab_whole[3] + offset[3]
        lab_multi = lab_whole
    else:
        for i in range(len(lab_multi)):
            lab_multi[i][0] = lab_multi[i][0] + offset[0]
            lab_multi[i][1] = lab_multi[i][1] + offset[1]
            lab_multi[i][2] = lab_multi[i][2] + offset[2]
            lab_multi[i][3] = lab_multi[i][3] + offset[3]
        lab_whole[0] = lab_whole[0] + offset[0]
        lab_whole[1] = lab_whole[1] + offset[1]
        lab_whole[2] = lab_whole[2] + offset[2]
        lab_whole[3] = lab_whole[3] + offset[3]
    
    return lab_whole, lab_multi


def add_pad_to_list(lab_whole, lab_multi, pad_top, pad_left):
    if lab_whole[0] == lab_multi[0] and lab_whole[1] == lab_multi[1] and lab_whole[2] == lab_multi[2] and lab_whole[3] == lab_multi[3]:
        lab_whole[0], lab_whole[2] = lab_whole[0] + pad_left, lab_whole[2] + pad_left # xmin + pad_left, xmax + pad_left
        lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top # ymin + pad_top, ymax + pad_top
        lab_multi = lab_whole
    else:
        for i in range(len(lab_multi)):
            lab_multi[i][0], lab_multi[i][2] = lab_multi[i][0] + pad_left, lab_multi[i][2] + pad_left # xmin + pad_left, xmax + pad_left
            lab_multi[i][1], lab_multi[i][3] = lab_multi[i][1] + pad_top, lab_multi[i][3] + pad_top # ymin + pad_top, ymax + pad_top
        lab_whole[0], lab_whole[2] = lab_whole[0] + pad_left, lab_whole[2] + pad_left # xmin + pad_left, xmax + pad_left
        lab_whole[1], lab_whole[3] = lab_whole[1] + pad_top, lab_whole[3] + pad_top # ymin + pad_top, ymax + pad_top
        
    return lab_whole, lab_multi