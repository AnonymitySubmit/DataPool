# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:23:10 2022

@author: Cheng Yuxuan Original
"""
import os
import cv2
# import csv
import copy
import torch
import random
import numpy as np
import pandas as pd


img_file = 'F:/VOCtrainval_images/'
# save_dir = 'D:/AICV-YoloXReDST-ADP/datasets_result/'


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


def four_to_one_less_config1_analysis(scope, height, width, ratio_height, ratio_width_top, ratio_width_bot): # scope: 目标大小, 默认[0, 1024], 即MSCOCO标准
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    # ratio_area = random.uniform(1/2, 1) # 生成[1/2, 1)之间的随机数,用于取得合成图与原图的比例
    # ratio_width_top = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    # ratio_width_bot = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左下右下两图块之间的比例
    # ratio_height = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下四图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    # search_patch_1_low = height * width * ratio_area * 0.8 * ratio_height * ratio_width_top
    # search_patch_2_low = height * width * ratio_area * 0.8 * ratio_height * (1-ratio_width_top)
    # search_patch_3_low = height * width * ratio_area * 0.8 * (1-ratio_height) * ratio_width_bot
    # search_patch_4_low = height * width * ratio_area * 0.8 * (1-ratio_height) * (1-ratio_width_bot)
    
    # 将搜索面积向下浮动0.2以增加能搜索到的图块的数量
    search_patch_1_low = height * width * 0.8 * ratio_height * ratio_width_top
    search_patch_2_low = height * width * 0.8 * ratio_height * (1-ratio_width_top)
    search_patch_3_low = height * width * 0.8 * (1-ratio_height) * ratio_width_bot
    search_patch_4_low = height * width * 0.8 * (1-ratio_height) * (1-ratio_width_bot)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite'
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite'
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite'
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = 'infinite'
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = search_patch_4_low / scope[0] # 拼图面积 / 目标面积最小值
    
    return ([search_patch_1_low, proportion_1_low, proportion_1_high],
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high],
           [search_patch_4_low, proportion_4_low, proportion_4_high])


class four_to_one_less_config1_search:
    def __init__(self, list1, list2, list3, list4, csv_path, ratio_width_top, ratio_width_bot, ratio_height, height, width): # ratio_area, 
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.list4 = list4
        
        self.csv_path = csv_path
        
        # self.ratio_area = ratio_area
        self.ratio_width_top = ratio_width_top
        self.ratio_width_bot = ratio_width_bot
        self.ratio_height = ratio_height
        self.height = height
        self.width = width
        
    def do_search(self, save_dir):
        # 检查文件夹是否存在,不存在则创建文件夹
        save_dir_temp = save_dir[:-12]
        if os.path.exists(save_dir_temp):
            pass
        else:
            os.mkdir(save_dir_temp)
        
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
        search_area_4_low, proportion_4_low, proportion_4_high = self.list4
        
        self.potential_area_1, self.potential_area_2, self.potential_area_3, self.potential_area_4 = [], [], [], []
        
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
            
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
        
            # 判断该目标是否满足search_area_4的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
            else:
                if proportion_4_high > list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
    
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        assert len(self.potential_area_4) != 0, 'No patch match the search condition 4'
        
        print("lef top patch num:", len(self.potential_area_1))
        print("rig top patch num:", len(self.potential_area_2))
        print("lef bot patch num:", len(self.potential_area_3))
        print("rig bot patch num:", len(self.potential_area_4))
        
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = [], [], [], []
        
        # 初始化合成计数器和递归停止符
        count, self.isStop = 0, False
        
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4) != 0:
            self.i, self.j = random.choice(self.potential_area_1), random.choice(self.potential_area_2)
            self.k, self.v = random.choice(self.potential_area_3), random.choice(self.potential_area_4)
            while(len(self.potential_area_1) != 0 or len(self.potential_area_2) != 0 or len(self.potential_area_3) != 0 or len(self.potential_area_4) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k, self.v = self.none_same_patch()
                
                # 读取并切割图片,修改标签
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img4 = cv2.imread(img_file + self.v[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                img4 = img4[self.v[7]:self.v[9], self.v[6]:self.v[8]]
                lab4 = [self.v[1]-self.v[6], self.v[2]-self.v[7], self.v[3]-self.v[6], self.v[4]-self.v[7], self.v[5]]
                
                # 调用合成函数合成图片
                four_to_one_less_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, 
                                                    self.ratio_width_top, self.ratio_width_bot, self.ratio_height, # self.ratio_area, 
                                                    count, self.height, self.width, save_dir)
                
                count = count + 1
                
                # 储存旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
                self.patch4_used, self.potential_area_4, self.v = self.store_remove_pick(self.patch4_used, self.potential_area_4, self.v)
    
        print("Synthesis Complete")
    
    def store_remove_pick(self, store_list, pick_list, patch_info):
        # 储存已用图块,但不储存复用图块
        if patch_info in store_list:
            pass
        else:
            store_list.append(patch_info)
        
        # 确认待抽列表是否为0
        if len(pick_list) == 0:
            pass
        else:
            if patch_info in pick_list:
                pick_list.remove(patch_info)
            else:
                pass
        
        # 待抽列表可能移除后为0,因此分为两次判断
        if len(pick_list) == 0:
            patch_info = random.choice(store_list)
        else:
            prob = random.choice(range(100))
            if prob < 30: # 30%概率取已用图块
                patch_info = random.choice(store_list)
            else: # 70%概率取未用图块
                patch_info = random.choice(pick_list)
        
        return store_list, pick_list, patch_info
    
    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.i, self.v) and \
               self.compare_info(self.j, self.k) and self.compare_info(self.k, self.v)) == False):
            if len(self.potential_area_1) == 0:
                list1 = self.patch1_used
            else:
                list1 = self.potential_area_1
            if len(self.potential_area_2) == 0:
                list2 = self.patch2_used
            else:
                list2 = self.potential_area_2
            if len(self.potential_area_3) == 0:
                list3 = self.patch3_used
            else:
                list3 = self.potential_area_3
            if len(self.potential_area_4) == 0:
                list4 = self.patch4_used
            else:
                list4 = self.potential_area_4
            
            self.i, self.j, self.k, self.v = random.choice(list1), random.choice(list2), random.choice(list3), random.choice(list4)
        
        return self.i, self.j, self.k, self.v # 成功跳出while循环即说明互不相同


def correct_distortion(width, height, img, lab, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1")
        top, bot, lef, rig = lab[1], img.shape[0] - lab[3], lab[0], img.shape[1] - lab[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_pad_w, need_pad_h = height - img.shape[0], width - img.shape[1]
        if lef == 0: 
            pad_lef = 0
        else: 
            pad_lef = int(need_pad_w * (lef/(lef+rig)))
        if rig == 0:
            pad_rig = 0
        else:
            pad_rig = int(need_pad_w * (rig/(lef+rig)))
        if top == 0:
            pad_top = 0
        else:
            pad_top = int(need_pad_h * (top/(top+bot)))
        if bot == 0:
            pad_bot = 0
        else:
            pad_bot = int(need_pad_h * (bot/(top+bot)))
        img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_lef, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[0], lab[2], lab[1], lab[3] = lab[0] + pad_lef, lab[2] + pad_lef, lab[1] + pad_top, lab[3] + pad_top
    
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab[1], img.shape[0] - lab[3], lab[0], img.shape[1] - lab[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
    
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab[1], img.shape[0] - lab[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
            
        lef, rig = lab[0], img.shape[1] - lab[2]
        if rig < 0: rig = 0
        need_pad_w = width - img.shape[1]
        pad_lef, pad_rig = int(need_pad_w * (lef/(lef+rig))), int(need_pad_w * (rig/(lef+rig)))
        img = cv2.copyMakeBorder(img, 0, 0, pad_lef, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[0], lab[2] = lab[0] + pad_lef, lab[2] + pad_lef
        
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab[0], img.shape[1] - lab[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        
        top, bot = lab[1], img.shape[0] - lab[3]
        if bot < 0: bot = 0
        need_pad_h = height - img.shape[0]
        pad_top, pad_bot = int(need_pad_h * (top/(top+bot))), int(need_pad_h * (bot/(top+bot)))
        img = cv2.copyMakeBorder(img, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[1], lab[3] = lab[1] + pad_top, lab[3] + pad_top
        
    # print("----------")
    return img, lab


def shrink_object(img, lab): # img.shape=[h, w, c]
    w, h = lab[2] - lab[0], lab[3] - lab[1]
    x = pow(32 * 32 * w / h, 0.5)
    ratio_shrink = x / w # new / ori
    img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
    img_temp = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                          size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                          mode='nearest').squeeze(0).numpy()
    lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
    img_temp = np.transpose(img_temp, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
    
    return img_temp, lab


def check_objectsize(width, height, img, lab): # [h, w, c]
    if (lab[3] - lab[1]) > height and (lab[2] - lab[0]) > width: # h > height, w > width
        x, y = lab[2] - lab[0], lab[3] - lab[1]
        ratio_shrink = min(width / x, height / y)
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab[3] - lab[1]) <= height and (lab[2] - lab[0]) > width: # h <= height, w > width
        x = lab[2] - lab[0]
        ratio_shrink = width / x
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab[3] - lab[1]) > height and (lab[2] - lab[0]) <= width: # h > height, w <= width
        y = lab[3] - lab[1]
        ratio_shrink = height / y
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab[3] - lab[1]) <= height and (lab[2] - lab[0]) <= width: # h <= height, w<= width
        pass

    return img, lab


def check_objectorient(width, height, img, lab):
    if int(width) > int(height): # require w > h
        # if top-left patch w < h then rotate 90°
        if img.shape[1] < img.shape[0]:
            # 首先将标签翻转90°
            xmin_new = lab[1] # xmin_new = ymin
            ymin_new = img.shape[1] - lab[2] # ymin_new = width - xmax
            xmax_new = lab[3] # xmax_new = ymin + target_height
            ymax_new = img.shape[1] - lab[0] # ymax_new = width - xmax + target_width
            lab[0] = xmin_new
            lab[1] = ymin_new
            lab[2] = xmax_new
            lab[3] = ymax_new
            # 再把图片翻转90°
            img = np.rot90(img)
    else: # require w < h
        # if top-left patch w > h then rotate 90°
        if img.shape[1] > img.shape[0]:
            # 首先将标签翻转90°
            xmin_new = lab[1] # xmin_new = ymin
            ymin_new = img.shape[1] - lab[2] # ymin_new = width - xmax
            xmax_new = lab[3] # xmax_new = ymin + target_height
            ymax_new = img.shape[1] - lab[0] # ymax_new = width - xmax + target_width
            lab[0] = xmin_new
            lab[1] = ymin_new
            lab[2] = xmax_new
            lab[3] = ymax_new
            # 再把图片翻转90°
            img = np.rot90(img)
    
    return img, lab


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


def four_to_one_less_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, 
                                        ratio_width_top, ratio_width_bot, ratio_height, 
                                        count, height, width, save_dir):
    # 取得合成图新宽高与缩放模式
    new_h, new_w, mode = height, width, 'nearest' #  * pow(ratio_area, 0.5)
    
    # 依据左上角需要的宽高比例来确定是否旋转左上图块 ---------------------------------
    img1, lab1 = check_objectorient(new_w * ratio_width_top, new_h * ratio_height, img1, lab1)
    
    # 检查目标宽高是否大于空位宽高,并缩放大于空位宽高的目标
    img1, lab1 = check_objectsize(new_w * ratio_width_top, new_h * ratio_height, img1, lab1)
    
    # 切割前确定获取的图块的目标小于32×32,并且目标宽高小于空位宽高
    if (lab1[3] - lab1[1]) * (lab1[2] - lab1[0]) <= 32 * 32:
        pass
    else:
        img1, lab1 = shrink_object(img1, lab1)
    
    # 进行切割然后填充以避免缩放导致的畸变
    if (new_w * ratio_width_top / img1.shape[1]) == 1 and (new_h * ratio_height / img1.shape[0]) == 1:
        pass
    else:
        img1, lab1 = correct_distortion(new_w * ratio_width_top, new_h * ratio_height, img1, lab1, count) # ) # 
    
    # 依据右上角需要的宽高比例来确定是否旋转右上图块 ---------------------------------
    img2, lab2 = check_objectorient(new_w * (1-ratio_width_top), new_h * ratio_height, img2, lab2)
    
    # 检查目标宽高是否大于空位宽高,并缩放大于空位宽高的目标
    img2, lab2 = check_objectsize(new_w * (1-ratio_width_top), new_h * ratio_height, img2, lab2)
    
    # 切割前确定获取的图块的目标小于32×32
    if (lab2[3] - lab2[1]) * (lab2[2] - lab2[0]) <= 32 * 32:
        pass
    else:
        img2, lab2 = shrink_object(img2, lab2)
    
    # 进行切割然后填充以避免后续处理时的畸变
    if (new_w * (1-ratio_width_top) / img2.shape[1]) == 1 and (new_h * ratio_height / img2.shape[0]) == 1:
        pass
    else:
        img2, lab2 = correct_distortion(new_w * (1-ratio_width_top), new_h * ratio_height, img2, lab2, count) # ) # 
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), 
                                                   size=(int(new_h * ratio_height), int(new_w * ratio_width_top)), 
                                                   mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), 
                                                   size=(int(new_h * ratio_height), int(new_w * (1-ratio_width_top))), 
                                                   mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width_top), int(new_h * ratio_height)), (img1.shape[2], img1.shape[1])))
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width_top)), int(new_h * ratio_height)), (img2.shape[2], img2.shape[1])))
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img1 = torch.zeros((3, divide_img_1.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img1[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img1[:3, :divide_img_1.shape[1], divide_img_1.shape[2]:].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width_top), 0.0, int(new_w * ratio_width_top), 0.0]] # 上左 上右
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 依据左下角需要的宽高比例来确定是否旋转左下图块 ---------------------------------
    img3, lab3 = check_objectorient(new_w * ratio_width_bot, new_h * (1-ratio_height), img3, lab3)
    
    # 检查目标宽高是否大于空位宽高,并缩放大于空位宽高的目标
    img3, lab3 = check_objectsize(new_w * ratio_width_bot, new_h * (1-ratio_height), img3, lab3)
    
    # 切割前确定获取的图块的目标小于32×32
    if (lab3[3] - lab3[1]) * (lab3[2] - lab3[0]) <= 32 * 32:
        pass
    else:
        img3, lab3 = shrink_object(img3, lab3)
    
    # 进行切割然后填充以避免后续处理时的畸变
    if (new_w * ratio_width_bot / img3.shape[1]) == 1 and (new_h * (1-ratio_height) / img3.shape[0]) == 1:
        pass
    else:
        img3, lab3 = correct_distortion(new_w * ratio_width_bot, new_h * (1-ratio_height), img3, lab3, count) # ) # 
    
    # 依据右下角需要的宽高比例来确定是否旋转右下图块 ---------------------------------
    img4, lab4 = check_objectorient(new_w * (1-ratio_width_bot), new_h * (1-ratio_height), img4, lab4)
    
    # 检查目标宽高是否大于空位宽高,并缩放大于空位宽高的目标
    img4, lab4 = check_objectsize(new_w * (1-ratio_width_bot), new_h * (1-ratio_height), img4, lab4)
    
    # 切割前确定获取的图块的目标小于32×32
    if (lab4[3] - lab4[1]) * (lab4[2] - lab4[0]) <= 32 * 32:
        pass
    else:
        img4, lab4 = shrink_object(img4, lab4)
    
    # 进行切割然后填充以避免后续处理时的畸变
    if (new_w * (1-ratio_width_bot) / img4.shape[1]) == 1 and (new_h * (1-ratio_height) / img4.shape[0]) == 1:
        pass
    else:
        img4, lab4 = correct_distortion(new_w * (1-ratio_width_bot), new_h * (1-ratio_height), img4, lab4, count) # ) # 
    
    # cv2.imwrite('D:/AICV-YoloXReDST-SGD/datasets_result/img4' + str(count) + '.jpg', img4)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    img4 = np.transpose(img4, (2, 0, 1))
    
    # 缩放img3与img4至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3.copy()).unsqueeze(0), 
                                                   size=(int(new_h * (1-ratio_height)), int(new_w * ratio_width_bot)), 
                                                   mode=mode).squeeze(0)
    divide_img_4 = torch.nn.functional.interpolate(torch.from_numpy(img4.copy()).unsqueeze(0), 
                                                   size=(int(new_h * (1-ratio_height)), int(new_w * (1-ratio_width_bot))), 
                                                   mode=mode).squeeze(0)
    
    # 缩放img3与img4的标签
    ratios_3 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width_bot), int(new_h * (1-ratio_height))), (img3.shape[2], img3.shape[1]))) # width, height
    ratios_4 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width_bot)), int(new_h * (1-ratio_height))), (img4.shape[2], img4.shape[1]))) # width, height
    lab3[0] = lab3[0] * ratios_3[0] # xmin * ratio_width
    lab3[2] = lab3[2] * ratios_3[0] # xmax * ratio_width
    lab3[1] = lab3[1] * ratios_3[1] # ymin * ratio_height
    lab3[3] = lab3[3] * ratios_3[1] # ymax * ratio_height
    lab4[0] = lab4[0] * ratios_4[0] # xmin * ratio_width
    lab4[2] = lab4[2] * ratios_4[0] # xmax * ratio_width
    lab4[1] = lab4[1] * ratios_4[1] # ymin * ratio_height
    lab4[3] = lab4[3] * ratios_4[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img2 = torch.zeros((3, divide_img_3.shape[1], divide_img_3.shape[2]+divide_img_4.shape[2]))
    syn_img2[:3, :divide_img_3.shape[1], :divide_img_3.shape[2]].copy_(divide_img_3)
    syn_img2[:3, :divide_img_3.shape[1], divide_img_3.shape[2]:].copy_(divide_img_4)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width_bot), 0.0, int(new_w * ratio_width_bot), 0.0]] # 下左 下右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    temp_img = copy.deepcopy(syn_img2)
    temp_img = np.ascontiguousarray(np.transpose(temp_img.numpy(), (1, 2, 0)))
    
    """
    # 可视化拼贴图片的标签是否与目标匹配,结果正确
    label_tensor = torch.cat((torch.from_numpy(np.array([lab3])), torch.from_numpy(np.array([lab4]))), 0)
    _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
    for i in range(len(label_tensor)):
        box = label_tensor[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[0] * 255).astype(np.uint8).tolist()
        cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
    cv2.imwrite('D:/AICV-YoloXReDST-SGD/datasets_test/syn_img_' + str(count) + '.jpg', temp_img) # cv2.imwrite reqire [h, w, c]
    """
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img3 = torch.zeros((3, syn_img1.shape[1]+syn_img2.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img3[:3, :syn_img1.shape[1], :syn_img1.shape[2]].copy_(syn_img1)
    syn_img3[:3, syn_img1.shape[1]:, :syn_img2.shape[2]].copy_(syn_img2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)], [0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)]] # 下左 下右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img3.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img3.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img3.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img3.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img3 = cv2.copyMakeBorder(np.transpose(syn_img3.numpy(), (1, 2, 0)), 
                                  pad_top, pad_bottom, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab4[0], lab4[2] = lab4[0] + pad_left, lab4[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab4[1], lab4[3] = lab4[1] + pad_top, lab4[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    """
    # 可视化拼贴图片的标签是否与目标匹配
    label_tensor = torch.cat((torch.from_numpy(np.array([lab1])), 
                              torch.from_numpy(np.array([lab2])), 
                              torch.from_numpy(np.array([lab3])), 
                              torch.from_numpy(np.array([lab4]))), 0)
    _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
    syn_test = copy.deepcopy(syn_img3)
    for i in range(len(label_tensor)):
        box = label_tensor[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[0] * 255).astype(np.uint8).tolist()
        cv2.rectangle(syn_test, (x0, y0), (x1, y1), color, 2)
    cv2.imwrite(save_dir + str(count) + '_test.jpg', syn_test)
    """
    
    cv2.imwrite(save_dir + str(count) + '.jpg', syn_img3) # cv2.imwrite reqire [h, w, c]
    
    label_tensor = torch.cat((torch.from_numpy(np.array([lab1])), 
                              torch.from_numpy(np.array([lab2])), 
                              torch.from_numpy(np.array([lab3])), 
                              torch.from_numpy(np.array([lab4]))), 0)
    
    xml_label_generate(syn_img3.shape[0], syn_img3.shape[1], label_tensor, save_dir + str(count) + '.xml')


def four_to_one_less_config2_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_area = random.uniform(1/2, 1) # 生成[1/2, 1)之间的随机数,用于取得合成图与原图的比例
    ratio_width = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    ratio_height_left = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下四图块之间的比例
    ratio_height_right = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下四图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * ratio_area * 0.8 * ratio_height_left * ratio_width
    search_patch_2_low = height * width * ratio_area * 0.8 * (1-ratio_height_left) * ratio_width
    search_patch_3_low = height * width * ratio_area * 0.8 * ratio_height_right * (1-ratio_width)
    search_patch_4_low = height * width * ratio_area * 0.8 * (1-ratio_height_right) * (1-ratio_width)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = search_patch_4_low / scope[0] # 拼图面积 / 目标面积最小值
        
    return (ratio_area, ratio_width, ratio_height_left, ratio_height_right, 
           [search_patch_1_low, proportion_1_low, proportion_1_high], 
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high],
           [search_patch_4_low, proportion_4_low, proportion_4_high])


class four_to_one_less_config2_search:
    def __init__(self, list1, list2, list3, list4, csv_path, ratio_area, ratio_width, ratio_height_left, ratio_height_right, height, width):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.list4 = list4
        
        self.csv_path = csv_path
        
        self.ratio_area = ratio_area
        self.ratio_width = ratio_width
        self.ratio_height_left = ratio_height_left
        self.ratio_height_right = ratio_height_right
        self.height = height
        self.width = width

    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
        search_area_4_low, proportion_4_low, proportion_4_high = self.list4
    
        self.potential_area_1, self.potential_area_2, self.potential_area_3, self.potential_area_4 = [], [], [], []
    
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
    
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
        
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
        
            # 判断该目标是否满足search_area_4的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
            else:
                if proportion_4_high > list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
    
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        assert len(self.potential_area_4) != 0, 'No patch match the search condition 4'
    
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        print(len(self.potential_area_3))
        print(len(self.potential_area_4))
    
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = [], [], [], []
    
        # 初始化计数器
        count = 0
    
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4) != 0:
            self.i, self.j, self.k, self.v = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3), random.choice(self.potential_area_4)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4)):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k, self.v = self.none_same_patch()
                self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img4 = cv2.imread(img_file + self.v[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                img4 = img4[self.v[7]:self.v[9], self.v[6]:self.v[8]]
                lab4 = [self.v[1]-self.v[6], self.v[2]-self.v[7], self.v[3]-self.v[6], self.v[4]-self.v[7], self.v[5]]
                        
                four_to_one_less_config2_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, 
                                                    self.ratio_area, self.ratio_width, self.ratio_height_left, self.ratio_height_right, 
                                                    count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
                self.patch4_used, self.potential_area_4, self.v = self.store_remove_pick(self.patch4_used, self.potential_area_4, self.v)
    
    
    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info


    def remove_pick(self, remove_list, patch_info):
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info
        

    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块  
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.i, self.v) and \
               self.compare_info(self.j, self.k) and self.compare_info(self.k, self.v)) == False):
            self.i, self.j, self.k, self.v = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3), random.choice(self.potential_area_4)
        
        return self.i, self.j, self.k, self.v # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used or self.i in self.patch4_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.j in self.patch1_used or self.j in self.patch3_used or self.j in self.patch4_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.k in self.patch1_used or self.k in self.patch2_used or self.k in self.patch4_used):
            self.potential_area_3, self.k = self.remove_pick(self.potential_area_3, self.k)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.v in self.patch1_used or self.v in self.patch2_used or self.v in self.patch3_used):
            self.potential_area_4, self.v = self.remove_pick(self.potential_area_4, self.v)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()

        return self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used


def four_to_one_less_config2_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, ratio_area, ratio_width, ratio_height_left, ratio_height_right, count, height, width):
    # 左上图块如果w < h则翻转目标变成w > h
    if img1.shape[1] < img1.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab1[1] # xmin_new = ymin
        ymin_new = img1.shape[1] - lab1[2] # ymin_new = width - xmax
        xmax_new = lab1[3] # xmax_new = ymin + target_height
        ymax_new = img1.shape[1] - lab1[0] # ymax_new = width - xmax + target_width
        
        lab1[0] = xmin_new
        lab1[1] = ymin_new
        lab1[2] = xmax_new
        lab1[3] = ymax_new
        
        # 再把图片翻转90°
        img1 = np.rot90(img1)
    
    # 左下图块如果w < h则翻转目标变成w > h
    if img2.shape[1] < img2.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab2[1] # xmin_new = ymin
        ymin_new = img2.shape[1] - lab2[2] # ymin_new = width - xmax
        xmax_new = lab2[3] # xmax_new = ymin + target_height
        ymax_new = img2.shape[1] - lab2[0] # ymax_new = width - xmax + target_width
        
        lab2[0] = xmin_new
        lab2[1] = ymin_new
        lab2[2] = xmax_new
        lab2[3] = ymax_new
        
        # 再把图片翻转90°
        img2 = np.rot90(img1)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 取得二合一新宽高与缩放模式
    new_h, new_w, mode = height * pow(ratio_area, 0.5), width * pow(ratio_area, 0.5), 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height_left), int(new_w * ratio_width)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height_left)), int(new_w * ratio_width)), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * ratio_height_left)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * (1-ratio_height_left))), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img1 = torch.zeros((3, divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]))
    syn_img1[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img1[:3, divide_img_1.shape[1]:, :divide_img_2.shape[2]].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [0.0, int(new_h * ratio_height_left), 0.0, int(new_h * ratio_height_left)]] # 左上 左下
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 右上图块如果w < h则翻转目标变成w > h
    if img3.shape[1] < img3.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab3[1] # xmin_new = ymin
        ymin_new = img3.shape[1] - lab3[2] # ymin_new = width - xmax
        xmax_new = lab3[3] # xmax_new = ymin + target_height
        ymax_new = img3.shape[1] - lab3[0] # ymax_new = width - xmax + target_width
        
        lab3[0] = xmin_new
        lab3[1] = ymin_new
        lab3[2] = xmax_new
        lab3[3] = ymax_new
        
        # 再把图片翻转90°
        img3 = np.rot90(img3)
    
    # 右下图块如果w < h则翻转目标变成w > h
    if img4.shape[1] < img4.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab4[1] # xmin_new = ymin
        ymin_new = img4.shape[1] - lab4[2] # ymin_new = width - xmax
        xmax_new = lab4[3] # xmax_new = ymin + target_height
        ymax_new = img4.shape[1] - lab4[0] # ymax_new = width - xmax + target_width
        
        lab4[0] = xmin_new
        lab4[1] = ymin_new
        lab4[2] = xmax_new
        lab4[3] = ymax_new
        
        # 再把图片翻转90°
        img4 = np.rot90(img4)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    img4 = np.transpose(img4, (2, 0, 1))
    
    # 缩放img3与img4至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3.copy()).unsqueeze(0), size=(int(new_h * ratio_height_right), int(new_w * (1-ratio_width))), mode=mode).squeeze(0)
    divide_img_4 = torch.nn.functional.interpolate(torch.from_numpy(img4.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height_right)), int(new_w * (1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width)), int(new_h * ratio_height_right)), (img3.shape[2], img3.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width)), int(new_h * (1-ratio_height_right))), (img4.shape[2], img4.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img2 = torch.zeros((3, divide_img_3.shape[1]+divide_img_4.shape[1], divide_img_3.shape[2]))
    syn_img2[:3, :divide_img_3.shape[1], :divide_img_3.shape[2]].copy_(divide_img_3)
    syn_img2[:3, divide_img_3.shape[1]:, :divide_img_4.shape[2]].copy_(divide_img_4)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [0.0, int(new_h * ratio_height_right), 0.0, int(new_h * ratio_height_right)]] # 左上 左下
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img3 = torch.zeros((3, divide_img_3.shape[1]+divide_img_4.shape[1], syn_img1.shape[2]+syn_img2.shape[2]))
    syn_img3[:3, :divide_img_3.shape[1]+divide_img_4.shape[1], :syn_img1.shape[2]].copy_(syn_img1)
    syn_img3[:3, :divide_img_3.shape[1]+divide_img_4.shape[1], syn_img1.shape[2]:].copy_(syn_img2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0], [int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0]] # 左 右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img3.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img3.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img3.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img3.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img3 = cv2.copyMakeBorder(np.transpose(syn_img3.numpy(), (1, 2, 0)), 
                                  pad_top, pad_bottom, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab4[0], lab4[2] = lab4[0] + pad_left, lab4[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab4[1], lab4[3] = lab4[1] + pad_top, lab4[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img3) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]


# 四合一解析算法,合成图面积大于原图面积,输入所需目标的尺寸范围并设定两图块的面积比例
def four_to_one_more_config1_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_width_top = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    ratio_width_bot = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左下右下两图块之间的比例
    ratio_height = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下四图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * 0.8 * ratio_height * ratio_width_top
    search_patch_2_low = height * width * 0.8 * ratio_height * (1-ratio_width_top)
    search_patch_3_low = height * width * 0.8 * (1-ratio_height) * ratio_width_bot
    search_patch_4_low = height * width * 0.8 * (1-ratio_height) * (1-ratio_width_bot)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = search_patch_4_low / scope[0] # 拼图面积 / 目标面积最小值
        
    return (ratio_width_top, ratio_width_bot, ratio_height,
           [search_patch_1_low, proportion_1_low, proportion_1_high],
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high],
           [search_patch_4_low, proportion_4_low, proportion_4_high])


class four_to_one_more_config1_search:
    def __init__(self, list1, list2, list3, list4, csv_path, ratio_width_top, ratio_width_bot, ratio_height, height, width):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.list4 = list4
        
        self.csv_path = csv_path
        
        self.ratio_width_top = ratio_width_top
        self.ratio_width_bot = ratio_width_bot
        self.ratio_height = ratio_height
        self.height = height
        self.width = width
        
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
        search_area_4_low, proportion_4_low, proportion_4_high = self.list4
    
        self.potential_area_1, self.potential_area_2, self.potential_area_3, self.potential_area_4 = [], [], [], []
    
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
    
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
        
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
        
            # 判断该目标是否满足search_area_4的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
            else:
                if proportion_4_high > list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
    
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        assert len(self.potential_area_4) != 0, 'No patch match the search condition 4'
    
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        print(len(self.potential_area_3))
        print(len(self.potential_area_4))
    
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = [], [], [], []
    
        # 初始化计数器
        count = 0
    
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4) != 0:
            self.i, self.j, self.k, self.v = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3), random.choice(self.potential_area_4)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4)):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k, self.v = self.none_same_patch()
                self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img4 = cv2.imread(img_file + self.v[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                img4 = img4[self.v[7]:self.v[9], self.v[6]:self.v[8]]
                lab4 = [self.v[1]-self.v[6], self.v[2]-self.v[7], self.v[3]-self.v[6], self.v[4]-self.v[7], self.v[5]]
                        
                four_to_one_more_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, 
                                                    self.ratio_width_top, self.ratio_width_bot, self.ratio_height, 
                                                    count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
                self.patch4_used, self.potential_area_4, self.v = self.store_remove_pick(self.patch4_used, self.potential_area_4, self.v)
    
    
    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info


    def remove_pick(self, remove_list, patch_info):
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info
        

    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块  
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.i, self.v) and \
               self.compare_info(self.j, self.k) and self.compare_info(self.k, self.v)) == False):
            self.i, self.j, self.k, self.v = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3), random.choice(self.potential_area_4)
        
        return self.i, self.j, self.k, self.v # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used or self.i in self.patch4_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.j in self.patch1_used or self.j in self.patch3_used or self.j in self.patch4_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.k in self.patch1_used or self.k in self.patch2_used or self.k in self.patch4_used):
            self.potential_area_3, self.k = self.remove_pick(self.potential_area_3, self.k)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.v in self.patch1_used or self.v in self.patch2_used or self.v in self.patch3_used):
            self.potential_area_4, self.v = self.remove_pick(self.potential_area_4, self.v)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()

        return self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used


def four_to_one_more_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, ratio_width_top, ratio_width_bot, ratio_height, count, height, width):
    # 左上图块如果w < h则翻转目标变成w > h
    if img1.shape[1] < img1.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab1[1] # xmin_new = ymin
        ymin_new = img1.shape[1] - lab1[2] # ymin_new = width - xmax
        xmax_new = lab1[3] # xmax_new = ymin + target_height
        ymax_new = img1.shape[1] - lab1[0] # ymax_new = width - xmax + target_width
        
        lab1[0] = xmin_new
        lab1[1] = ymin_new
        lab1[2] = xmax_new
        lab1[3] = ymax_new
        
        # 再把图片翻转90°
        img1 = np.rot90(img1)
    
    # 右上侧图块如果w > h则翻转目标变成w < h
    if img2.shape[1] > img2.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab2[1] # xmin_new = ymin
        ymin_new = img2.shape[1] - lab2[2] # ymin_new = width - xmax
        xmax_new = lab2[3] # xmax_new = ymin + target_height
        ymax_new = img2.shape[1] - lab2[0] # ymax_new = width - xmax + target_width
        
        lab2[0] = xmin_new
        lab2[1] = ymin_new
        lab2[2] = xmax_new
        lab2[3] = ymax_new
        
        # 再把图片翻转90°
        img2 = np.rot90(img2)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 取得二合一新宽高与缩放模式
    new_h, new_w, mode = height, width, 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * ratio_width_top)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * (1-ratio_width_top))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width_top), int(new_h * ratio_height)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width_top)), int(new_h * ratio_height)), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img1 = torch.zeros((3, divide_img_1.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img1[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img1[:3, :divide_img_1.shape[1], divide_img_1.shape[2]:].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width_top), 0.0, int(new_w * ratio_width_top), 0.0]] # 上左 上右
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 左下图块如果w < h则翻转目标变成w > h
    if img3.shape[1] < img3.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab3[1] # xmin_new = ymin
        ymin_new = img3.shape[1] - lab3[2] # ymin_new = width - xmax
        xmax_new = lab3[3] # xmax_new = ymin + target_height
        ymax_new = img3.shape[1] - lab3[0] # ymax_new = width - xmax + target_width
        
        lab3[0] = xmin_new
        lab3[1] = ymin_new
        lab3[2] = xmax_new
        lab3[3] = ymax_new
        
        # 再把图片翻转90°
        img3 = np.rot90(img3)
    
    # 右下侧图块如果w > h则翻转目标变成w < h
    if img4.shape[1] > img4.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab4[1] # xmin_new = ymin
        ymin_new = img4.shape[1] - lab4[2] # ymin_new = width - xmax
        xmax_new = lab4[3] # xmax_new = ymin + target_height
        ymax_new = img4.shape[1] - lab4[0] # ymax_new = width - xmax + target_width
        
        lab4[0] = xmin_new
        lab4[1] = ymin_new
        lab4[2] = xmax_new
        lab4[3] = ymax_new
        
        # 再把图片翻转90°
        img4 = np.rot90(img4)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    img4 = np.transpose(img4, (2, 0, 1))
    
    # 缩放img3与img4至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w * ratio_width_bot)), mode=mode).squeeze(0)
    divide_img_4 = torch.nn.functional.interpolate(torch.from_numpy(img4.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w * (1-ratio_width_bot))), mode=mode).squeeze(0)
    
    # 缩放img3与img4的标签
    ratios_3 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width_bot), int(new_h * (1-ratio_height))), (img3.shape[2], img3.shape[1]))) # width, height
    ratios_4 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width_bot)), int(new_h * (1-ratio_height))), (img4.shape[2], img4.shape[1]))) # width, height
    lab3[0] = lab3[0] * ratios_3[0] # xmin * ratio_width
    lab3[2] = lab3[2] * ratios_3[0] # xmax * ratio_width
    lab3[1] = lab3[1] * ratios_3[1] # ymin * ratio_height
    lab3[3] = lab3[3] * ratios_3[1] # ymax * ratio_height
    lab4[0] = lab4[0] * ratios_4[0] # xmin * ratio_width
    lab4[2] = lab4[2] * ratios_4[0] # xmax * ratio_width
    lab4[1] = lab4[1] * ratios_4[1] # ymin * ratio_height
    lab4[3] = lab4[3] * ratios_4[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img2 = torch.zeros((3, divide_img_3.shape[1], divide_img_3.shape[2]+divide_img_4.shape[2]))
    syn_img2[:3, :divide_img_3.shape[1], :divide_img_3.shape[2]].copy_(divide_img_3)
    syn_img2[:3, :divide_img_3.shape[1], divide_img_3.shape[2]:].copy_(divide_img_4)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width_bot), 0.0, int(new_w * ratio_width_bot), 0.0]] # 下左 下右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img3 = torch.zeros((3, syn_img1.shape[1]+syn_img2.shape[1], divide_img_3.shape[2]+divide_img_4.shape[2]))
    syn_img3[:3, :syn_img1.shape[1], :divide_img_3.shape[2]+divide_img_4.shape[2]].copy_(syn_img1)
    syn_img3[:3, syn_img1.shape[1]:, :divide_img_3.shape[2]+divide_img_4.shape[2]].copy_(syn_img2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)], [0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)]] # 下左 下右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img3.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img3.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img3.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img3.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img3 = cv2.copyMakeBorder(np.transpose(syn_img3.numpy(), (1, 2, 0)), 
                                  pad_top, pad_bottom, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab4[0], lab4[2] = lab4[0] + pad_left, lab4[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab4[1], lab4[3] = lab4[1] + pad_top, lab4[3] + pad_top # ymin + pad_top, ymax + pad_top

    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img3) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]


def four_to_one_more_config2_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_width = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    ratio_height_left = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下四图块之间的比例
    ratio_height_right = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下四图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * 0.8 * ratio_height_left * ratio_width
    search_patch_2_low = height * width * 0.8 * (1-ratio_height_left) * ratio_width
    search_patch_3_low = height * width * 0.8 * ratio_height_right * (1-ratio_width)
    search_patch_4_low = height * width * 0.8 * (1-ratio_height_right) * (1-ratio_width)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_4_low = search_patch_4_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_4_high = search_patch_4_low / scope[0] # 拼图面积 / 目标面积最小值
        
    return (ratio_width, ratio_height_left, ratio_height_right, 
           [search_patch_1_low, proportion_1_low, proportion_1_high], 
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high],
           [search_patch_4_low, proportion_4_low, proportion_4_high])


class four_to_one_more_config2_search:
    def __init__(self, list1, list2, list3, list4, csv_path, ratio_width, ratio_height_left, ratio_height_right, height, width):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.list4 = list4
        
        self.csv_path = csv_path
        
        self.ratio_width = ratio_width
        self.ratio_height_left = ratio_height_left
        self.ratio_height_right = ratio_height_right
        self.height = height
        self.width = width
    
    
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
        search_area_4_low, proportion_4_low, proportion_4_high = self.list4
    
        self.potential_area_1, self.potential_area_2, self.potential_area_3, self.potential_area_4 = [], [], [], []
    
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
    
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
        
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
        
            # 判断该目标是否满足search_area_4的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
            else:
                if proportion_4_high > list2[12] > proportion_4_low and list2[11] > search_area_4_low:
                    self.potential_area_4.append(list2)
        
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        assert len(self.potential_area_4) != 0, 'No patch match the search condition 4'
        
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        print(len(self.potential_area_3))
        print(len(self.potential_area_4))
    
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = [], [], [], []
        
        # 初始化计数器
        count = 0
    
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4) != 0:
            self.i, self.j, self.k, self.v = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3), random.choice(self.potential_area_4)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4)):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k, self.v = self.none_same_patch()
                self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img4 = cv2.imread(img_file + self.v[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                img4 = img4[self.v[7]:self.v[9], self.v[6]:self.v[8]]
                lab4 = [self.v[1]-self.v[6], self.v[2]-self.v[7], self.v[3]-self.v[6], self.v[4]-self.v[7], self.v[5]]
                        
                four_to_one_more_config2_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, 
                                                    self.ratio_width, self.ratio_height_left, self.ratio_height_right, 
                                                    count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
                self.patch4_used, self.potential_area_4, self.v = self.store_remove_pick(self.patch4_used, self.potential_area_4, self.v)
    
    
    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info


    def remove_pick(self, remove_list, patch_info):
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info
        

    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块  
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.i, self.v) and \
               self.compare_info(self.j, self.k) and self.compare_info(self.k, self.v)) == False):
            self.i, self.j, self.k, self.v = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3), random.choice(self.potential_area_4)
        
        return self.i, self.j, self.k, self.v # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used or self.i in self.patch4_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.j in self.patch1_used or self.j in self.patch3_used or self.j in self.patch4_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.k in self.patch1_used or self.k in self.patch2_used or self.k in self.patch4_used):
            self.potential_area_3, self.k = self.remove_pick(self.potential_area_3, self.k)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
        while(self.v in self.patch1_used or self.v in self.patch2_used or self.v in self.patch3_used):
            self.potential_area_4, self.v = self.remove_pick(self.potential_area_4, self.v)
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()

        return self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used


def four_to_one_more_config2_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, ratio_width, ratio_height_left, ratio_height_right, count, height, width):
    # 左上图块如果w < h则翻转目标变成w > h
    if img1.shape[1] < img1.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab1[1] # xmin_new = ymin
        ymin_new = img1.shape[1] - lab1[2] # ymin_new = width - xmax
        xmax_new = lab1[3] # xmax_new = ymin + target_height
        ymax_new = img1.shape[1] - lab1[0] # ymax_new = width - xmax + target_width
        
        lab1[0] = xmin_new
        lab1[1] = ymin_new
        lab1[2] = xmax_new
        lab1[3] = ymax_new
        
        # 再把图片翻转90°
        img1 = np.rot90(img1)
    
    # 左下图块如果w < h则翻转目标变成w > h
    if img2.shape[1] < img2.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab2[1] # xmin_new = ymin
        ymin_new = img2.shape[1] - lab2[2] # ymin_new = width - xmax
        xmax_new = lab2[3] # xmax_new = ymin + target_height
        ymax_new = img2.shape[1] - lab2[0] # ymax_new = width - xmax + target_width
        
        lab2[0] = xmin_new
        lab2[1] = ymin_new
        lab2[2] = xmax_new
        lab2[3] = ymax_new
        
        # 再把图片翻转90°
        img2 = np.rot90(img1)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 取得二合一新宽高与缩放模式
    new_h, new_w, mode = height, width, 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height_left), int(new_w * ratio_width)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height_left)), int(new_w * ratio_width)), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * ratio_height_left)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * (1-ratio_height_left))), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img1 = torch.zeros((3, divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]))
    syn_img1[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img1[:3, divide_img_1.shape[1]:, :divide_img_2.shape[2]].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [0.0, int(new_h * ratio_height_left), 0.0, int(new_h * ratio_height_left)]] # 左上 左下
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 右上图块如果w < h则翻转目标变成w > h
    if img3.shape[1] < img3.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab3[1] # xmin_new = ymin
        ymin_new = img3.shape[1] - lab3[2] # ymin_new = width - xmax
        xmax_new = lab3[3] # xmax_new = ymin + target_height
        ymax_new = img3.shape[1] - lab3[0] # ymax_new = width - xmax + target_width
        
        lab3[0] = xmin_new
        lab3[1] = ymin_new
        lab3[2] = xmax_new
        lab3[3] = ymax_new
        
        # 再把图片翻转90°
        img3 = np.rot90(img3)
    
    # 右下图块如果w < h则翻转目标变成w > h
    if img4.shape[1] < img4.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab4[1] # xmin_new = ymin
        ymin_new = img4.shape[1] - lab4[2] # ymin_new = width - xmax
        xmax_new = lab4[3] # xmax_new = ymin + target_height
        ymax_new = img4.shape[1] - lab4[0] # ymax_new = width - xmax + target_width
        
        lab4[0] = xmin_new
        lab4[1] = ymin_new
        lab4[2] = xmax_new
        lab4[3] = ymax_new
        
        # 再把图片翻转90°
        img4 = np.rot90(img4)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    img4 = np.transpose(img4, (2, 0, 1))
    
    # 缩放img3与img4至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3.copy()).unsqueeze(0), size=(int(new_h * ratio_height_right), int(new_w * (1-ratio_width))), mode=mode).squeeze(0)
    divide_img_4 = torch.nn.functional.interpolate(torch.from_numpy(img4.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height_right)), int(new_w * (1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width)), int(new_h * ratio_height_right)), (img3.shape[2], img3.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width)), int(new_h * (1-ratio_height_right))), (img4.shape[2], img4.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img2 = torch.zeros((3, divide_img_3.shape[1]+divide_img_4.shape[1], divide_img_3.shape[2]))
    syn_img2[:3, :divide_img_3.shape[1], :divide_img_3.shape[2]].copy_(divide_img_3)
    syn_img2[:3, divide_img_3.shape[1]:, :divide_img_4.shape[2]].copy_(divide_img_4)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [0.0, int(new_h * ratio_height_right), 0.0, int(new_h * ratio_height_right)]] # 左上 左下
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img3 = torch.zeros((3, divide_img_3.shape[1]+divide_img_4.shape[1], syn_img1.shape[2]+syn_img2.shape[2]))
    syn_img3[:3, :divide_img_3.shape[1]+divide_img_4.shape[1], :syn_img1.shape[2]].copy_(syn_img1)
    syn_img3[:3, :divide_img_3.shape[1]+divide_img_4.shape[1], syn_img1.shape[2]:].copy_(syn_img2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0], [int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0]] # 左 右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img3.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img3.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img3.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img3.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img3 = cv2.copyMakeBorder(np.transpose(syn_img3.numpy(), (1, 2, 0)), 
                                  pad_top, pad_bottom, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab4[0], lab4[2] = lab4[0] + pad_left, lab4[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab4[1], lab4[3] = lab4[1] + pad_top, lab4[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img3) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]