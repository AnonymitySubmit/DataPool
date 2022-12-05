# -*- coding: utf-8 -*-

# import cv2
"""
from datapool_syn2to1 import two_to_one_less_analysis, two_to_one_less_search
from datapool_syn2to1 import two_to_one_more_analysis, two_to_one_more_search

from datapool_syn3to1 import three_to_one_less_config1_analysis, three_to_one_less_config1_search
from datapool_syn3to1 import three_to_one_less_config2_analysis, three_to_one_less_config2_search
from datapool_syn3to1 import three_to_one_more_config1_analysis, three_to_one_more_config1_search
from datapool_syn3to1 import three_to_one_more_config2_analysis, three_to_one_more_config2_search
"""

from datapool_syn4to1_v4 import four_to_one_less_config1_analysis, four_to_one_less_config1_search
"""
from datapool_syn4to1 import four_to_one_less_config2_analysis, four_to_one_less_config2_search
from datapool_syn4to1 import four_to_one_more_config1_analysis, four_to_one_more_config1_search
from datapool_syn4to1 import four_to_one_more_config2_analysis, four_to_one_more_config2_search
"""

if __name__ == "__main__":
    # 设定target_info路径
    csv_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/VOC_0712trainval_ObjectInfo.csv' # D:/AICV-DSTRethink/DataPoolTest&Results/VOC_0712trainval_noandoverlapmix.csv
    save_path = 'D:/AICV-DSTRethink/DataPoolTest&Results/vocresults/'
    
    # 取得图片宽高
    # img = cv2.imread('D:/AICV-YoloXReDST-SGD/000000000049.jpg') # cv2.imread [h, w, c]
    # height, width = img.shape[0], img.shape[1]
    scope, height, width = [0, 1024], 512, 512 # 32 * 32 = , 64 * 64 = 4096
    
    """
    # 二合一增强, 生成图小于原图
    ratio_area, ratio_divide, list1, list2 = two_to_one_less_analysis(scope, height, width)
    syn_tool_2l = two_to_one_less_search(list1, list2, csv_path, ratio_area, ratio_divide, height, width)
    syn_tool_2l.do_search()
    
    # 二合一增强, 生成图大于原图
    ratio_divide, list1, list2 = two_to_one_more_analysis(scope, height, width)
    syn_tool_2m = two_to_one_more_search(list1, list2, csv_path, ratio_divide, height, width)
    syn_tool_2m.do_search()
    
    # 三合一构型1增强, 生成图小于原图
    ratio_area, ratio_width, ratio_height, list1, list2, list3 = three_to_one_less_config1_analysis(scope, height, width)
    syn_tool_3lc1 = three_to_one_less_config1_search(list1, list2, list3, csv_path, ratio_area, ratio_width, ratio_height, height, width)
    syn_tool_3lc1.do_search()
    
    # 三合一构型2增强, 生成图小于原图
    ratio_area, ratio_width, ratio_height, list1, list2, list3 = three_to_one_less_config2_analysis(scope, height, width)
    syn_tool_3lc2 = three_to_one_less_config2_search(list1, list2, list3, csv_path, ratio_area, ratio_width, ratio_height, height, width)
    syn_tool_3lc2.do_search()
    
    # 三合一构型1增强, 生成图大于原图
    ratio_width, ratio_height, list1, list2, list3 = three_to_one_more_config1_analysis(scope, height, width)
    syn_tool_3mc1 = three_to_one_more_config1_search(list1, list2, list3, csv_path, ratio_width, ratio_height, height, width)
    syn_tool_3mc1.do_search()
    
    # 三合一构型2增强, 生成图大于原图
    ratio_width, ratio_height, list1, list2, list3 = three_to_one_more_config2_analysis(scope, height, width)
    syn_tool_3mc2 = three_to_one_more_config2_search(list1, list2, list3, csv_path, ratio_width, ratio_height, height, width)
    syn_tool_3mc2.do_search()
    """
    # 四合一构型1增强, 生成图小于原图
    ratio_width_top, ratio_width_bot, ratio_height = [1/3, 2/3], [1/3, 2/3], [1/3, 2/3]
    for i in ratio_height:
        for j in ratio_width_top:
            for k in ratio_width_bot:
                if i == 1/3 and j == 1/3 and k == 1/3:
                    name = 'hig_lef_lef'
                if i == 1/3 and j == 1/3 and k == 2/3:
                    name = 'hig_lef_rig'
                if i == 1/3 and j == 2/3 and k == 1/3:
                    name = 'hig_rig_lef'
                if i == 1/3 and j == 2/3 and k == 2/3:
                    name = 'hig_rig_rig'
                if i == 2/3 and j == 1/3 and k == 1/3:
                    name = 'low_lef_lef'
                if i == 2/3 and j == 1/3 and k == 2/3:
                    name = 'low_lef_rig'
                if i == 2/3 and j == 2/3 and k == 1/3:
                    name = 'low_rig_lef'
                if i == 2/3 and j == 2/3 and k == 2/3:
                    name = 'low_rig_rig'
                list1, list2, list3, list4 = four_to_one_less_config1_analysis(scope, height, width, i, j, k)
                syn_tool_4lc1 = four_to_one_less_config1_search(list1, list2, list3, list4, csv_path, j, k, i, height, width)
                syn_tool_4lc1.do_search(save_path + 'h' + str(round(i,2)) + '_wt' + str(round(j,2)) + '_wb' + str(round(k,2)) + '/' + name + '_')
    """
    # 四合一构型2增强, 生成图小于原图
    ratio_area, ratio_width, ratio_height_left, ratio_height_right, list1, list2, list3, list4 = four_to_one_less_config2_analysis(scope, height, width)
    syn_tool_4lc2 = four_to_one_less_config2_search(list1, list2, list3, list4, csv_path, ratio_area, ratio_width, ratio_height_left, ratio_height_right, height, width)
    syn_tool_4lc2.do_search()
    
    # 四合一构型1增强, 生成图大于原图
    ratio_width_top, ratio_width_bot, ratio_height, list1, list2, list3, list4 = four_to_one_more_config1_analysis(scope, height, width)
    syn_tool_4mc1 = four_to_one_more_config1_search(list1, list2, list3, list4, csv_path, ratio_width_top, ratio_width_bot, ratio_height, height, width)
    syn_tool_4mc1.do_search()
    
    # 四合一构型2增强, 生成图大于原图
    ratio_width, ratio_height_left, ratio_height_right, list1, list2, list3, list4 = four_to_one_more_config2_analysis(scope, height, width)
    syn_tool_4mc2 = four_to_one_more_config2_search(list1, list2, list3, list4, csv_path, ratio_width, ratio_height_left, ratio_height_right, height, width)
    syn_tool_4mc2.do_search()
    """