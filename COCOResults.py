# -*- coding:utf-8 -*-
# !/usr/bin/env python

import cv2
import numpy as np
import glob
import PIL.Image
import os, sys
import json

# def loadFont():
#     f = open("/home/kuangrx/TopEval/testJson/result1.json")
#     setting = json.load(f)
#     family = setting['annotations'][10]
#     # size = setting['fontSize']
#     return family
#
# t = loadFont()
# print(t)

class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.data_coco = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.score = 0
        self.ob = []

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.xml):
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()

            self.json_file = json_file
            # print("self.json", self.json_file)
            self.num = num
            # print(self.num)
            path = os.path.dirname(self.json_file)
            # print(path)
            path = os.path.dirname(path)
            with open(json_file, 'r') as fp:
                # print(fp)
                flag = 0
                for p in fp:
                    data = {}
                    f_name = 1
                    if 'filename' in p:
                        self.filen_ame = p.split('>')[1].split('<')[0]
                        # print(self.filen_ame)
                        f_name = 0

                        self.path = os.path.join(path, 'SegmentationObject', self.filen_ame.split('.')[0] + '.png')
                        # if self.path not in obj_path:
                        #    break

                    if 'score' in p:
                        self.score = float(p.split('>')[1].split('<')[0])

                        self.images.append(self.image())
                        # print(self.image())
                    if flag == 1:
                        self.supercategory = self.ob[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)

                        # 边界框
                        x1 = int(self.ob[1]);
                        y1 = int(self.ob[2]);
                        x2 = int(self.ob[3]);
                        y2 = int(self.ob[4])
                        self.rectangle = [x1, y1, x2, y2]
                        self.bbox = [x1, y2, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]

                        self.annID += 1
                        self.ob = []
                        flag = 0
                        if(self.score):
                            data['image_id'] = self.num + 1
                            data['category_id'] = self.getcatid(self.supercategory)
                            data['bbox'] = list(map(float, self.bbox))
                            data['score'] = self.score
                    elif f_name == 1:
                        if 'name' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                        if 'xmin' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                        if 'ymin' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                        if 'xmax' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                        if 'ymax' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])
                            flag = 1
                    if(len(data)):
                        self.data_coco.append(data)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie



    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i;
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i;
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox = []
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox  # list(contours[1][0].flatten())

    # '''
    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    # '''
    def data2coco(self):
        data_coco = self.data_coco
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


xml_file = glob.glob('/home/kuangrx/xmls/0003_score/*.xml')
# xml_file=['./Annotations/000032.xml']
# xml_file=['00000007_05499_d_0000037.xml']
PascalVOC2coco(xml_file, '/home/kuangrx/TopEval/json/result_0003.json')


