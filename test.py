#import cv2
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import zipfile

#---------------------------------
#Example1
# img = cv2.imread("/home/kuangrx/TopEval/image/0_25.jpg",-1)
# print(img)
# cv2.namedWindow("Image")
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#---------------------------------


# for i in os.listdir('/home/kuangrx/DataSet/images'):
#     print(i)

# Z = zipfile.ZipFile('/home/kuangrx/DataSet/images/train2017.zip')
# print(Z.namelist()[7])
#
# img_a = Z.read(Z.namelist()[7])
# print(type(img_a))
#
# img_flatten = np.frombuffer(img_a, 'B')
# img_cv = cv2.imdecode(img_flatten, cv2.IMREAD_ANYCOLOR)
# print(img_cv.shape)

# plt.subplot(231)
# plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
# plt.title('OpenCV')
# plt.axis('off')
# plt.show()

# pylab.rcParams['figure.figsize'] = (8.0,10.0)
#
# dataDir = '/home/kuangrx/DataSet/'
# dataType='train2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
#
# coco = COCO(annFile)

#displays categories
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories:\n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories:\n{}'.format(''.join(nms)))

# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['dog','person'])
# imgIds = coco.getImgIds(catIds=catIds)
# imgIds = coco.getImgIds(imgIds=[324158])
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# img = coco.loadImgs(imgIds[324158])
#
# #load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# #use url to load image
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()
#
# # load and display instance annotations
# plt.imshow(I);
# plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# plt.show()
