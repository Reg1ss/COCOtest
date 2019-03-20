from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]  # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))

#initialize COCO ground truth api

dataDir='/home/kuangrx/TopEval'
#dataType='val2017'
labelName = 'label_0003'
annFile = '%s/json/%s.json'%(dataDir,labelName)
cocoGt=COCO(annFile)

#initialize COCO detections api

resultName = 'result_0003'
resFile='%s/json/%s.json'%(dataDir,resultName)

# resFile='%s/result/%s_%s_fake%s100_results.json'
# resFile = resFile%(dataDir, prefix, annType)
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[1:111]
# imgId = imgIds[np.random.randint(1,111)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
