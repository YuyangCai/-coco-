import os
import json
import cv2
import random
import time
from PIL import Image
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

'''
将需要进行测量全面评价指标的模型送入验证，这会生成一个.json文件
'''

model = YOLO('/runs/detect/80yoloc/weights/best.pt')

validation_results = model.val(data = 'coco.yaml',save_json=True)


