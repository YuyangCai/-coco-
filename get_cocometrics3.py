import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

'''使用验证后的.json文件和真实.json文件进行评价指标计算'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='/txttococo/instances_val2017.json', help='training model path')#这里改为数据集转化后生成的json文件路径
    parser.add_argument('--pred_json', type=str, default='/runs/detect/val40/predictions.json', help='data yaml path')#这里改为val后生成的预测json文件路径
    
    return parser.parse_known_args()[0]
 
if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
 
    anno = COCO(anno_json)  # init annotations api
    print(pred_json)
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

