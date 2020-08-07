import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import model
import torchvision_model
import torchvision.ops as ops
'''
description: 
param {type} 人脸图像batch，模型，分数阈值，iou阈值
return {type} 人脸boxes，检测分数
'''
def get_detections(image_batch, model, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        classifications, bboxes, _ = model(image_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_scores = []
        
        for i in range(batch_size):
            classification = torch.exp(classifications[i,:,:])
            bbox = bboxes[i,:,:]

            # 阈值选择
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax==0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice
            
            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]

            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_scores.append(keep_scores)
    return picked_boxes, picked_scores


'''
description: 参数输入
param {type} 
return {type} args[list]
'''
def arg_parse():
    parser = argparse.ArgumentParser(description="Face detection")

    parser.add_argument("--model_path", default="./model.pt", help="pretraining parameter path")
    parser.add_argument("--input", default="./must_test.jpg", help="input image path")
    parser.add_argument("--output", default="./detected.jpg", help="output detected image save path")
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()

    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)

    # Load 训练好的权重文件
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(args.model_path)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()

    # 读取文件
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    img = img.permute(2,0,1) # 通道转换  h, w, c -> c, h, w

    # 进入网络
    input_img = img.unsqueeze(0).float().cuda() #扩展维度
    picked_boxes, picked_scores = get_detections(input_img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)

    # np_img = resized_img.cpu().permute(1,2,0).numpy()
    np_img = img.cpu().permute(1,2,0).numpy()
    np_img.astype(int)
    img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX# 设置字体

    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, score in zip(boxes,picked_scores[j]):
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=2)
                cv2.putText(img, text=str(score.item())[:5], org=(box[0],box[1]), fontFace=font, fontScale=0.5,
                            thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

    # 保存并展示处理后的图像
    cv2.imwrite(args.output, img)
    cv2.imshow('RetinaFace-Pytorch',img)
    cv2.waitKey() # 

if __name__=='__main__':
    main()
