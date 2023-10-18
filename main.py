import numpy as np

from ultralytics import YOLO

import sys
sys.path.insert(0, './yolov5')

import argparse

import cv2
import torch


from yolov5.utils.general import  xyxy2xywh

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


def detect(opt):
    out, yolo_model, deep_sort_model,data_video = opt.output, opt.yolo_model, opt.deep_sort_model,opt.data_video
    x_min, y_min, x_max, y_max = 15,279,640,552


    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


    model = YOLO(yolo_model)


    vid = cv2.VideoCapture(data_video)
    skip_frame = 0
    visited = {}
    count = 0
    while True:
            skip_frame+=1
            if skip_frame%2 == 0:
                continue
            ret,im0= vid.read()
            if not ret:
                break
            im0 = cv2.resize(im0,(640,640))

            img = np.copy(im0)
            tensor = torch.tensor(img[None, :, :]) / 255.0
            tensor = tensor.permute(0, 3, 1, 2)

            output = model(tensor, verbose=False)

            boxes = output[0]

            if  len(boxes):
                boxes = boxes.boxes

                xywhs = xyxy2xywh(boxes.xyxy.detach())
                
                confs = boxes.conf
                clss = boxes.conf

                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        id = output[4]
                        # cls = output[5]
                        x1, y1, x2, y2 =   output[0:4]
                        cx,cy = get_centre(x1,y1,x2,y2)
                        if cy>=y_min and visited.get(id) is None:
                            visited[id] = 1
                            count+=1
                        x1 = x1.item()
                        y1 = y1.item()
                        x2 = x2.item()
                        y2 = y2.item()
                        im0 = cv2.rectangle(im0, (x1, y1), (x2, y2), (250, 0, 0), 2)
                        cv2.putText(im0, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255,255),1)




            else:
                deepsort.increment_ages()
            im0  = cv2.putText(im0,"number: "+str(count),(20,70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            im0 = cv2.line(im0,(0, y_min), (640, y_min), (0, 255, 0), 1)
            cv2.imshow("hihi", im0)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()







def get_centre(x1,y1,x2,y2):
    w_x = x2-x1
    w_y = y2-y1
    cx = int(x1+w_x/2)
    cy = int(y1+w_y/2)
    return (cx,cy)
def is_in_region_detect(xc,yc,x_min = 0,x_max =9999,y_min = 0,y_max = 9999):
    if xc<x_min or xc > x_max or yc<y_min or yc>y_max:
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov8.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--data_video', type=str, default='data_vehicels.mp4')
    parser.add_argument('--source', type=str, default='videos/Traffic.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")


    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        output = detect(opt)
