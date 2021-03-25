import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from conversion_utils import bbox_to_rbbox
import numpy as np
from numpy import random
from PIL import ImageFont, ImageDraw, Image
import os

class CvPutJaText:
    def __init__(self):
        pass

    @classmethod
    def puttext(cls, cv_image, text, point, font_path, font_size, color=(0,0,0)):
        font = ImageFont.truetype(font_path, font_size)
        
        cv_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)
        
        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font)
        
        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv2.cvtColor(cv_rgb_result_image, cv2.COLOR_RGB2BGR)

        return cv_bgr_result_image

def plot_one_box_j(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        #c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        c2 = c2[0] , c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf'
        img = CvPutJaText.puttext(img, label, (c1[0], c1[1] -18 ) , font_path, 18, (255, 255, 255) )
    return img

class ObjectDetector:
    def __init__(self, model_path):
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16

        self.stride = int(self.model.stride.max())  # model stride
        imgsz = 640
        self.img_size = check_img_size(imgsz, s=self.stride)  # check img_size

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False

    def __call__(self, img):
        if img is None:
            return None
        (height, width, channel) = img.shape
        preds = self.predict(img)

        detections = []
        for *xyxy, conf, cls in reversed(preds):
            xyxy[0] = int(xyxy[0].to('cpu').detach().numpy().copy())
            xyxy[1] = int(xyxy[1].to('cpu').detach().numpy().copy())
            xyxy[2] = int(xyxy[2].to('cpu').detach().numpy().copy())
            xyxy[3] = int(xyxy[3].to('cpu').detach().numpy().copy())
            conf = conf.to('cpu').detach().numpy()
            label = f'{self.names[int(cls)]}'
            list = {'bbox': xyxy, 'rbbox': bbox_to_rbbox(xyxy, img), 'label' : int(cls), 
            'label_name': f'{self.names[int(cls)]}', 'prob' : conf, 'color' :self.colors[int(cls)]}
            detections.append(list)

        #print('detections :{}'.format(detections))
        return [detections]

    def predict(self, img0):
        if img0 is None:
            return None
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment='True')[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        pred = pred[0]
        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        return pred

    def print_detections(self, det, im0):
        s = ''  # print string
        if len(det):           
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print("s :{}".format(s))
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{self.conv_name(self.names[int(cls)])} {conf:.2f}'
                #plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                im0 = plot_one_box_j(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
        return im0

    def conv_name(self, str):
        names = {'Blackback':'アケボノチョウ',
             'DoubleSaddle':'スダレチョウ',
             'Racoon':'チョウハン',
             'Blotchedfox':'ヒフキアイゴ',
             'Clowntrigger':'モンガラカワハギ',
             'Threadfin':'トゲチョウ',
             'Palette':'ナンヨウハギ',
             'Clownfish' :'カクレクマノミ', 
             'Longnose':'フエフキダイ',
             'YellowSurgeon':'キイロハギ',
             'Moorish':'ツノダシ',
             'Pyramid':'スミレチョウ'
             }
        return names.get(str, str)

if __name__ == '__main__':

    #model_path = 'weights/yolov5m.pt'
    model_path = 'runs/train/exp23/weights/best.pt'
    oc = ObjectDetector(model_path)
    #in_file = '/media/nakahira/GOLDUSB/fishesforNHK/akebono.mp4'
    in_file = '/media/nakahira/additional/yolov5/fishes/movie/blackback/blackback1.mp4'
    bs = os.path.basename(in_file)
    out_file = in_file.replace(bs, 'recog_'+bs)
    #capcher_file = 0
    cap = cv2.VideoCapture(in_file)   # capture from camera
    if not cap.isOpened():
        raise ImportError("Couldn't open video file or webcam.")
    # 幅と高さを取得
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, size)
    while True:
        ret,im = cap.read()
        if im is None:
            continue
        #oc(im)     
        pred = oc.predict(im)
        im = oc.print_detections(pred, im)
        cv2.imshow("yolov5 Result", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        writer.write(im)
    cap.release()
    cv2.destroyAllWindows()
    writer.release()
