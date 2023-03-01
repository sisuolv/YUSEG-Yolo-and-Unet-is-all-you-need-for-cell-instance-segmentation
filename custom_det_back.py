# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import  time
t0 = time.time()
from pynvml import *
import torch

import argparse
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, create_dataloader
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import albumentations as A
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler
from ensemble_boxes import weighted_boxes_fusion
import tifffile as tif
import glob
import shutil
#from tqdm import tqdm
import numpy as np

print("import takes:", time.time() - t0)
t0 = time.time()

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=1280, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # data_loader = torch.utils.data.DataLoader(dataset,batch_size=6,pin_memory=True,shuffle=False,num_workers=6,drop_last=False)
    all_results = []

    for path, im, im0s, _, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
        pred = pred[0].unsqueeze(0)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        single_patch_info = []

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                lales_list = []
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        label = (line[0].cpu().item(),line[1],line[2],line[3],line[4],line[5].cpu().item())
                        lales_list.append(label)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return  all_results

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

class BboxDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img,
                 boxes,
                 mode='train'):
        self.img = img
        self.boxes = boxes
        self.mode = mode

        mean = (0.485, 0.456, 0.406)  # RGB
        std = (0.229, 0.224, 0.225)  # RGB

        self.albu_transforms = {
            'valid': A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean, std),
            ]),
        }

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx: int):

        shape = self.img.shape
        xmin, ymin, xmax, ymax = self.boxes[idx]
        xmin, ymin, xmax, ymax = round(xmin*shape[1]), round(ymin*shape[0]), round(xmax*shape[1]), round(ymax*shape[0])
        croped = self.img[ymin:ymax, xmin:xmax]
        #print(croped.shape)
        auged = self.albu_transforms['valid'](image=croped)
        image = torch.from_numpy(auged['image']).permute(2,0,1)

        return image

class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1):
        super().__init__()


        self.encoder = smp.Unet(
                encoder_name=base_model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=pretrained,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=num_classes,                      # model output channels (number of classes in your dataset)
            )


    def forward(self, input_data):
        x = input_data 
        logit = self.encoder(x)

        return logit

def bbox_inference(image, boxes):
    ds = BboxDataset(image, boxes)
    dl = torch.utils.data.DataLoader(
            ds, batch_size=32, num_workers=6, pin_memory=True, shuffle=False, drop_last=False
        )
    results = []
    for data in dl:
        with torch.no_grad():
            with autocast():
                seg_results = torch.sigmoid(model(data.to('cuda'))[:,1])
                #print(seg_results.shape)
                final_result = (seg_results>0.5).int().to('cpu').numpy().tolist()
        results += final_result
    return results

def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    return run(**vars(opt))


if __name__ == "__main__":
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)

    opt = parse_opt()
    img_list = []
    input_path = 'inputs'
    output_path = 'outputs'
    all_images = glob.glob(os.path.join(input_path, '*.*'))
    patch_size = 1000
    stride = 500

    if not os.path.isdir('./patched_cache'):
        os.makedirs('./patched_cache')
    else:
        #os.removedirs('./patched_cache/')
        shutil.rmtree('./patched_cache/', ignore_errors=True)
        os.makedirs('./patched_cache')
    print('Patching...')
    t0 = time.time()
    for img_path in all_images:
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img += img.min()
        img = img/(img.max()/255.)
        if len(img.shape) == 2:
            img = np.stack([img]*3, -1)
            #print('gray')
        img = img.astype('uint8')
        img_list.append(img)
        shape = img.shape
        if min(shape[:2]) > patch_size:

            x_count = shape[1]//stride
            y_count = shape[0]//stride

            for x_id in range(x_count):
                for y_id in range(y_count):

                    if x_id == x_count - 1:
                        xmin, xmax = shape[1]-patch_size, shape[1]
                        #print(xmin, xmax)
                    else:
                        xmin, xmax = x_id*stride, x_id*stride + patch_size
                    if y_id == y_count - 1:
                        ymin, ymax = shape[0]-patch_size, shape[0]
                        #print(ymin, ymax)
                    else:
                        ymin, ymax = y_id*stride, y_id*stride + patch_size

                    cv2.imwrite('{}&{}&{}.png'.format('patched_cache'+os.path.sep+img_path.split(os.path.sep)[-1].split('.')[0], xmin, ymin), img[ymin:ymax, xmin:xmax])

        else:
            cv2.imwrite('{}${}.png'.format('patched_cache'+os.path.sep+img_path.split(os.path.sep)[-1].split('.')[0], 0), img)
    
    print("patching takes:",time.time()-t0)
    t0 = time.time()

    print('YOLO Inferencing...')
    #yolov5
    all_results = main(opt)

    
    mean = (0.485, 0.456, 0.406) # RGB
    std = (0.229, 0.224, 0.225) # RGB
    
    albu_transforms = {
        'valid' : A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean, std),
        ]),
    }
    print("yolo boxes takes:",time.time()-t0)
    t0 = time.time()
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
    model = TimmSED(
    base_model_name="efficientnet-b0",
    pretrained=None,
    num_classes=2,
    in_channels=3)

    model.to('cuda')
    model.load_state_dict(torch.load('fold-0.bin'))
    model.eval()
    print('Seg Weithts Loaded.')

    

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)
    
    print('UNet Inferencing and Postprocessing...')
    for raw_image, img_path in zip(img_list, all_images):
        #raw_image = cv2.imread(img_path)
        shape = raw_image.shape
        if min(shape[:2]) > 1000:
            print('Patch Inferencing...')
            det_files =  glob.glob(os.path.join("patched_cache","detect","testa","labels","{}&*&*.txt".format(img_path.split(os.path.sep)[-1].split('.')[0])))
            box_set = []
            conf_set = []
            cls_set = []
            for file in det_files:
                boxes = []
                confs = []
                split = file.split('/')[-1].split('&')
                leftx, topy = int(split[-2]) / shape[1], int(split[-1][:-4]) / shape[0]

                with open(file, 'r') as f:
                    data = f.readlines()
                    f.close()
                for res in data:
                    cls, x, y, w, h, conf = res.split(' ')
                    x, y, w, h, conf = float(x), float(y), float(w), float(h), float(conf)

                    xmin, ymin, xmax, ymax = (x-0.5*w)*1000, (y-0.5*h)*1000, (x+0.5*w)*1000, (y+0.5*h)*1000

                    if min(xmin, ymin, xmax, ymax) > 5 and max(xmin, ymin, xmax, ymax) < 995:
                        #print((xmin, ymin, xmax, ymax), min(xmin, ymin, xmax, ymax), max(xmin, ymin, xmax, ymax))
                        xmin, ymin, xmax, ymax = xmin/shape[1]+leftx, ymin/shape[0]+topy, xmax/shape[1]+leftx, ymax/shape[0]+topy
                        boxes.append([xmin, ymin, xmax, ymax])
                        confs.append(conf)
                        #print(1)
                        #continue

                box_set.append(boxes)
                conf_set.append(confs)
                cls_set.append([0]*len(confs))


            # boxes, confs, _ = weighted_boxes_fusion(box_set, conf_set, cls_set)
        else:
            print('Direct Inferencing...')
            file = glob.glob('patched_cache/detect/testa/labels/{}$0.txt'.format(img_path.split(os.path.sep)[-1].split('.')[0]))
            if os.path.isfile(file):
                #print('No Patch')
                boxes = []
                confs = []
                with open(file, 'r') as f:
                    data = f.readlines()
                    f.close()
                for res in data:
                    cls, x, y, w, h, conf = res.split(' ')
                    x, y, w, h, conf = float(x), float(y), float(w), float(h), float(conf)
                    xmin, ymin, xmax, ymax = (x-0.5*w), (y-0.5*h), (x+0.5*w), (y+0.5*h)
                    if min((xmax-xmin)*shape[1], (ymax-ymin)*shape[0]) > 2:
                        boxes.append([xmin, ymin, xmax, ymax])
                        confs.append(conf)

        base = np.zeros((shape[0], shape[1]), dtype='uint16')
        if boxes is not None:
            if len(boxes) > 65000:
                base = np.zeros((shape[0], shape[1]), dtype='uint32')
            masks = bbox_inference(raw_image, boxes)

            cell_count = 1
            for box, mask in zip(boxes, masks):
                xmin, ymin, xmax, ymax = box
                xmin, ymin, xmax, ymax = round(xmin*shape[1]), round(ymin*shape[0]), round(xmax*shape[1]), round(ymax*shape[0])
                mask = cv2.resize(np.array(mask), (xmax-xmin, ymax-ymin), interpolation=cv2.INTER_NEAREST).astype(bool)

                base[ymin:ymax, xmin:xmax][mask] = cell_count
                cell_count+=1

        tif.imwrite(os.path.join(output_path, '{}_label.tiff'.format(img_path.split(os.path.sep)[-1].split('.')[0])), base, compression='zlib')
    print("seg  takes:",time.time()-t0)
    t0 = time.time()