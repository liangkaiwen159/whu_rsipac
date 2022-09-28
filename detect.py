import argparse
from asyncio import WriteTransport
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dataset.pro_dataset import LoadImages
from utils.plot import Annotator, colors
from utils.crop_img_for_detect import crop_img, cat_img
from utils.torch_utils import select_device, time_sync
from utils.write_json import Write_json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import check_img_size, check_suffix, colorstr, increment_path, nms_for_self, non_max_suppression, save_one_box, scale_coords, xyxy2xywh


@torch.no_grad()
def run(
        weights=ROOT / 'last.pt',  # model.pt path(s)
        source='D:\\chusai_crop\\test\\images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.01,  # confidence threshold
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
        json_path=None):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    (save_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (save_dir / 'imgs').mkdir(parents=True, exist_ok=True)
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        device = torch.device('cuda:1')
        model = torch.load(w, map_location=device)['model']
        # model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    write_json = Write_json(json_path)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        # img C H W,  R G B
        ori_shape = img.shape  # (3,3449,5903)
        ori_img = np.ascontiguousarray(img[::-1].transpose(1, 2, 0))  # HWC BRG
        t1 = time_sync()
        _img_list, XY_sequence = crop_img(img.transpose(1, 2, 0), gap=160)  #rgb
        __img_list = {k: torch.from_numpy(np.ascontiguousarray(img)).to(device) for (k, img) in _img_list.items()}
        mask_img_list = {k: None for k, img in _img_list.items()}
        img_list = {}
        over_all_label = []
        over_all_xyxy = []
        over_all_c = []
        over_all_to_nms = []
        for k, img in __img_list.items():
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            img_list[k] = img
        t2 = time_sync()
        dt[0] += t2 - t1
        for k, img in img_list.items():
            # Inference
            if pt:
                model.eval()
                pred, mask = model(img)
                # img = (img[0] * 255).to('cpu').numpy().transpose((1, 2, 0)).astype(np.int)
                # cv2.imwrite('111.jpg', img)
                # cv2.imencode('.jpg', mask_img)[1].tofile('test_output.jpg')
                pred = pred[0]
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, s, im0, frame = path, '', img[0].to('cpu').numpy().transpose(
                    (1, 2, 0)).copy(), getattr(dataset, 'frame', 0)  #im0 rgb
                p = Path(p)  # to Path
                save_path = str(save_dir / 'imgs' / p.name.replace('tif', 'png'))  # img.jpg
                save_mask_path = str(save_dir / 'masks' / p.name.replace('tif', 'png'))
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}'
                                                                )  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(np.ascontiguousarray(img.to('cpu')[0].numpy().transpose(
                    (1, 2, 0)) * 255).astype(np.uint8),
                                      line_width=line_thickness,
                                      example=str(names))  #bgr 扔进画图
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape[2:]).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            j, i = map(int, k.split('_'))
                            _over_all_xyxy = np.concatenate((XY_sequence[j][i][0:2], XY_sequence[j][i][0:2])) + xyxy
                            over_all_c.append(c)
                            over_all_xyxy.append(_over_all_xyxy)
                            over_all_label.append(label)
                            over_all_to_nms.append(
                                np.array([
                                    *[int(x.cpu().numpy().astype(np.int)) for x in _over_all_xyxy], c,
                                    float(conf.cpu().numpy())
                                ]))

            img_list[k] = annotator.result().transpose((2, 0, 1))  #rgb
            mask = mask[0, ...].sigmoid()
            fill_zero = mask < 0.5
            fill_255 = mask >= 0.5
            _mask = mask.clone().int()
            _mask[fill_zero] = 0
            _mask[fill_255] = 255
            mask_img = _mask.to('cpu').numpy()
            mask_img = mask_img.astype(np.uint8)
            mask_img_list[k] = mask_img
            # Print time (inference-only)
        print(f'{ori_shape[1:]}Done. ({t3 - t2:.3f}s)')
        # Stream results
        ori_img_annotator = Annotator(ori_img, line_width=line_thickness, example=str(names))
        nmsed_boxes = nms_for_self(over_all_to_nms, iou_thres=0.5)
        if nmsed_boxes is not None:
            for nmsed_box in nmsed_boxes:
                c = int(nmsed_box[5])
                conf = nmsed_box[4]
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                ori_img_annotator.box_label(nmsed_box[0:4], label, color=colors(c, True))
            write_json.write_info(nmsed_boxes, Path(p).name)
        # for index in range(len(over_all_xyxy)):  # 获取所有标注 绘制
        #     c = over_all_c[index]
        #     ori_img_annotator.box_label(over_all_xyxy[index], over_all_label[index], color=colors(c, True))
        # im0 = cat_img(ori_shape, img_list, XY_sequence)  #绘制好的图片直接拼
        im0 = ori_img_annotator.result()  # 获取所有标注 绘制的图片
        msk0 = cat_img(ori_shape, mask_img_list, XY_sequence, mask=True)
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                cv2.imwrite(save_mask_path, msk0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, img.shape)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    write_json.write_json_file(save_dir)  # for whu save json


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default=ROOT / 'test_weights' / 'last-127.pt',
                        help='model path(s)')
    parser.add_argument(
        '--source',
        type=str,
        # default='/mnt/users/datasets/chusai_crop/test/images/',
        default='/home/xcy/dataset/chusai_release/test/images/',
        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
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
    parser.add_argument('--half', default=True, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--json_path', default='/home/xcy/dataset/chusai_release/test/test_write.json')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
