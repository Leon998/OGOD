import argparse
import os
import sys
from pathlib import Path
import time
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.myutils import *


@torch.no_grad()
def run(
        weights=ROOT / 'weights/yolov5m.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
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
        ground_truth='bottle',
):
    name = ground_truth
    source = str(source)
    vid_name = get_vid_name(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = Path(project) / name
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    t0 = time.time()
    # stream_log：记录视频流每一帧累积信息的
    # class_score_lod: 80×n维的列表，表示80个类别的得分记录
    stream_log = []
    class_score_log = np.zeros((80, 1))
    new_frame = np.zeros(80)
    frame_idx = 0
    trigger_flag = [False, "None"]

    for path, im, im0s, vid_cap, s in dataset:
        # 分数记录
        if frame_idx >= 1:
            class_score_log = np.column_stack((class_score_log, new_frame))
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # ===================================至此，推理过程已经结束================================= #
        # 记录每张图片所有目标结果的列表
        frame_log = []
        # 记录图片里每个目标得分的列表
        score_list = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            im1 = im0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    coffset = get_centraloffset(xyxy, gn, normalize=True)  # 获得每个目标的中心偏移量coffset
                    # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    thres = Box_thres[int(cls)]
                    box_rate = get_box_thres_rate(xywh, thres)  # 获取阈值比
                    box_size = get_box_size((xywh))  # 只获得框大小
                    score = box_rate / coffset  # 计分score
                    # 记录当前这个种类的特征
                    frame_log.append(
                        {"cls": names[int(cls)], "cls_num": int(cls), "conf": conf, "xyxy": xyxy, "xywh": xywh, "coffset": coffset,
                         "box_rate": box_rate, "box_size": box_size, "score": score})
                    score_list.append(score)  # score_list每帧都更新
                    # 每次直接对应int(cls)的那个class_score_log进行append操作
                    if score >= class_score_log[int(cls), :][frame_idx]:
                        class_score_log[int(cls), :][frame_idx] = score
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # =====================================单个object检测结束================================= #
                # ==================================TargetChoosing===================================== #
                # Not voting
                # target_idx = score_list.index(max(score_list))  # 这一步可以修改成voting之类的方式

                # Voting
                if frame_idx < step:
                    target_idx = score_list.index(max(score_list))  # 注意这里是因为之前保持了score_list和frame_log的目标索引是一样的
                else:
                    target_idx = vote_score(frame_log, class_score_log, step=step)  # 连续step帧累积投票

                # 归一法计算概率
                prob = norm_prob(score_list)
                target = frame_log[target_idx]
                # 这里是判断是否预测对了target
                save_eval_seq(str(save_dir / str('eval_seq' + vid_name + '.txt')), target["cls"], ground_truth, prob)
                target_xyxy = target["xyxy"]
                im1 = info_on_img(im0, gn, zoom=[0.45, 0.9], label="Box_x_loc: " + str(round(target["xywh"][0], 3)))
                im1 = info_on_img(im1, gn, zoom=[0.75, 0.9], label="Box_y_loc: " + str(round(target["xywh"][1], 3)))
                im1 = info_on_img(im1, gn, zoom=[0.45, 0.95], label="Box_size: " + str(round(target["box_size"], 3)))
                im1 = info_on_img(im1, gn, zoom=[0.75, 0.95], label="Box_rate: " + str(round(target["box_rate"], 3)))
                im1 = info_on_img(im1, gn, zoom=[0.75, 0.85], label="Score: " + str(round(target["score"].item(), 3)))
                im1 = plot_target_box(target_xyxy, im1, color=colors(0, True), line_thickness=2)
                trigger_flag = check_trigger(target["box_rate"], target["xywh"], target["cls"], trigger_flag)
                if trigger_flag[0]:
                    # 判断是否在grasping
                    im1 = text_on_img(im1, gn, zoom=[0.05, 0.95], label="Grasping " + trigger_flag[1])
                    eval_instance_file = str(save_dir / str('eval_instance' + vid_name + '.txt'))
                    if not os.path.exists(eval_instance_file):
                        save_eval_instance(eval_instance_file, target["cls"], ground_truth)
                else:
                    im1 = text_on_img(im1, gn, zoom=[0.05, 0.95], label="Targeting: " + target["cls"])
                stream_log.append(frame_log)

            else:
                # 如果没有预测出目标
                save_eval_seq(str(save_dir / str('eval_seq' + vid_name + '.txt')), "None", ground_truth, 0)
                trigger_flag = check_trigger_null(trigger_flag)
                if trigger_flag[0]:
                    im1 = text_on_img(im1, gn, zoom=[0.05, 0.95], label="Grasping " + trigger_flag[1])
                else:
                    im1 = text_on_img(im1, gn, zoom=[0.05, 0.95], label="No Target")
                stream_log.append(["None"])

            im1 = text_on_img(im1, gn, zoom=[0.05, 0.1], color=[0, 0, 255], label="Frame " + str(frame_idx))
            # 记录当前帧Trigger_flag的状态
            im1 = text_on_img(im1, gn, zoom=[0.05, 0.2], color=[0, 0, 255],
                                      label="Flag on" if trigger_flag[0] else "Flag off")

            # Stream results
            # im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im1)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im1)
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
                            fps, w, h = 30, im1.shape[1], im1.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im1)
        frame_idx += 1
        # 至此结束当前帧
        # Print time (inference-only)
        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    # 打印预测的总时间
    print(frame_idx)
    print(f'Done. ({time.time() - t0:.3f}s)')
    # 把所有class都保存到file
    # save_score_to_file(save_dir, class_score_log)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5l.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
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
    parser.add_argument('--project', default=ROOT / 'evals/clips', help='save results to project/name')  # Adjustable
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--ground-truth', type=str, default='bottle', help='ground truth label of target')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
