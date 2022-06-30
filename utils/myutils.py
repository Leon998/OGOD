import torch
import cv2
import numpy as np


# Box_thres = [0.8 for idx in range(80)]
# Box_thres[39] = 0.8  # bottle
# Box_thres[41] = 0.7  # cup
# Box_thres[44] = 0.6  # spoon
# Box_thres[49] = 0.5  # orange
# Box_thres[64] = 0.5  # mouse
# Box_thres[76] = 0.5  # scissors

Grasp_type = ['Grasping' for idx in range(80)]
Grasp_type[39] = 'Medium wrap: '  # bottle
Grasp_type[41] = 'Medium wrap: '  # cup
Grasp_type[44] = 'Tip pinch: '  # spoon
Grasp_type[49] = 'Power sphere: '  # orange
Grasp_type[64] = 'Tripod: '  # mouse
Grasp_type[76] = 'Tripod: '  # scissors

def get_centraloffset(xyxy, gn, normalize=False):
    """
    计算的是边界框中心与整个画面中心的L2距离（默认为像素距离，normalize后为归一化的结果），其中
    归一化的方式为L2像素距离除以一半对角线像素长度，越接近0表示离中心越近，约接近1表示离中心越远
    """
    ic = torch.tensor([gn[0] / 2, gn[1] / 2])  # image_centre
    i2c = (ic[0].pow(2) + ic[1].pow(2)).sqrt()  # half_diagonal
    bc = torch.tensor([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])  # box_centre
    cft = (ic[0] - bc[0]).pow(2) + (ic[1] - bc[1]).pow(2)  # centraloffset
    if normalize:
        cft = cft.sqrt() / i2c
    else:
        cft = cft.sqrt()
    return cft

def get_centeroffset_2version(xywh, normalize=False):
    """
    直接用框中心与(0.5,0.5)的距离表示
    """
    ic = torch.tensor([0.5, 0.5])
    i2c = (ic[0].pow(2) + ic[1].pow(2)).sqrt()
    cft = (xywh[0] - ic[0]).pow(2) + (xywh[1] - ic[1]).pow(2)
    if normalize:
        cft = cft.sqrt() / i2c
    else:
        cft = cft.sqrt()
    return cft

def get_box_thres_rate(xywh, thres):
    """
    计算边界框大小与标准化阈值的比例
    """
    rate = max(xywh[2], xywh[3]) / thres
    return rate

def get_box_size(xywh):
    """
    计算边界框大小
    """
    size = max(xywh[2], xywh[3])
    return size

def vote_score(frame_log, class_score_log, step=3):
    class_score_seq = np.zeros((len(frame_log), step))
    # class_score_seq是累积score矩阵，用来记录当前帧的排布，后续会联系上下文累积成sequence，但是索引保持与frame_log一致
    for i in range(len(frame_log)):
        cls = frame_log[i]["cls_num"]  # 提取出当前帧每个目标的类别索引（类别cls）
        class_score_seq[i] = class_score_log[cls, -step:]  # 累积score矩阵直接从类别score记录矩阵中找对应类别cls的step步score情况
    score_sum = class_score_seq.sum(axis=1)  # 直接加和
    # target_idx = score_sum.argmax()  # 找出最大的那个索引（弃用）（注意这里如果有多个相同类别的目标的话，只找得出数字更小的那个索引）
    target_idx = np.argwhere(score_sum == score_sum.max())
    target_idx = np.array(target_idx).reshape(-1).tolist()  # 经过这两步，就算有相同类别的目标，它们的索引也会被找出来
    coffset = []
    for i in target_idx:
        coffset.append(frame_log[i]["coffset"])
    idx = coffset.index(min(coffset))  # 找出相同类别的目标中，里中心最近的那一个，这里的idx是索引的索引
    target_idx = target_idx[idx]  # 获得真实索引（对于frame_log而言的目标索引）
    return target_idx

def check_trigger(box_rate, xywh, cls, trigger_flag, x=1):
    # box_bool表示三个条件：框阈值比，框x位置，框y位置。x位置严格，y位置可以相对宽松一点
    box_bool = box_rate > x and 0.33 < xywh[0] < 0.66 and 0.33 < xywh[1] < 0.66
    if trigger_flag[0]:  # 上一时刻是抓的状态
        if cls == trigger_flag[1]:  # 如果还是要抓的目标
            if box_bool:  # 还在接近
                trigger_flag[0] = True
            else:  # 已经离开了
                trigger_flag[0] = False
                trigger_flag[1] = "None"
        else:  # 检测到了其他东西
            trigger_flag[0] = True
    else:  # 上一时刻不是抓取状态
        if box_bool:  # 新的目标进入抓取状态
            trigger_flag[0] = True
            trigger_flag[1] = cls
        else:  # 还在瞄准
            trigger_flag[0] = False
            trigger_flag[1] = "None"
    return trigger_flag

def check_trigger_null(trigger_flag):
    if trigger_flag[0]:  # 上一时刻是抓的状态
        trigger_flag[0] = True
    else:  # 上一时刻不是抓取状态
        trigger_flag[0] = False
    return trigger_flag


def plot_target_box(x, im, color=(128, 128, 128), label=None, line_thickness=2):
    """一般会用在detect.py中在nms之后变量每一个预测框，再将每个预测框画在原图上
    使用opencv在原图im上画一个bounding box
    :params x: 预测得到的bounding box  [x1 y1 x2 y2]
    :params im: 原图 要将bounding box画在这个图上  array
    :params color: bounding box线的颜色
    :params labels: 标签上的框框信息  类别 + score
    :params line_thickness: bounding box的线宽，-1表示框框为实心
    """
    # check im内存是否连续
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # 这里在画高亮部分
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
    # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
    blk = np.zeros(im.shape, np.uint8)
    cv2.rectangle(blk, c1, c2, color, -1)  # 注意在 blk的基础上进行绘制；
    img = cv2.addWeighted(im, 1.0, blk, 0.5, 1)
    return img

def text_on_img(im, gn, zoom, color=[0,0,255], label=None, line_thickness=2):
    # Used for demo words
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    scale = int(gn[1]) / 250 + 0.3
    tf = int(scale)  # label字体的线宽 font thickness
    d1 = (int(gn[0] * zoom[0]), int(gn[1] * zoom[1]))
    img = cv2.putText(im, label, (d1[0], d1[1]), 0, scale, color, thickness=tf + 3, lineType=cv2.LINE_AA)
    return img

def info_on_img(im, gn, zoom, color=[0,0,255], label=None, line_thickness=2):
    # Used for debugging words
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    scale = int(gn[1]) / 666 + 0.2
    tf = int(scale)  # label字体的线宽 font thickness
    d1 = (int(gn[0] * zoom[0]), int(gn[1] * zoom[1]))
    img = cv2.putText(im, label, (d1[0], d1[1]), 0, scale, color, thickness=tf + 1, lineType=cv2.LINE_AA)
    return img

def norm_prob(score_list):
    prob_list = []
    for i in range(len(score_list)):
        prob_list.append(round(score_list[i].item() / (sum(score_list)).item(), 4))
    return prob_list

def equal_len(seq, length=75):
    new_seq = [0] * length
    if len(seq)>=length:
        new_seq[:] = seq[-length:]
    else:
        new_seq[length-len(seq):] = seq[:]
    return new_seq

def list_sum(a,b):
    c = list(map(lambda x:x[0]+x[1],zip(a,b)))
    return c

def list_mean(a,times):
    c = [x / times for x in a]
    return c

def seq_accuracy(seq):
    hit = 0
    for i in seq:
        if i >0:
            hit += 1
    accuracy = hit / len(seq)
    return accuracy

def save_file_discrete(path, var):
    filename = open(path, 'a')
    filename.write(str(var) + '\n')
    filename.close()

def save_file_continue(path, list):
    filename = open(path, 'a')
    for i in list:
        filename.write(str(i) + '\n')
    filename.close()

def save_eval_seq(eval_seq, target, cls, prob):
    """
    在eval_seq类脚本中，用于保存序列中准确预测情况的函数
    """
    if target == cls:  # 预测正确
        eval_seq.append(prob)
    elif target == "None":  # 啥都没预测出来
        eval_seq.append(0)
    else:  # 预测错误
        eval_seq.append(0-prob)
    return eval_seq

def save_eval_instance(eval_inst, target, cls):
    """
    在eval_instance类脚本中，用于保存整体预测情况的函数
    """
    if target == cls:  # 预测正确
        eval_inst = 1
    return eval_inst

def save_eval_grasp(eval_grasp, trigger_flag, cls):
    """
    在eval_grasp类脚本中，用于保存预测抓取状态情况的函数
    """
    if trigger_flag[0] and trigger_flag[1] == cls:
        eval_grasp.append(1)
    else:
        eval_grasp.append(0)
    return eval_grasp

def check_gp(eval_grasp):
    flag = 1
    for i in eval_grasp:
        if i != 1:
            flag = 0
            break
    eval_grasp.append(flag)
    return eval_grasp

def save_score(path, cls, class_score_log):
    """
    保存单个class的score到file
    """
    filename = open(path, 'w')
    for value in class_score_log[cls, :]:
        value = value.item()
        filename.write(str(value) + '\n')
    filename.close()

def save_score_to_file(save_dir, class_score_log):
    """
    把多个class汇集到一起保存到file
    """
    save_score(str(save_dir / 'bottle_score.txt'), 39, class_score_log)
    save_score(str(save_dir / 'cup_score.txt'), 41, class_score_log)
    save_score(str(save_dir / 'spoon_score.txt'), 44, class_score_log)
    save_score(str(save_dir / 'orange_score.txt'), 49, class_score_log)
    save_score(str(save_dir / 'mouse_score.txt'), 64, class_score_log)
    save_score(str(save_dir / 'scissors_score.txt'), 76, class_score_log)

def get_vid_name(source):
    name = source[-7:-4]
    return name

def flag2target(name):
    if name == 'bottle':
        target_num = 39
    elif name == 'cup':
        target_num = 41
    elif name == 'spoon':
        target_num = 44
    elif name == 'orange':
        target_num = 49
    elif name == 'mouse':
        target_num = 64
    else:
        target_num = 76
    return target_num
