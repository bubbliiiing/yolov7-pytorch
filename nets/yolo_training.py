import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0):
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        #-----------------------------------------------------------#
        self.anchors        = [anchors[mask] for mask in anchors_mask]
        self.num_classes    = num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.balance        = [0.4, 1.0, 4]
        self.stride         = [32, 16, 8]
        
        self.box_ratio      = 0.05
        self.obj_ratio      = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio      = 0.5 * (num_classes / 80)
        self.threshold      = 4

        self.cp, self.cn                    = smooth_BCE(eps=label_smoothing)  
        self.BCEcls, self.BCEobj, self.gr   = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(), 1

    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        box2 = box2.T

        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1  = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2  = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union   = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union

        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU
    
    def __call__(self, predictions, targets, imgs): 
        #-------------------------------------------#
        #   对输入进来的预测结果进行reshape
        #   bs, 255, 20, 20 => bs, 3, 20, 20, 85
        #   bs, 255, 40, 40 => bs, 3, 40, 40, 85
        #   bs, 255, 80, 80 => bs, 3, 80, 80, 85
        #-------------------------------------------#
        for i in range(len(predictions)):
            bs, _, h, w = predictions[i].size()
            predictions[i] = predictions[i].view(bs, len(self.anchors_mask[i]), -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
            
        #-------------------------------------------#
        #   获得工作的设备
        #-------------------------------------------#
        device              = targets.device
        #-------------------------------------------#
        #   初始化三个部分的损失
        #-------------------------------------------#
        cls_loss, box_loss, obj_loss    = torch.zeros(1, device = device), torch.zeros(1, device = device), torch.zeros(1, device = device)
        
        #-------------------------------------------#
        #   进行正样本的匹配
        #-------------------------------------------#
        bs, as_, gjs, gis, targets, anchors = self.build_targets(predictions, targets, imgs)
        #-------------------------------------------#
        #   计算获得对应特征层的高宽
        #-------------------------------------------#
        feature_map_sizes = [torch.tensor(prediction.shape, device=device)[[3, 2, 3, 2]].type_as(prediction) for prediction in predictions] 
    
        #-------------------------------------------#
        #   计算损失，对三个特征层各自进行处理
        #-------------------------------------------#
        for i, prediction in enumerate(predictions): 
            #-------------------------------------------#
            #   image, anchor, gridy, gridx
            #-------------------------------------------#
            b, a, gj, gi    = bs[i], as_[i], gjs[i], gis[i]
            tobj            = torch.zeros_like(prediction[..., 0], device=device)  # target obj

            #-------------------------------------------#
            #   获得目标数量，如果目标大于0
            #   则开始计算种类损失和回归损失
            #-------------------------------------------#
            n = b.shape[0]
            if n:
                prediction_pos = prediction[b, a, gj, gi]  # prediction subset corresponding to targets

                #-------------------------------------------#
                #   计算匹配上的正样本的回归损失
                #-------------------------------------------#
                #-------------------------------------------#
                #   grid 获得正样本的x、y轴坐标
                #-------------------------------------------#
                grid    = torch.stack([gi, gj], dim=1)
                #-------------------------------------------#
                #   进行解码，获得预测结果
                #-------------------------------------------#
                xy      = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                wh      = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                box     = torch.cat((xy, wh), 1)
                #-------------------------------------------#
                #   对真实框进行处理，映射到特征层上
                #-------------------------------------------#
                selected_tbox           = targets[i][:, 2:6] * feature_map_sizes[i]
                selected_tbox[:, :2]    -= grid.type_as(prediction)
                #-------------------------------------------#
                #   计算预测框和真实框的回归损失
                #-------------------------------------------#
                iou                     = self.bbox_iou(box.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                box_loss                += (1.0 - iou).mean()
                #-------------------------------------------#
                #   根据预测结果的iou获得置信度损失的gt
                #-------------------------------------------#
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                #-------------------------------------------#
                #   计算匹配上的正样本的分类损失
                #-------------------------------------------#
                selected_tcls               = targets[i][:, 1].long()
                t                           = torch.full_like(prediction_pos[:, 5:], self.cn, device=device)  # targets
                t[range(n), selected_tcls]  = self.cp
                cls_loss                    += self.BCEcls(prediction_pos[:, 5:], t)  # BCE

            #-------------------------------------------#
            #   计算目标是否存在的置信度损失
            #   并且乘上每个特征层的比例
            #-------------------------------------------#
            obj_loss += self.BCEobj(prediction[..., 4], tobj) * self.balance[i]  # obj loss
            
        #-------------------------------------------#
        #   将各个部分的损失乘上比例
        #   全加起来后，乘上batch_size
        #-------------------------------------------#
        box_loss    *= self.box_ratio
        obj_loss    *= self.obj_ratio
        cls_loss    *= self.cls_ratio
        bs          = tobj.shape[0]
        
        loss    = box_loss + obj_loss + cls_loss
        return loss
        
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def build_targets(self, predictions, targets, imgs):
        #-------------------------------------------#
        #   匹配正样本
        #-------------------------------------------#
        indices, anch       = self.find_3_positive(predictions, targets)

        matching_bs         = [[] for _ in predictions]
        matching_as         = [[] for _ in predictions]
        matching_gjs        = [[] for _ in predictions]
        matching_gis        = [[] for _ in predictions]
        matching_targets    = [[] for _ in predictions]
        matching_anchs      = [[] for _ in predictions]
        
        #-------------------------------------------#
        #   一共三层
        #-------------------------------------------#
        num_layer = len(predictions)
        #-------------------------------------------#
        #   对batch_size进行循环，进行OTA匹配
        #   在batch_size循环中对layer进行循环
        #-------------------------------------------#
        for batch_idx in range(predictions[0].shape[0]):
            #-------------------------------------------#
            #   先判断匹配上的真实框哪些属于该图片
            #-------------------------------------------#
            b_idx       = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            #-------------------------------------------#
            #   如果没有真实框属于该图片则continue
            #-------------------------------------------#
            if this_target.shape[0] == 0:
                continue
            
            #-------------------------------------------#
            #   真实框的坐标进行缩放
            #-------------------------------------------#
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            #-------------------------------------------#
            #   从中心宽高到左上角右下角
            #-------------------------------------------#
            txyxy = self.xywh2xyxy(txywh)

            pxyxys      = []
            p_cls       = []
            p_obj       = []
            from_which_layer = []
            all_b       = []
            all_a       = []
            all_gj      = []
            all_gi      = []
            all_anch    = []
            
            #-------------------------------------------#
            #   对三个layer进行循环
            #-------------------------------------------#
            for i, prediction in enumerate(predictions):
                #-------------------------------------------#
                #   b代表第几张图片 a代表第几个先验框
                #   gj代表y轴，gi代表x轴
                #-------------------------------------------#
                b, a, gj, gi    = indices[i]
                idx             = (b == batch_idx)
                b, a, gj, gi    = b[idx], a[idx], gj[idx], gi[idx]       
                       
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                #-------------------------------------------#
                #   取出这个真实框对应的预测结果
                #-------------------------------------------#
                fg_pred = prediction[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                #-------------------------------------------#
                #   获得网格后，进行解码
                #-------------------------------------------#
                grid    = torch.stack([gi, gj], dim=1).type_as(fg_pred)
                pxy     = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]
                pwh     = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]
                pxywh   = torch.cat([pxy, pwh], dim=-1)
                pxyxy   = self.xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            #-------------------------------------------#
            #   判断是否存在对应的预测框，不存在则跳过
            #-------------------------------------------#
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            
            #-------------------------------------------#
            #   进行堆叠
            #-------------------------------------------#
            p_obj       = torch.cat(p_obj, dim=0)
            p_cls       = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b       = torch.cat(all_b, dim=0)
            all_a       = torch.cat(all_a, dim=0)
            all_gj      = torch.cat(all_gj, dim=0)
            all_gi      = torch.cat(all_gi, dim=0)
            all_anch    = torch.cat(all_anch, dim=0)
        
            #-------------------------------------------------------------#
            #   计算当前图片中，真实框与预测框的重合程度
            #   iou的范围为0-1，取-log后为0~inf
            #   重合程度越大，取-log后越小
            #   因此，真实框与预测框重合度越大，pair_wise_iou_loss越小
            #-------------------------------------------------------------#
            pair_wise_iou       = self.box_iou(txyxy, pxyxys)
            pair_wise_iou_loss  = -torch.log(pair_wise_iou + 1e-8)

            #-------------------------------------------#
            #   最多二十个预测框与真实框的重合程度
            #   然后求和，找到每个真实框对应几个预测框
            #-------------------------------------------#
            top_k, _    = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks  = torch.clamp(top_k.sum(1).int(), min=1)

            #-------------------------------------------#
            #   gt_cls_per_image    种类的真实信息
            #-------------------------------------------#
            gt_cls_per_image = F.one_hot(this_target[:, 1].to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            
            #-------------------------------------------#
            #   cls_preds_  种类置信度的预测信息
            #               cls_preds_越接近于1，y越接近于1
            #               y / (1 - y)越接近于无穷大
            #               也就是种类置信度预测的越准
            #               pair_wise_cls_loss越小
            #-------------------------------------------#
            num_gt              = this_target.shape[0]
            cls_preds_          = p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            y                   = cls_preds_.sqrt_()
            pair_wise_cls_loss  = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_cls_per_image, reduction="none").sum(-1)
            del cls_preds_
        
            #-------------------------------------------#
            #   求cost的总和
            #-------------------------------------------#
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            #-------------------------------------------#
            #   求cost最小的k个预测框
            #-------------------------------------------#
            matching_matrix = torch.zeros_like(cost)
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks

            #-------------------------------------------#
            #   如果一个预测框对应多个真实框
            #   只使用这个预测框最对应的真实框
            #-------------------------------------------#
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1]          *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            #-------------------------------------------#
            #   取出符合条件的框
            #-------------------------------------------#
            from_which_layer    = from_which_layer.to(fg_mask_inboxes.device)[fg_mask_inboxes]
            all_b               = all_b[fg_mask_inboxes]
            all_a               = all_a[fg_mask_inboxes]
            all_gj              = all_gj[fg_mask_inboxes]
            all_gi              = all_gi[fg_mask_inboxes]
            all_anch            = all_anch[fg_mask_inboxes]
            this_target         = this_target[matched_gt_inds]
        
            for i in range(num_layer):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(num_layer):
            matching_bs[i]      = torch.cat(matching_bs[i], dim=0) if len(matching_bs[i]) != 0 else torch.Tensor(matching_bs[i])
            matching_as[i]      = torch.cat(matching_as[i], dim=0) if len(matching_as[i]) != 0 else torch.Tensor(matching_as[i])
            matching_gjs[i]     = torch.cat(matching_gjs[i], dim=0) if len(matching_gjs[i]) != 0 else torch.Tensor(matching_gjs[i])
            matching_gis[i]     = torch.cat(matching_gis[i], dim=0) if len(matching_gis[i]) != 0 else torch.Tensor(matching_gis[i])
            matching_targets[i] = torch.cat(matching_targets[i], dim=0) if len(matching_targets[i]) != 0 else torch.Tensor(matching_targets[i])
            matching_anchs[i]   = torch.cat(matching_anchs[i], dim=0) if len(matching_anchs[i]) != 0 else torch.Tensor(matching_anchs[i])

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, predictions, targets):
        #------------------------------------#
        #   获得每个特征层先验框的数量
        #   与真实框的数量
        #------------------------------------#
        num_anchor, num_gt  = len(self.anchors_mask[0]), targets.shape[0] 
        #------------------------------------#
        #   创建空列表存放indices和anchors
        #------------------------------------#
        indices, anchors    = [], []
        #------------------------------------#
        #   创建7个1
        #   序号0,1为1
        #   序号2:6为特征层的高宽
        #   序号6为1
        #------------------------------------#
        gain    = torch.ones(7, device=targets.device)
        #------------------------------------#
        #   ai      [num_anchor, num_gt]
        #   targets [num_gt, 6] => [num_anchor, num_gt, 7]
        #------------------------------------#
        ai      = torch.arange(num_anchor, device=targets.device).float().view(num_anchor, 1).repeat(1, num_gt)
        targets = torch.cat((targets.repeat(num_anchor, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g   = 0.5 # offsets
        off = torch.tensor([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], device=targets.device).float() * g 

        for i in range(len(predictions)):
            #----------------------------------------------------#
            #   将先验框除以stride，获得相对于特征层的先验框。
            #   anchors_i [num_anchor, 2]
            #----------------------------------------------------#
            anchors_i = torch.from_numpy(self.anchors[i] / self.stride[i]).type_as(predictions[i])
            anchors_i, shape = torch.from_numpy(self.anchors[i] / self.stride[i]).type_as(predictions[i]), predictions[i].shape
            #-------------------------------------------#
            #   计算获得对应特征层的高宽
            #-------------------------------------------#
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
            
            #-------------------------------------------#
            #   将真实框乘上gain，
            #   其实就是将真实框映射到特征层上
            #-------------------------------------------#
            t = targets * gain
            if num_gt:
                #-------------------------------------------#
                #   计算真实框与先验框高宽的比值
                #   然后根据比值大小进行判断，
                #   判断结果用于取出，获得所有先验框对应的真实框
                #   r   [num_anchor, num_gt, 2]
                #   t   [num_anchor, num_gt, 7] => [num_matched_anchor, 7]
                #-------------------------------------------#
                r = t[:, :, 4:6] / anchors_i[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < self.threshold
                t = t[j]  # filter
                
                #-------------------------------------------#
                #   gxy 获得所有先验框对应的真实框的x轴y轴坐标
                #   gxi 取相对于该特征层的右小角的坐标
                #-------------------------------------------#
                gxy     = t[:, 2:4] # grid xy
                gxi     = gain[[2, 3]] - gxy # inverse
                j, k    = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m    = ((gxi % 1. < g) & (gxi > 1.)).T
                j       = torch.stack((torch.ones_like(j), j, k, l, m))
                
                #-------------------------------------------#
                #   t   重复5次，使用满足条件的j进行框的提取
                #   j   一共五行，代表当前特征点在五个
                #       [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]
                #       方向是否存在
                #-------------------------------------------#
                t       = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            #-------------------------------------------#
            #   b   代表属于第几个图片
            #   gxy 代表该真实框所处的x、y中心坐标
            #   gwh 代表该真实框的wh坐标
            #   gij 代表真实框所属的特征点坐标
            #-------------------------------------------#
            b, c    = t[:, :2].long().T  # image, class
            gxy     = t[:, 2:4]  # grid xy
            gwh     = t[:, 4:6]  # grid wh
            gij     = (gxy - offsets).long()
            gi, gj  = gij.T  # grid xy indices

            #-------------------------------------------#
            #   gj、gi不能超出特征层范围
            #   a代表属于该特征点的第几个先验框
            #-------------------------------------------#
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            anchors.append(anchors_i[a])  # anchors

        return indices, anchors

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
