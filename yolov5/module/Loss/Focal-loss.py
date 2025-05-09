def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False,  FocalLoss_= 'none', eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area mg
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1mg
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU: #DIoU
                if FocalLoss_ == 'DIoU':
                    print(' use focal-diou 🚀')
                    return iou - rho2 / c2, (inter/(union + eps)) ** 0.5
                return iou - rho2 / c2  # DIoU
            elif CIoU:  #CIoU  https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47mg
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                if FocalLoss_ == 'CIoU':
                    print(' use ciou 🚀')
                    return iou - (rho2 / c2 + v * alpha), (inter/(union + eps)) ** 0.5# mg
                return iou - (rho2 / c2 + v * alpha)  # CIoU 
            elif SIoU:# SIoU
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if FocalLoss_ == 'SIoU':
                    print(' use focal-siou 🚀')
                    return iou - 0.5 * (distance_cost + shape_cost), (inter/(union + eps)) ** 0.5
                return iou - 0.5 * (distance_cost + shape_cost)
            else:# EIoU
                w_dis=torch.pow(b1_x2-b1_x1-b2_x2+b2_x1, 2)
                h_dis=torch.pow(b1_y2-b1_y1-b2_y2+b2_y1, 2)
                cw2=torch.pow(cw , 2)+eps
                ch2=torch.pow(ch , 2)+eps
                if FocalLoss_ == 'EIoU':
                    print(' use focal-eiou 🚀')
                    return iou - (rho2 / c2 + w_dis / cw2 + h_dis / ch2), (inter/(union + eps)) ** 0.5
                return iou - (rho2 / c2 + w_dis / cw2 + h_dis / ch2)
        else:
            c_area = cw * ch + eps  # convex area
            if FocalLoss_ == 'GIoU':
                print(' use focal-giou 🚀')
                return iou - (c_area - union) / c_area, (inter/(union + eps)) ** 0.5
            return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU