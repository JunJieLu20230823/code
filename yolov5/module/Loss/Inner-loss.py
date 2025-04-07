class Inner_WIoU_Scale:
    ''' monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''

    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


def bbox_iou(box1, box2, x1y1x2y2=True, ratio=1, inner_GIoU=False, inner_DIoU=False, inner_CIoU=False, inner_SIoU=False,
             inner_EIoU=False, inner_WIoU=False, Focal=False, alpha=1, gamma=0.5, scale=False, eps=1e-7):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    # IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU        #IoU
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps

    # Inner-IoU      #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_ * ratio, x1 + w1_ * ratio, \
                                                         y1 - h1_ * ratio, y1 + h1_ * ratio
    inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_ * ratio, x2 + w2_ * ratio, \
                                                         y2 - h2_ * ratio, y2 + h2_ * ratio
    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                  (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    inner_union = w1 * ratio * h1 * ratio + w2 * ratio * h2 * ratio - inner_inter + eps

    inner_iou = inner_inter / inner_union  # inner_iou

    if scale:
        self = Inner_WIoU_Scale(1 - (inner_inter / inner_union))

    if inner_CIoU or inner_DIoU or inner_GIoU or inner_EIoU or inner_SIoU or inner_WIoU:
        cw = inner_b1_x2.maximum(inner_b2_x2) - inner_b1_x1.minimum(
            inner_b2_x1)  # convex (smallest enclosing box) width
        ch = inner_b1_y2.maximum(inner_b2_y2) - inner_b1_y1.minimum(inner_b2_y1)  # convex height
        if inner_CIoU or inner_DIoU or inner_EIoU or inner_SIoU or inner_WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((inner_b2_x1 + inner_b2_x2 - inner_b1_x1 - inner_b1_x2) ** 2 + (
                    inner_b2_y1 + inner_b2_y2 - inner_b1_y1 - inner_b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if inner_CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - inner_iou + (1 + eps))
                if Focal:
                    return inner_iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(
                        inner_inter / (inner_union + eps),
                        gamma)  # Focal_CIoU
                else:
                    return inner_iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif inner_EIoU:
                rho_w2 = ((inner_b2_x2 - inner_b2_x1) - (inner_b1_x2 - inner_b1_x1)) ** 2
                rho_h2 = ((inner_b2_y2 - inner_b2_y1) - (inner_b1_y2 - inner_b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Focal:
                    return inner_iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(
                        inner_inter / (inner_union + eps),
                        gamma)  # Focal_EIou
                else:
                    return inner_iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
            elif inner_SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (inner_b2_x1 + inner_b2_x2 - inner_b1_x1 - inner_b1_x2) * 0.5 + eps
                s_ch = (inner_b2_y1 + inner_b2_y2 - inner_b1_y1 - inner_b1_y2) * 0.5 + eps
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
                if Focal:
                    return inner_iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(
                        inner_inter / (inner_union + eps), gamma)  # Focal_SIou
                else:
                    return inner_iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)  # SIou
            elif inner_WIoU:
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(Inner_WIoU_Scale, '_scaled_loss')(self), (1 - inner_iou) * torch.exp(
                        (rho2 / c2)), inner_iou  # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return inner_iou, torch.exp((rho2 / c2))  # WIoU v1
            if Focal:
                return inner_iou - rho2 / c2, torch.pow(inner_inter / (inner_union + eps), gamma)  # Focal_DIoU
            else:
                return inner_iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        if Focal:
            return inner_iou - torch.pow((c_area - inner_union) / c_area + eps, alpha), torch.pow(
                inner_inter / (inner_union + eps),
                gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            return inner_iou - torch.pow((c_area - inner_union) / c_area + eps,
                                         alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        return inner_iou, torch.pow(inner_inter / (inner_union + eps), gamma)  # Focal_IoU
    else:
        return inner_iou  # IoU