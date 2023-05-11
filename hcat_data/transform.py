import torch
from torch import Tensor
import torchvision.transforms.functional as ttf
from torchvision.ops import box_convert
from typing import Dict, Tuple, Union, Sequence, List
import math
import random


@torch.jit.script
def _get_box(mask: Tensor, device: str, threshold: int) -> Tuple[Tensor, Tensor]:
    # mask in shape of 300, 400, 1 [H, W, z=1]
    nonzero = torch.nonzero(mask)  # Q, 3=[x,y,z]
    label = mask.max()

    box = torch.tensor([-1, -1, -1, -1], dtype=torch.long, device=device)

    # Recall, image in shape of [C, H, W]

    if nonzero.numel() > threshold:
        x0 = torch.min(nonzero[:, 1])
        x1 = torch.max(nonzero[:, 1])
        y0 = torch.min(nonzero[:, 0])
        y1 = torch.max(nonzero[:, 0])

        if (x1 - x0 > 0) and (y1 - y0 > 0):
            box[0] = x0
            box[1] = y0
            box[2] = x1
            box[3] = y1

    return label, box


@torch.jit.script
def _get_affine_matrix(
    center: List[float],
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
    device: str,
) -> Tensor:
    # We need compute the affine transformation matrix: M = T * C * RSS * C^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    T: Tensor = torch.eye(3, device=device)
    T[0, -1] = translate[0]
    T[1, -1] = translate[1]

    C: Tensor = torch.eye(3, device=device)
    C[0, -1] = center[0]
    C[1, -1] = center[1]

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    RSS = torch.tensor([[a, b, 0.0], [c, d, 0.0], [0.0, 0.0, 1.0]], device=device)
    RSS = RSS * scale
    RSS[-1, -1] = 1

    return T @ C @ RSS @ torch.inverse(C)


@torch.jit.script
def _prepare_box_matrix(boxes: Tensor) -> Tensor:
    """
    :param boxes: [N, [x0, y0, x1, y1]]
    :return:
    """
    boxes = boxes.unsqueeze(0)

    bl = boxes[:, :, [0, 1]]  # Bottom Left
    br = boxes[:, :, [2, 1]]  # Bottom Right
    tl = boxes[:, :, [0, 3]]  # Top Left
    tr = boxes[:, :, [2, 3]]  # Top Right

    out = torch.cat((bl, br, tl, tr), dim=0)
    ones = torch.ones((4, out.shape[1], 1), device=out.device)
    return torch.cat((out, ones), dim=2).transpose(1, 2)


@torch.jit.script
def _revert_box_matrix(boxes: Tensor) -> Tensor:
    """
    Reverts each explicit corner matrix to bbox format.

    :param boxes: [4=[bl,br,tl,tr], 2, N]
    :return: boxes[N, [x0, y0, x1, y1]]
    """

    x0 = torch.min(boxes[:, 0, :], dim=0)[0].unsqueeze(0)
    y0 = torch.min(boxes[:, 1, :], dim=0)[0].unsqueeze(0)
    x1 = torch.max(boxes[:, 0, :], dim=0)[0].unsqueeze(0)
    y1 = torch.max(boxes[:, 1, :], dim=0)[0].unsqueeze(0)

    return torch.cat((x0, y0, x1, y1), dim=0).T


# @torch.jit.script
def merged_transform_2D(data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # constants #
    device: str = str(data_dict["image"].device)

    # image should be in shape of [c, h, w]
    CROP_WIDTH = torch.tensor([512], device=device)
    CROP_HEIGHT = torch.tensor([512], device=device)

    AFFINE_RATE = torch.tensor(0.66, device=device)
    AFFINE_SCALE = torch.tensor((0.85, 1.45), device=device)
    AFFINE_YAW = torch.tensor((-180, 180), device=device)
    AFFINE_SHEAR = torch.tensor((-7, 7), device=device)

    FLIP_RATE = torch.tensor(0.5, device=device)

    BLUR_RATE = torch.tensor(0.33, device=device)
    BLUR_KERNEL_TARGETS = torch.tensor([3], device=device, dtype=torch.int)

    BRIGHTNESS_RATE = torch.tensor(0.33, device=device)
    BRIGHTNESS_RANGE: List[Tensor] = [
        torch.tensor((-0.15, 0.15), device=device),  # actin
        torch.tensor((-0.15, 0.15), device=device),  # gfp
        torch.tensor((-0.15, 0.15), device=device),  # noise
    ]

    CONTRAST_RATE = torch.tensor(0.33, device=device)
    CONTRAST_RANGE = torch.tensor((0.75, 2.0), device=device)

    NOISE_GAMMA = torch.tensor(0.1, device=device)
    NOISE_RATE = torch.tensor(0.33, device=device)

    CHANNEL_SHUFFLE_RATE = torch.tensor(0.5, device=device)
    COLOR_TRANSFORM_RATE = torch.tensor(0.33, device=device)

    SOLARIZE_THRESHOLD = torch.tensor(0.66, device=device)

    image: Tensor = torch.clone(data_dict["image"].to(device))
    boxes: Tensor = torch.clone(data_dict["boxes"].to(device))
    labels: Tensor = torch.clone(data_dict["labels"].to(device))

    # ---------- random crop
    # only run if the image is bigger than crop size...
    w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])

    if image.shape[1] > CROP_WIDTH or image.shape[2] > CROP_HEIGHT:
        ind = torch.randint(
            boxes.shape[0], (1, 1), dtype=torch.long, device=device
        )  # randomly select box

        box = boxes[ind, :].squeeze()

        x0 = (
            box[0]
            .sub(torch.floor(w / 2))
            .long()
            .clamp(min=0, max=image.shape[1] - w.item())
        )
        y0 = (
            box[1]
            .sub(torch.floor(h / 2))
            .long()
            .clamp(min=0, max=image.shape[2] - h.item())
        )

        x1 = x0 + w
        y1 = y0 + h

        # Strange edge cases. Should never happen I think....
        if image.shape[1] < CROP_WIDTH:
            x0 = torch.tensor(0)
            x1 = torch.tensor(image.shape[1] - 1)

        if image.shape[2] < CROP_HEIGHT:
            y0 = torch.tensor(0)
            y1 = torch.tensor(image.shape[2] - 1)

        image = image[:, y0.item() : y1.item(), x0.item() : x1.item()]
        # image should be of shape [c, h, w]

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - x0
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - y0

        ind_x = torch.logical_and(boxes[:, 0] >= 0, boxes[:, 2] < w)
        ind_y = torch.logical_and(boxes[:, 1] >= 0, boxes[:, 3] < h)
        ind = torch.logical_and(ind_x, ind_y)

        boxes = boxes[ind, :]
        labels = labels[ind]

    # -------------------affine
    if torch.rand(1, device=device) < AFFINE_RATE:
        angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(
            1, device=device
        ) + AFFINE_YAW[0]
        shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(
            1, device=device
        ) + AFFINE_SHEAR[0]
        scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(
            1, device=device
        ) + AFFINE_SCALE[0]

        # angle = torch.tensor(45.0)
        # shear = torch.tensor(0.0)
        # scale = torch.tensor(1.0)

        _, x, y = image.shape

        mat: Tensor = _get_affine_matrix(
            center=[image.shape[1] / 2, image.shape[2] / 2],
            angle=angle.item(),
            translate=[0.0, 0.0],
            scale=scale.item(),
            shear=[float(shear.item()), float(shear.item())],
            device=str(image.device),
        )

        boxes = _revert_box_matrix(mat @ _prepare_box_matrix(boxes))

        image = ttf.affine(
            image,
            angle=angle.item(),
            shear=[float(shear.item())],
            scale=scale.item(),
            translate=[0, 0],
        )

        # Correct via scale facor!
        # At a 45 degree angle the boxes are inflated. We can reduce to keep more consistent around bundle...
        boxes = box_convert(boxes, "xyxy", "cxcywh")  # From torchvision.ops
        box_correction_factor = (
            torch.abs(torch.cos((angle * 2) * 3.14159 / 180)).mul(0.2).add(0.8)
        )
        boxes[:, [-2, -1]] = boxes[:, [-2, -1]] * box_correction_factor
        boxes = box_convert(boxes, "cxcywh", "xyxy")  # From torchvision.ops

        # Remove out of bounds Boxes (with some buffer)
        buffer = 15
        ind_x = torch.logical_and(boxes[:, 0] >= -buffer, boxes[:, 2] < w + buffer)
        ind_y = torch.logical_and(boxes[:, 1] >= -buffer, boxes[:, 3] < h + buffer)
        ind = torch.logical_and(ind_x, ind_y)

        boxes = boxes[ind, :]
        labels = labels[ind]

    # ------------------- horizontal flip
    if torch.rand(1, device=device) < FLIP_RATE:
        image = ttf.vflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [0, 2]] = image.shape[2] - boxes[:, [2, 0]]

    # ------------------- vertical flip
    if torch.rand(1, device=device) < FLIP_RATE:
        image = ttf.hflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [1, 3]] = image.shape[1] - boxes[:, [3, 1]]

    # -------------------2Channel -> 3Channel
    c, x, y = image.shape
    if c < 3:
        image = torch.cat(
            (torch.rand((1, x, y), device=image.device).mul(0.2), image), dim=0
        )[:, :, :]

    # -------------------Shuffle Channels
    if torch.rand(1, device=device) < CHANNEL_SHUFFLE_RATE:
        indicies = torch.randperm(3)
        image = image[indicies, ...]

    # ------------------- blur
    if torch.rand(1, device=device) < BLUR_RATE:
        kern: int = int(
            BLUR_KERNEL_TARGETS[
                int(torch.randint(0, len(BLUR_KERNEL_TARGETS), (1, 1)).item())
            ].item()
        )
        image = ttf.gaussian_blur(
            image.unsqueeze(1).transpose(1, -1).squeeze(-1), [kern, kern]
        )
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)

    # ------------------- bright
    if torch.rand(1, device=device) < BRIGHTNESS_RATE:
        # get random brightness value for actin
        alpha = torch.ones(len(BRIGHTNESS_RANGE), device=device, dtype=torch.float)
        for i, val in enumerate(BRIGHTNESS_RANGE):
            a = (val[1] - val[0]) * torch.rand((1), device=device) + val[0]
            alpha[i] = a.item()
        alpha = alpha.reshape(image.shape[0], 1, 1)
        image = image.add(alpha)

    # ------------------- contrast
    if torch.rand(1, device=device) < CONTRAST_RATE:
        contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
            (image.shape[0]), device=device
        ) + CONTRAST_RANGE[0]

        for c in range(image.shape[0]):
            image[c, ...] = ttf.adjust_contrast(
                image[[c], ...], contrast_val[c].item()
            ).squeeze(0)

    # --------------------- Solarize or Invert (mutually exclusive)
    if torch.rand(1, device=device) < COLOR_TRANSFORM_RATE:
        rand = torch.rand(1, device=device)
        image = (
            ttf.solarize(image, SOLARIZE_THRESHOLD.item())
            if rand > 0.5
            else ttf.invert(image)
        )

    # ------------------- noise
    if torch.rand(1, device=device) < NOISE_RATE:
        image = image.add(torch.rand(image.shape, device=device) * NOISE_GAMMA)
        image = torch.clamp(image, 0, 1)

    # ------------- wrap up
    ind = boxes[:, 0:2] < boxes[:, 2:]
    ind = torch.logical_and(ind[:, 0], ind[:, 1])

    boxes = boxes[ind, :]
    labels = labels[ind]

    # _, x, y = image.shape
    # image = torch.cat((torch.rand((1, x, y), device=image.device).mul(0.2), image), dim=0)[:, :, :]

    data_dict["image"] = image
    data_dict["boxes"] = boxes
    data_dict["labels"] = labels

    return data_dict


# @torch.jit.script
def cochlea_transform_2D(data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # constants #
    device: str = str(data_dict["image"].device)

    # image should be in shape of [c, h, w]
    CROP_WIDTH = torch.tensor([250], device=device)
    CROP_HEIGHT = torch.tensor([250], device=device)

    FLIP_RATE = torch.tensor(0.5, device=device)

    BLUR_RATE = torch.tensor(0.33, device=device)
    BLUR_KERNEL_TARGETS = torch.tensor([3], device=device, dtype=torch.int)

    BRIGHTNESS_RATE = torch.tensor(0.33, device=device)
    BRIGHTNESS_RANGE: List[Tensor] = [
        torch.tensor((-0.15, 0.15), device=device),  # actin
        torch.tensor((-0.15, 0.15), device=device),  # gfp
        torch.tensor((-0.15, 0.15), device=device),  # noise
    ]

    CONTRAST_RATE = torch.tensor(0.33, device=device)
    CONTRAST_RANGE = torch.tensor((0.75, 2.0), device=device)

    NOISE_GAMMA = torch.tensor(0.1, device=device)
    NOISE_RATE = torch.tensor(0.33, device=device)

    CHANNEL_SHUFFLE_RATE = torch.tensor(0.5, device=device)
    COLOR_TRANSFORM_RATE = torch.tensor(0.33, device=device)

    SOLARIZE_THRESHOLD = torch.tensor(0.66, device=device)

    image: Tensor = torch.clone(data_dict["image"].to(device))
    boxes: Tensor = torch.clone(data_dict["boxes"].to(device))
    labels: Tensor = torch.clone(data_dict["labels"].to(device))

    w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])

    if image.shape[1] > CROP_WIDTH or image.shape[2] > CROP_HEIGHT:
        ind = torch.randint(
            boxes.shape[0], (1, 1), dtype=torch.long, device=device
        )  # randomly select box

        box = boxes[
            ind, :
        ].squeeze()  # [x0, y0, x1, y1] image in shape [C, Y, X] ??????

        x0 = (
            box[0]
            .sub(torch.floor(w / 2))
            .long()
            .clamp(min=0, max=image.shape[2] - w.item())
        )
        y0 = (
            box[1]
            .sub(torch.floor(h / 2))
            .long()
            .clamp(min=0, max=image.shape[1] - h.item())
        )

        x1 = x0 + w
        y1 = y0 + h

        # print(f'Image Shape: {image.shape} | w:{w.item()} h:{h.item()} | {x0.item()}:{x1.item()}, {y0.item()}:{y1.item()} | Center Box: {box}')

        # Strange edge cases. Should never happen I think....
        if image.shape[1] < CROP_WIDTH:
            print(image.shape)
            x0 = torch.tensor(0)
            x1 = torch.tensor(image.shape[1] - 1)

        if image.shape[2] < CROP_HEIGHT:
            print(image.shape)
            y0 = torch.tensor(0)
            y1 = torch.tensor(image.shape[2] - 1)

        image = image[:, y0.item() : y1.item(), x0.item() : x1.item()]
        # image should be of shape [c, h, w]

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - x0
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - y0
        _temp = boxes[ind, :]

        ind_x = torch.logical_and(boxes[:, 0] >= -10, boxes[:, 2] < w + 10)
        ind_y = torch.logical_and(boxes[:, 1] >= -10, boxes[:, 3] < h + 10)
        ind = torch.logical_and(ind_x, ind_y)

        boxes = boxes[ind, :]
        labels = labels[ind]

    # ------------------- horizontal flip
    if torch.rand(1, device=device) < FLIP_RATE:
        image = ttf.vflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [0, 2]] = image.shape[2] - boxes[:, [2, 0]]

    # ------------------- vertical flip
    if torch.rand(1, device=device) < FLIP_RATE:
        image = ttf.hflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [1, 3]] = image.shape[1] - boxes[:, [3, 1]]

    # -------------------2Channel -> 3Channel
    c, x, y = image.shape
    if c < 3:
        image = torch.cat(
            (torch.rand((1, x, y), device=image.device).mul(0.2), image), dim=0
        )[:, :, :]

    # -------------------Shuffle Channels
    if torch.rand(1, device=device) < CHANNEL_SHUFFLE_RATE:
        indicies = torch.randperm(3)
        image = image[indicies, ...]

    # ------------------- blur
    if torch.rand(1, device=device) < BLUR_RATE:
        kern: int = int(
            BLUR_KERNEL_TARGETS[
                int(torch.randint(0, len(BLUR_KERNEL_TARGETS), (1, 1)).item())
            ].item()
        )
        image = ttf.gaussian_blur(
            image.unsqueeze(1).transpose(1, -1).squeeze(-1), [kern, kern]
        )
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)

    # ------------------- bright
    if torch.rand(1, device=device) < BRIGHTNESS_RATE:
        # get random brightness value for actin
        alpha = torch.ones(len(BRIGHTNESS_RANGE), device=device, dtype=torch.float)
        for i, val in enumerate(BRIGHTNESS_RANGE):
            a = (val[1] - val[0]) * torch.rand((1), device=device) + val[0]
            alpha[i] = a.item()
        alpha = alpha.reshape(image.shape[0], 1, 1)
        image = image.add(alpha)

    # ------------------- contrast
    if torch.rand(1, device=device) < CONTRAST_RATE:
        contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
            (image.shape[0]), device=device
        ) + CONTRAST_RANGE[0]

        for c in range(image.shape[0]):
            image[c, ...] = ttf.adjust_contrast(
                image[[c], ...], contrast_val[c].item()
            ).squeeze(0)

    # --------------------- Solarize or Invert (mutually exclusive)
    if torch.rand(1, device=device) < COLOR_TRANSFORM_RATE:
        rand = torch.rand(1, device=device)
        image = (
            ttf.solarize(image, SOLARIZE_THRESHOLD.item())
            if rand > 0.5
            else ttf.invert(image)
        )

    # ------------------- noise
    if torch.rand(1, device=device) < NOISE_RATE:
        image = image.add(torch.rand(image.shape, device=device) * NOISE_GAMMA)
        image = torch.clamp(image, 0, 1)

    # ------------- wrap up
    ind = boxes[:, 0:2] < boxes[:, 2:]
    ind = torch.logical_and(ind[:, 0], ind[:, 1])

    boxes = boxes[ind, :]
    labels = labels[ind]

    # _, x, y = image.shape
    # image = torch.cat((torch.rand((1, x, y), device=image.device).mul(0.2), image), dim=0)[:, :, :]

    data_dict["image"] = image
    data_dict["boxes"] = boxes
    data_dict["labels"] = labels

    return data_dict


@torch.jit.script
def val_transforms_2D(data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # constants #
    device: str = str(data_dict["image"].device)

    # image should be in shape of [c, h, w]
    CROP_WIDTH = torch.tensor([300], device=device)
    CROP_HEIGHT = torch.tensor([300], device=device)

    CHANNEL_SHUFFLE_RATE = torch.tensor(0.5, device=device)

    image: Tensor = data_dict["image"].to(device)

    boxes: Tensor = torch.clone(data_dict["boxes"]).to(device)
    labels = torch.clone(data_dict["labels"]).to(device)

    w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])

    if image.shape[1] > CROP_WIDTH or image.shape[2] > CROP_HEIGHT:
        ind = torch.randint(
            boxes.shape[0], (1, 1), dtype=torch.long, device=device
        )  # randomly select box

        box = boxes[
            ind, :
        ].squeeze()  # [x0, y0, x1, y1] image in shape [C, Y, X] ??????

        x0 = (
            box[0]
            .sub(torch.floor(w / 2))
            .long()
            .clamp(min=0, max=image.shape[2] - w.item())
        )
        y0 = (
            box[1]
            .sub(torch.floor(h / 2))
            .long()
            .clamp(min=0, max=image.shape[1] - h.item())
        )

        x1 = x0 + w
        y1 = y0 + h

        # print(f'Image Shape: {image.shape} | w:{w.item()} h:{h.item()} | {x0.item()}:{x1.item()}, {y0.item()}:{y1.item()} | Center Box: {box}')

        # Strange edge cases. Should never happen I think....
        if image.shape[1] < CROP_WIDTH:
            x0 = torch.tensor(0)
            x1 = torch.tensor(image.shape[1] - 1)

        if image.shape[2] < CROP_HEIGHT:
            y0 = torch.tensor(0)
            y1 = torch.tensor(image.shape[2] - 1)

        image = image[:, y0.item() : y1.item(), x0.item() : x1.item()]
        # image should be of shape [c, h, w]

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - x0
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - y0
        _temp = boxes[ind, :]

        ind_x = torch.logical_and(boxes[:, 0] >= -10, boxes[:, 2] < w + 10)
        ind_y = torch.logical_and(boxes[:, 1] >= -10, boxes[:, 3] < h + 10)
        ind = torch.logical_and(ind_x, ind_y)

        boxes = boxes[ind, :]
        labels = labels[ind]

    # -------------------2Channel -> 3Channel
    c, x, y = image.shape
    if c < 3:
        image = torch.cat(
            (torch.rand((1, x, y), device=image.device).mul(0.2), image), dim=0
        )[:, :, :]

    # -------------------Shuffle Channels
    if torch.rand(1, device=device) < CHANNEL_SHUFFLE_RATE:
        indicies = torch.randperm(3)
        image = image[indicies, ...]

    # ------------- wrap up
    ind = boxes[:, 0:2] < boxes[:, 2:]
    ind = torch.logical_and(ind[:, 0], ind[:, 1])

    boxes = boxes[ind, :]
    labels = labels[ind]

    # _, x, y = image.shape
    # image = torch.cat((torch.rand((1, x, y), device=image.device).mul(0.2), image), dim=0)[:, :, :]

    data_dict["image"] = image.float()
    data_dict["boxes"] = boxes
    data_dict["labels"] = labels

    return data_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.utils

    # torch.manual_seed(0)

    image = torch.rand((2, 1000, 1000))
    # boxes = torch.rand((100, 4))
    boxes = torch.rand((20000, 2)).mul(1000)
    boxes = torch.cat((boxes, boxes.add(20)), dim=1)
    labels = torch.rand((20000))

    a = merged_transform_2D({"image": image, "boxes": boxes, "labels": labels})
    images = a["image"]
    boxes = a["boxes"]

    labels = ["OHC" if x == 1 else "IHC" for x in a["labels"]]
    colors = ["red" if x == 1 else "blue" for x in a["labels"]]

    overlay = torchvision.utils.draw_bounding_boxes(
        images.mul(256).type(torch.uint8), boxes, labels, colors=colors
    )
    plt.imshow(overlay.permute(1, 2, 0))
    plt.show()

    # for i in range(10):
    #     a = merged_transform_2D({'image': image, 'boxes': boxes, 'labels': labels})
    #     plt.imshow(a['image'][1,...].numpy())
    #     for box in a['boxes']:
    #         plt.plot(box[[0,2]], box[[1,1]], 'r')
    #         plt.plot(box[[0,2]], box[[3,3]], 'r')
    #         plt.plot(box[[0,0]], box[[1,3]], 'r')
    #         plt.plot(box[[2,2]], box[[1,3]], 'r')
    #         # plt.plot(box[[0,0]], box[[1,3]], 'r')
    #         # plt.plot(box[2], box[3], 'bo')
    #     plt.show()
