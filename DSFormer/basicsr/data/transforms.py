import cv2
import random
import numpy as np


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

#单分支
# def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale, gt_path):
#     """Paired random crop.
#
#     It crops lists of lq and gt images with corresponding locations.
#
#     Args:
#         img_gts (list[ndarray] | ndarray): GT images. Note that all images
#             should have the same shape. If the input is an ndarray, it will
#             be transformed to a list containing itself.
#         img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
#             should have the same shape. If the input is an ndarray, it will
#             be transformed to a list containing itself.
#         lq_patch_size (int): LQ patch size.
#         scale (int): Scale factor.
#         gt_path (str): Path to ground-truth.
#
#     Returns:
#         list[ndarray] | ndarray: GT images and LQ images. If returned results
#             only have one element, just return ndarray.
#     """
#
#     if not isinstance(img_gts, list):
#         img_gts = [img_gts]
#     if not isinstance(img_lqs, list):
#         img_lqs = [img_lqs]
#
#     h_lq, w_lq, _ = img_lqs[0].shape
#     h_gt, w_gt, _ = img_gts[0].shape
#     # print(f"💥 DEBUG: scale = {scale}, type: {type(scale)}")
#     # print(f"💥 DEBUG: lq_patch_size = {lq_patch_size}, type: {type(lq_patch_size)}")
#
#     gt_patch_size = int(lq_patch_size * scale)
#
#     if h_gt != h_lq * scale or w_gt != w_lq * scale:
#         print(gt_path)
#         raise ValueError(
#             f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
#             f'multiplication of LQ ({h_lq}, {w_lq}).')
#     if h_lq < lq_patch_size or w_lq < lq_patch_size:
#         raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
#                          f'({lq_patch_size}, {lq_patch_size}). '
#                          f'Please remove {gt_path}.')
#
#     # randomly choose top and left coordinates for lq patch
#     top = random.randint(0, h_lq - lq_patch_size)
#     left = random.randint(0, w_lq - lq_patch_size)
#
#     # crop lq patch
#     img_lqs = [
#         v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
#         for v in img_lqs
#     ]
#
#     # crop corresponding gt patch
#     top_gt, left_gt = int(top * scale), int(left * scale)
#     img_gts = [
#         v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
#         for v in img_gts
#     ]
#     if len(img_gts) == 1:
#         img_gts = img_gts[0]
#     if len(img_lqs) == 1:
#         img_lqs = img_lqs[0]
#     return img_gts, img_lqs

import random
import numpy as np
#双分支
def paired_random_crop(img_gts, img_lq1s, img_lq2s, lq_patch_size, scale, gt_path):
    """Paired random crop.
img_gt, img_lq1, img_lq2, gt_size, scale
    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. If it's an ndarray, it will be wrapped in a list.
        img_lqs (list[ndarray] | ndarray): LQ images. If it's an ndarray, it will be wrapped in a list.
        lq_patch_size (int or ndarray): LQ patch size (should be an int, but may be wrongly passed as an array).
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: Cropped GT and LQ images.
    """

    # 💥 **防止 `lq_patch_size` 不是 `int`，确保它是整数**
    if isinstance(lq_patch_size, np.ndarray):
        H, W, _ = lq_patch_size.shape  # 获取图像的高和宽
        lq_patch_size = min(H, W)  # 取最小边长作为 patch_size，防止超出范围
    elif not isinstance(lq_patch_size, int):
        raise TypeError(f"🚨 `lq_patch_size` 必须是 `int`，但却是 {type(lq_patch_size)}")

    # 确保 `img_gts` 和 `img_lqs` 是列表
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lq1s, list):
        img_lq1s = [img_lq1s]
    if not isinstance(img_lq2s, list):
        img_lq2s = [img_lq2s]
    # 获取 LQ 和 GT 的图像尺寸
    h_lq, w_lq, _ = img_lq1s[0].shape
    h_lq, w_lq, _ = img_lq2s[0].shape
    h_gt, w_gt, _ = img_gts[0].shape

    # 计算 GT patch 尺寸，确保是整数
    gt_patch_size = int(lq_patch_size * scale)
    # print(f"💥 DEBUG: gt_patch_size = {gt_patch_size}, type: {type(gt_patch_size)}")
    # 💥 关键检查：确保 GT 和 LQ 形状匹配
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(f"🚨 错误: Scale 不匹配! GT: ({h_gt}, {w_gt}), LQ: ({h_lq}, {w_lq}), Scale: {scale}")
        raise ValueError(f"Scale mismatch! GT ({h_gt}, {w_gt}) should be {scale}x of LQ ({h_lq}, {w_lq}).")

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f"🚨 LQ ({h_lq}, {w_lq}) 太小，无法裁剪 ({lq_patch_size}, {lq_patch_size})，请检查数据集 {gt_path}！")

    # **随机选择裁剪的左上角坐标**
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # **裁剪 LQ patch**
    img_lq1s = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq1s]
    img_lq2s = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq2s]
    # **裁剪对应的 GT patch**
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    # **如果只有一个元素，就解包返回**
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lq1s) == 1:
        img_lq1s = img_lq1s[0]
    if len(img_lq2s) == 1:
        img_lq2s = img_lq2s[0]

    return img_gts, img_lq1s,img_lq2s
#共享权重
def paired_random_crop_DP(img_gts, img_lq1s, img_lq2s, lq_patch_size, scale, gt_path):
    """Paired random crop.
img_gt, img_lq1, img_lq2, gt_size, scale
    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. If it's an ndarray, it will be wrapped in a list.
        img_lqs (list[ndarray] | ndarray): LQ images. If it's an ndarray, it will be wrapped in a list.
        lq_patch_size (int or ndarray): LQ patch size (should be an int, but may be wrongly passed as an array).
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: Cropped GT and LQ images.
    """

    # 💥 **防止 `lq_patch_size` 不是 `int`，确保它是整数**
    if isinstance(lq_patch_size, np.ndarray):
        H, W, _ = lq_patch_size.shape  # 获取图像的高和宽
        lq_patch_size = min(H, W)  # 取最小边长作为 patch_size，防止超出范围
    elif not isinstance(lq_patch_size, int):
        raise TypeError(f"🚨 `lq_patch_size` 必须是 `int`，但却是 {type(lq_patch_size)}")

    # 确保 `img_gts` 和 `img_lqs` 是列表
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lq1s, list):
        img_lq1s = [img_lq1s]
    if not isinstance(img_lq2s, list):
        img_lq2s = [img_lq2s]
    # 获取 LQ 和 GT 的图像尺寸
    h_lq, w_lq, _ = img_lq1s[0].shape
    h_lq, w_lq, _ = img_lq2s[0].shape
    h_gt, w_gt, _ = img_gts[0].shape

    # 💥 额外 Debug 打印，确保数据类型正确
    # print(f"💥 DEBUG: lq_patch_size = {lq_patch_size}, type: {type(lq_patch_size)}")
    # print(f"💥 DEBUG: scale = {scale}, type: {type(scale)}")

    # 计算 GT patch 尺寸，确保是整数
    gt_patch_size = int(lq_patch_size * scale)
    # print(f"💥 DEBUG: gt_patch_size = {gt_patch_size}, type: {type(gt_patch_size)}")

    # 💥 关键检查：确保 GT 和 LQ 形状匹配
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(f"🚨 错误: Scale 不匹配! GT: ({h_gt}, {w_gt}), LQ: ({h_lq}, {w_lq}), Scale: {scale}")
        raise ValueError(f"Scale mismatch! GT ({h_gt}, {w_gt}) should be {scale}x of LQ ({h_lq}, {w_lq}).")

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f"🚨 LQ ({h_lq}, {w_lq}) 太小，无法裁剪 ({lq_patch_size}, {lq_patch_size})，请检查数据集 {gt_path}！")

    # **随机选择裁剪的左上角坐标**
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # **裁剪 LQ patch**
    img_lq1s = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq1s]
    img_lq2s = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq2s]
    # **裁剪对应的 GT patch**
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    # **如果只有一个元素，就解包返回**
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lq1s) == 1:
        img_lq1s = img_lq1s[0]
    if len(img_lq2s) == 1:
        img_lq2s = img_lq2s[0]

    return img_gts, img_lq1s,img_lq2s

#单分支
# def paired_random_crop_DP(img_lqLs, img_lqRs, img_gts, gt_patch_size, scale, gt_path):
#     if not isinstance(img_gts, list):
#         img_gts = [img_gts]
#     if not isinstance(img_lqLs, list):
#         img_lqLs = [img_lqLs]
#     if not isinstance(img_lqRs, list):
#         img_lqRs = [img_lqRs]
#
#     h_lq, w_lq, _ = img_lqLs[0].shape
#     h_gt, w_gt, _ = img_gts[0].shape
#     lq_patch_size = gt_patch_size // scale
#
#     if h_gt != h_lq * scale or w_gt != w_lq * scale:
#         raise ValueError(
#             f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
#             f'multiplication of LQ ({h_lq}, {w_lq}).')
#     if h_lq < lq_patch_size or w_lq < lq_patch_size:
#         raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
#                          f'({lq_patch_size}, {lq_patch_size}). '
#                          f'Please remove {gt_path}.')
#
#     # randomly choose top and left coordinates for lq patch
#     top = random.randint(0, h_lq - lq_patch_size)
#     left = random.randint(0, w_lq - lq_patch_size)
#
#     # crop lq patch
#     img_lqLs = [
#         v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
#         for v in img_lqLs
#     ]
#
#     img_lqRs = [
#         v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
#         for v in img_lqRs
#     ]
#
#     # crop corresponding gt patch
#     top_gt, left_gt = int(top * scale), int(left * scale)
#     img_gts = [
#         v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
#         for v in img_gts
#     ]
#     if len(img_gts) == 1:
#         img_gts = img_gts[0]
#     if len(img_lqLs) == 1:
#         img_lqLs = img_lqLs[0]
#     if len(img_lqRs) == 1:
#         img_lqRs = img_lqRs[0]
#     return img_lqLs, img_lqRs, img_gts

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5


    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
