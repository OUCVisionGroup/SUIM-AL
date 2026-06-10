
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import torch
from os import path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from skimage import img_as_ubyte
from basicsr.models import create_model
from basicsr.utils.options import parse
import utils


def load_model(opt_path, weight_path):
    """加载 RetinexFormer 模型"""
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False

    model = create_model(opt).net_g
    checkpoint = torch.load(weight_path)

    try:
        model.load_state_dict(checkpoint['params'])
    except RuntimeError:
        new_checkpoint = {f"module.{k}": v for k, v in checkpoint['params'].items()}
        model.load_state_dict(new_checkpoint)

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    return model

#
def pad_image(img, factor=4):
    """给图像做 padding，确保尺寸是4的倍数"""
    h, w = img.shape[2], img.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    return F.pad(img, (0, padw, 0, padh), 'reflect')

def process_image(img_path, model):
    """处理单张图像，并返回推理时间"""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_ = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    input_ = pad_image(input_)

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.inference_mode():
        outputs = model(input_)
        if isinstance(outputs, tuple):
            restored = outputs[1]
        else:
            restored = outputs

    torch.cuda.synchronize()
    infer_time = time.time() - start_time

    restored = restored[:, :, :img.shape[0], :img.shape[1]]
    restored = torch.clamp(restored, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

    return img_as_ubyte(restored), infer_time

def retinexformer_test(input_dir, output_dir, model, result_dir_enhanced):
    image_paths = natsorted(glob(os.path.join(input_dir, "*.*")))
    os.makedirs(result_dir_enhanced, exist_ok=True)

    total_time = 0.0
    num_images = len(image_paths)

    for img_path in tqdm(image_paths, desc="Processing Images"):
        torch.cuda.empty_cache()

        restored, infer_time = process_image(img_path, model)
        total_time += infer_time

        save_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(
            os.path.join(result_dir_enhanced, f"{save_name}.png"),
            cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        )

    avg_time = total_time / num_images
    fps = 1.0 / avg_time

    print("\n========== Inference Time Statistics ==========")
    print(f"Total images      : {num_images}")
    print(f"Total time (s)    : {total_time:.3f}")
    print(f"Avg time / image  : {avg_time*1000:.2f} ms")
    print(f"FPS               : {fps:.2f}")
    print("==============================================\n")


if __name__ == "__main__":
    # 路径设置
    input_dir = ''  # 输入图像路径
    output_dir = '/'  # 结果保存路径
    opt_path = ''  # 配置文件路径
    weight_path = ''  # 权重路径
    result_dir_enhanced = os.path.join(output_dir, 'illu')
    # 加载模型
    model = load_model(opt_path, weight_path)
    # 批量处理图像
    retinexformer_test(input_dir, output_dir, model, result_dir_enhanced)

