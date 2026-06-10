
def invert_saliency_map(saliency_map):
    # 假设 saliency_map 是单通道图像 (batch_size, 1, H, W)
    inverted_map = 1 - saliency_map  # 反转显著图，显著物体为黑色，非显著物体为白色
    return inverted_map

# 步骤2：与增强图相乘
def apply_inverted_saliency_to_enhanced_image(enhanced_image, inverted_map):
    # 确保 inverted_map 和 enhanced_image 的尺寸一致
    if enhanced_image.size(2) != inverted_map.size(2) or enhanced_image.size(3) != inverted_map.size(3):
        inverted_map = F.interpolate(inverted_map, size=(enhanced_image.size(2), enhanced_image.size(3)),
                                     mode='bilinear', align_corners=False)
    # 将反转图和增强图相乘
    enhanced_image_no_saliency = enhanced_image * inverted_map
    return enhanced_image_no_saliency
def apply_inverted_saliency_to_ref_image(gt, inverted_map):
    # 确保 inverted_map 和 enhanced_image 的尺寸一致
    if gt.size(2) != inverted_map.size(2) or gt.size(3) != inverted_map.size(3):
        inverted_map = F.interpolate(inverted_map, size=(gt.size(2), gt.size(3)),
                                     mode='bilinear', align_corners=False)

    # 将反转图和增强图相乘
    ref_image_no_saliency = gt * inverted_map
    return ref_image_no_saliency
# 步骤3：计算像素损失（例如 L1 损失）
def compute_pixel_loss(enhanced_image_no_saliency, target_image):
    # 计算 L1 损失，忽略显著部分，计算非显著部分
    pixel_loss = F.mse_loss(enhanced_image_no_saliency, target_image)
    return pixel_loss


from datetime import date
import importlib
import logging
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob
import torch.nn as nn
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from torchvision import transforms

try:
    from torch.cuda.amp import autocast, GradScaler

    load_amp = True
except:
    load_amp = False

class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3, cuda=True):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
        self.Kx = torch.tensor(SobelX, dtype=torch.float32, device='cuda').expand(self.chans, 1, 3, 3)
        self.Ky = torch.tensor(SobelY, dtype=torch.float32, device='cuda').expand(self.chans, 1, 3, 3)

    def get_gradients(self, im):
        gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
        gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        gradX_true, gradY_true = self.get_gradients(true)
        gradX_pred, gradY_pred = self.get_gradients(pred)
        return (torch.abs(gradX_true.abs() - gradX_pred.abs())**2 + torch.abs(gradY_true.abs() - gradY_pred.abs())**2).mean()


def edge_aware_loss(pred, gt):
    B, C, H, W = pred.shape

    # 定义 Sobel 卷积核
    sobel_kernel_x = torch.tensor([[[[-1, 1]]]], dtype=torch.float32, device=pred.device)
    sobel_kernel_y = torch.tensor([[[[-1], [1]]]], dtype=torch.float32, device=pred.device)

    # 将通道数扩展到输入通道数
    sobel_kernel_x = sobel_kernel_x.expand(C, 1, 1, 2)
    sobel_kernel_y = sobel_kernel_y.expand(C, 1, 2, 1)

    # 计算预测和真实图像的梯度
    grad_pred_x = F.conv2d(pred, sobel_kernel_x, padding=(0, 1), groups=C)
    grad_pred_y = F.conv2d(pred, sobel_kernel_y, padding=(1, 0), groups=C)

    grad_gt_x = F.conv2d(gt, sobel_kernel_x, padding=(0, 1), groups=C)
    grad_gt_y = F.conv2d(gt, sobel_kernel_y, padding=(1, 0), groups=C)

    # 计算损失
    loss = F.l1_loss(grad_pred_x, grad_gt_x) + F.l1_loss(grad_pred_y, grad_gt_y)
    return loss

import math

def rgb_to_lab(image):
    """将RGB图像转换为Lab颜色空间，返回L通道"""
    # 转换为 NumPy 数组以便使用 OpenCV 函数
    image = image.detach().cpu().numpy()
    # 如果图像范围是 [0, 1]，转换为 [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # 将图像从 RGB 转换为 BGR (OpenCV 使用 BGR 格式)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 转换为 LAB 色彩空间
    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    # 提取 L 通道
    L_channel = lab_image[:, :, 0]

    return L_channel

import torch.nn as nn

class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def rgb_to_lab(self, rgb_img):
            """
            将 RGB 图像转换为 LAB 图像
            输入：RGB 图像 (B, C, H, W)
            输出：LAB 图像 (B, C, H, W)
            """
            batch_size, _, height, width = rgb_img.size()
            rgb_img = rgb_img.permute(0, 2, 3, 1).cpu()  # 转为 (B, H, W, C) 并转回 CPU

            lab_img = torch.zeros_like(rgb_img).float()  # 初始化空的 LAB 图像

            # 使用 PIL 转换每一张图像
            for b in range(batch_size):
                # 将 RGB 图像转为 PIL 格式
                pil_image = Image.fromarray((rgb_img[b].detach().numpy() * 255).astype('uint8'))

                # 转换为 LAB 色彩空间
                lab_image = pil_image.convert('LAB')

                # 将 LAB 图像转换为 numpy 数组
                lab_array = torch.tensor(np.array(lab_image)).float()

                # 归一化到 [0, 1] 范围
                lab_array = lab_array / 255.0

                # 存储转换后的图像
                lab_img[b] = lab_array

            # 转换回原设备
            lab_img = lab_img.to(rgb_img.device)

            # 转换维度为 (B, C, H, W)
            return lab_img.permute(0, 3, 1, 2)
    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        lab_img = self.rgb_to_lab(x)
        L = lab_img[:, 0:1, :, :]
        # x = torch.mean(x, 1, keepdim=True)
        L = L.to(x.device)
        mean = self.pool(L)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d

# class L_exp(nn.Module):
#
#     def __init__(self, patch_size, mean_val):
#         super(L_exp, self).__init__()
#         # print(1)
#         self.pool = nn.AvgPool2d(patch_size)
#         self.mean_val = mean_val
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = torch.mean(x, 1, keepdim=True)
#         mean = self.pool(x)
#
#         d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
#         return d

from PIL import Image
from turtle import forward
import torchvision.transforms as transforms
import torch
import clip
import torch.nn as nn
from torch.nn import functional as F
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")  # ViT-B/32
model.to(device)
img_resize = transforms.Resize((224, 224))
for para in model.parameters():
    para.requires_grad = False


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class Config:
    def __init__(self):
        self.length_prompt = 16
        self.prompt_pretrain_dir = "checkpoints/pretrained_prompt.pth"
        self.load_pretrain_prompt = True
        self.prompt_snapshots_folder = "snapshots/"
        self.num_clip_pretrained_iters = 8000


config = Config()

clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

class Prompts(nn.Module):
    def __init__(self, initials=None):
        super(Prompts, self).__init__()
        self.text_encoder = TextEncoder(model)

        if isinstance(initials, list):
            # 使用自定义文本列表初始化
            tokenized = clip.tokenize(initials).cuda()
            with torch.no_grad():
                embed = model.token_embedding(tokenized)
            self.embedding_prompt = nn.Parameter(embed.clone().detach()).cuda()

        elif isinstance(initials, str):
            # 加载已有的预训练 embedding
            state_dict = torch.load(initials)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            self.embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt'].clone().detach()).cuda()

        else:
            # 默认初始化为 prompt_len 个 "X"
            prompt_text = " ".join(["X"] * config.length_prompt)
            tokenized = clip.tokenize([prompt_text]).cuda()
            with torch.no_grad():
                embed = model.token_embedding(tokenized)
            self.embedding_prompt = nn.Parameter(embed.clone().detach()).cuda()

        self.embedding_prompt.requires_grad = True

    def forward(self, tensor, flag=1):
        prompt_text = " ".join(["X"] * config.length_prompt)
        tokenized_prompts = clip.tokenize([prompt_text]).cuda()

        text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts)

        probs = []
        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            nor = torch.norm(text_features, dim=-1, keepdim=True)
            sim = (100.0 * image_features @ (text_features / nor).T)
            if flag:
                sim = sim.softmax(dim=-1)
                probs.append(sim[:, 0])
            else:
                probs.append(sim)

        return torch.stack(probs, dim=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

# 预处理函数
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))

def img_resize(img_tensor):
    return F.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

from clip import clip
#指定语句
# class L_clip_from_feature(nn.Module):
#     def __init__(self, prompt="Underwater images with uneven illumination"):
#         super().__init__()
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#
#         self.model, _ = clip.load("ViT-B/32", device=self.device)
#         self.model.eval()
#
#         for p in self.model.parameters():
#             p.requires_grad = False
#
#         tokenized = clip.tokenize([prompt]).to(self.device)
#         text_feat = self.model.encode_text(tokenized)
#         self.register_buffer(
#             "text_features",
#             text_feat / text_feat.norm(dim=-1, keepdim=True)
#         )
#
#     def forward(self, x):
#         batch = []
#         for i in range(x.shape[0]):
#             img = img_resize(x[i]).reshape(1, 3, 224, 224)
#             batch.append(img)
#
#         batch = torch.cat(batch, dim=0)
#         batch = clip_normalizer(batch).to(self.device)
#
#         image_features = self.model.encode_image(batch)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#
#         similarity = image_features @ self.text_features.T
#         return similarity.mean()

class L_clip_from_feature(nn.Module):
    def __init__(self, prompts):
        """
        prompts: List[str]  # 三个（或多个）负文本
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # 冻结 CLIP
        for p in self.model.parameters():
            p.requires_grad = False

        # --- 多文本编码 ---
        tokenized = clip.tokenize(prompts).to(self.device)
        text_feat = self.model.encode_text(tokenized)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # 保存为 buffer
        self.register_buffer("text_features", text_feat)

    def forward(self, x):
        batch = []
        for i in range(x.shape[0]):
            img = img_resize(x[i]).reshape(1, 3, 224, 224)
            batch.append(img)

        batch = torch.cat(batch, dim=0)
        batch = clip_normalizer(batch).to(self.device)

        image_features = self.model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # similarity: [B, K]，K=文本个数
        similarity = image_features @ self.text_features.T

        # ❗负 CLIP loss：最像哪一句，就惩罚哪一句
        loss = similarity.max(dim=1).values.mean()

        return loss

import torch
import torch.nn as nn

# 初始化你的 prompt 模块
learn_prompt = Prompts().cuda()

def get_option(opt_path):
    with open(opt_path, 'r') as f:
        option = yaml.safe_load(f)

    option.setdefault('seed', 2022)

    # 👉 自动把某些字段的字符串数字列表转成 int
    def convert_list(val):
        return [int(x) if isinstance(x, str) and x.isdigit() else x for x in val]

    model = option.get('model', {})
    if 'embed_dims' in model:
        model['embed_dims'] = convert_list(model['embed_dims'])
    if 'mlp_ratios' in model:
        model['mlp_ratios'] = convert_list(model['mlp_ratios'])
    if 'serial_depths' in model:
        model['serial_depths'] = convert_list(model['serial_depths'])
    if 'in_chans' in model:
        model['in_chans'] = int(model['in_chans']) if isinstance(model['in_chans'], str) else model['in_chans']
    if 'patch_size' in model:
        model['patch_size'] = int(model['patch_size']) if isinstance(model['patch_size'], str) else model['patch_size']

    return option


def build_optimizer(opt, model):
    optimizer_name = opt['optimizer']
    try:
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(model.parameters(), lr=opt['lr'])
    except:
        raise NotImplementedError('Unable to load optimizer: \'%s\' ' % optimizer_name)

    return optimizer


def build_lr_scheduler(opt, optimizer):
    lr_scheduler_name = opt['lr_scheduler'] if 'lr_scheduler' in opt.keys() else None
    if lr_scheduler_name:
        try:
            lr_scheduler_class = getattr(getattr(torch.optim, 'lr_scheduler'), lr_scheduler_name)
        except:
            raise NotImplementedError(
                'Unable to load lr_scheduler: \'%s\', please check if there are any spelling errors ' % lr_scheduler_name)
        try:
            lr_scheduler = lr_scheduler_class(optimizer, **opt['lr_scheduler_arg'])
        except:
            raise NotImplementedError('Failed to load optimizer')
        return lr_scheduler
    else:
        return None


def build_dataloader(opt, type='train'):
    dataset_name = opt['dataset_name']
    module = __import__('dataset.dataset')
    dataset_class = getattr(module, dataset_name)
    dataset = dataset_class(opt, type)
    dataloader = date.DataLoader(dataset,
                                 batch_size=opt['bs'] if type == 'train' else 1,
                                 num_workers=opt['num_workers'],
                                 shuffle=True if type == 'train' else False)
    return dataloader


def build_model(opt):
    model_name = opt['model_name']
    module = __import__('all_model.' + model_name + '.model')
    model_class = getattr(module, model_name)

    # load model args
    all_args = list(opt.keys())
    model_args = {}
    for i in range(len(all_args) - 4):
        model_args[all_args[i + 4]] = opt.get(all_args[i + 4])
    model = model_class(**model_args)

    if opt['cuda']:
        model = model.cuda()
    if opt['parallel']:
        model = torch.nn.DataParallel(model)

    # load pretrained dict
    if opt['resume_ckpt_path']:
        ckpt_dict = torch.load(opt['resume_ckpt_path'])['net']
        model.load_state_dict(ckpt_dict)

    return model


def build_logger(opt):
    make_dir(os.path.join(opt['save_root'], opt['log']))
    log_path = os.path.join(opt['save_root'], opt['log'], 'logs.log')
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        paths = path.split('/')
        now_path = ''
        for temp_path in paths:
            now_path = os.path.join(now_path, temp_path)
            if not os.path.exists(now_path):
                os.mkdir(now_path)
        return

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_
# -------------------------------


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""
    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)
        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def rgb_to_lab(self, rgb_img):
        """
        将 RGB 图像转换为 LAB 图像
        输入：RGB 图像 (B, C, H, W)
        输出：LAB 图像 (B, C, H, W)
        """
        batch_size, _, height, width = rgb_img.size()
        rgb_img = rgb_img.permute(0, 2, 3, 1).cpu()  # 转为 (B, H, W, C) 并转回 CPU

        lab_img = torch.zeros_like(rgb_img).float()  # 初始化空的 LAB 图像

        # 使用 PIL 转换每一张图像
        for b in range(batch_size):
            # 将 RGB 图像转为 PIL 格式
            pil_image = Image.fromarray((rgb_img[b].detach().cpu().numpy() * 255).astype('uint8'))

            # 转换为 LAB 色彩空间
            lab_image = pil_image.convert('LAB')

            # 将 LAB 图像转换为 numpy 数组
            lab_array = torch.tensor(np.array(lab_image)).float()

            # 归一化到 [0, 1] 范围
            lab_array = lab_array / 255.0

            # 存储转换后的图像
            lab_img[b] = lab_array

        # 转换回原设备
        lab_img = lab_img.to(rgb_img.device)

        # 转换维度为 (B, C, H, W)
        return lab_img.permute(0, 3, 1, 2)

    # 在模型类 (ImageCleanModel) 内部定义
    def get_enhanced_result(self):
        """获取当前模型增强结果"""
        if hasattr(self, 'output'):
            return self.output.detach()  # 返回增强后的图像
        else:
            raise AttributeError("模型尚未生成增强结果，请检查前向传播过程。")

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)  # 如何写 weighted loss 呢？传参构造Loss函数
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    # 教师
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    # 单分支
    # def setup_optimizers(self):
    #     train_opt = self.opt['train']
    #     optim_params = []

    #     for k, v in self.net_g.named_parameters():
    #         if v.requires_grad:
    #             optim_params.append(v)
    #         else:
    #             logger = get_root_logger()
    #             logger.warning(f'Params {k} will not be optimized.')

    #     optim_type = train_opt['optim_g'].pop('type')
    #     if optim_type == 'Adam':
    #         self.optimizer_g = torch.optim.Adam(
    #             optim_params, **train_opt['optim_g'])
    #     elif optim_type == 'AdamW':
    #         self.optimizer_g = torch.optim.AdamW(
    #             optim_params, **train_opt['optim_g'])
    #     else:
    #         raise NotImplementedError(
    #             f'optimizer {optim_type} is not supperted yet.')
    #     self.optimizers.append(self.optimizer_g)

    # def feed_train_data(self, data):
    #     self.lq = data['lq'].to(self.device)
    #     if 'gt' in data:
    #         self.gt = data['gt'].to(self.device)

    #     if self.mixing_flag:
    #         self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    # def feed_data(self, data):
    #     self.lq = data['lq'].to(self.device)
    #     if 'gt' in data:
    #         self.gt = data['gt'].to(self.device)

    def feed_train_data(self, lq1,lq2, gt):
        self.lq1 = lq1.to(self.device)
        self.lq2 = lq2.to(self.device)
        self.gt = gt.to(self.device)
        # self.saliency = saliency_img.to(self.device)
        if self.mixing_flag:
            self.gt, self.lq1 = self.mixing_augmentation(self.gt, self.lq1)
            self.gt, self.lq2 = self.mixing_augmentation(self.gt, self.lq2)
        # if self.real_imgs is not None:
        #     self.gt, self.real_imgs = self.mixing_augmentation(self.gt, self.real_imgs)
    def feed_data(self, data):
        self.lq1 = data['lq1'].to(self.device)
        self.lq2 = data['lq2'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gt' in data:
            self.gt = self.gt

    def optimize_parameters(self, current_iter):
            self.optimizer_g.zero_grad()
            with autocast(enabled=self.use_amp):
                preds1, preds2, illu1, illu2, input1, input2, img1, img2 = self.net_g(self.lq1, self.lq2)
                # preds1, illu1, input1, img1= self.net_g(self.lq)
                device = self.lq1.device  # 假设lq是输入图像，pred和gt会在相同设备上
                self.gt = self.gt.to(device)
                device = self.lq2.device
                # saliency = self.saliency.to(device)
                # real_pred, illu3, real_input, fire3, img3, illu_fea3, feat_real = self.net_g(real_degraded)
                loss_dict = OrderedDict()
                # 开启梯度计算
                self.gt.requires_grad_()
                l_pix1 = F.l1_loss(preds1.to(device), self.gt)
                l_pix2 = F.l1_loss(preds2.to(device), self.gt)
                neg_prompts = [
                    "An underwater photo of uneven illumination.",
                    "A photo of an underwater artificial light source.",
                    "A photo of extreme underwater lighting."
                ]

                clip_loss = L_clip_from_feature(neg_prompts).to(device)

                # pixel loss
                # # 计算损失
                # vgg=VGGContentLoss().to(self.device)
                grad_loss = Gradient_Difference_Loss(alpha=1, chans=3, cuda=True)
                # losssmooth = loss_fn(pred, self.gt)
                edgeloss1 = grad_loss(preds1, self.gt)
                edgeloss2 = grad_loss(preds2, self.gt)
                edgeloss3 = grad_loss(preds1, preds2)
                lossval = F.l1_loss(preds1, preds2)
                clip_score1 = clip_loss(preds1)
                clip_score2 = clip_loss(preds2)
                # ✅ 创建 L_exp 的时候就传好参数：
                Lloss2 = L_exp(patch_size=16, mean_val=0.6).to(device)
                # Lloss3 = L_TV()
                # Lloss4 = L_color()
                # L11 = Lloss1(img1,input1).mean()
                # criterion = BCESIMIoULoss(alpha=1.0, beta=1.0, gamma=1.0, weights=[1.0, 0.8, 0.6, 0.4])
                # losssBCE = criterion(s_list1, s_map1)
                L21 = Lloss2(input1).mean()
                L22 = Lloss2(input2).mean()
                # 经过模型增强后的图像
                # === 总损失分支1和分支2 ===
                total_loss1 = l_pix1 + edgeloss1 + 0.1*L21 +0.05*clip_score1
                total_loss2 = l_pix2 + edgeloss2 + 0.1*L22 +0.05*clip_score2
                # 加权总损失
                total_loss = total_loss1 + total_loss2 +lossval + edgeloss3
                loss_dict['clip_score1'] = clip_score1.detach().item()
                loss_dict['l_pix1'] = l_pix1.detach().item()
                # loss_dict['edgeloss1'] = edgeloss1.mean()  # 立即压成标量
                # del edgeloss1
                loss_dict['edgeloss1'] = edgeloss1.detach().item()
                loss_dict['edgeloss2'] = edgeloss2.detach().item()
                loss_dict['edgeloss3'] = edgeloss3.detach().item()
                loss_dict['lossval'] = lossval.detach().item()
                loss_dict['total_loss'] = total_loss.detach()
                total_loss.backward()

                # 如果启用梯度裁剪
                if self.opt['train']['use_grad_clip']:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

                # 执行优化器步骤
                self.optimizer_g.step()

                # 清零梯度
                self.optimizer_g.zero_grad()

                # 更新 EMA（如果需要）
                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)

                # 打印日志或其他操作
                self.log_dict = self.reduce_loss_dict(loss_dict)
            # 清零梯度
            self.optimizer_g.zero_grad()
    # 单分支
    # def pad_test(self, window_size):
    #     scale = self.opt.get('scale', 1)
    #     mod_pad_h, mod_pad_w = 0, 0
    #     _, _, h, w = self.lq.size()
    #
    #     # 计算需要填充的高度和宽度
    #     if h % window_size != 0:
    #         mod_pad_h = window_size - h % window_size
    #     if w % window_size != 0:
    #         mod_pad_w = window_size - w % window_size
    #
    #     # 使用反射填充图像
    #     img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #     self.nonpad_test(img)
    #
    #     # 解包 self.output 如果它是元组
    #     if isinstance(self.output, tuple):
    #         self.output = self.output[0]  # 获取第一个张量部分
    #
    #     # 对 output 做裁剪，去掉填充部分
    #     _, _, h, w = self.output.size()
    #     self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
    #
    # def nonpad_test(self, img=None):
    #     if img is None:
    #         img = self.lq  # 默认使用 `self.lq` 作为输入图像
    #
    #     # 将输入图像转换为 float32 类型
    #     img = img.to(torch.float32)
    #
    #     # 使用 ema 网络进行推理
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             # 仅传递图像给网络
    #             pred = self.net_g_ema(img)
    #         if isinstance(pred, list):
    #             pred = pred[-1]
    #         # 只保留增强图的输出
    #         if isinstance(pred, tuple):
    #             self.output = pred[0]  # 增强图像
    #         else:
    #             self.output = pred  # 增强图像
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             # 仅传递图像给网络
    #             pred = self.net_g(img)
    #         if isinstance(pred, list):
    #             pred = pred[-1]
    #         # 只保留增强图的输出
    #         if isinstance(pred, tuple):
    #             self.output = pred[0]  # 增强图像
    #         else:
    #             self.output = pred  # 增强图像
    #
    #     self.net_g.train()  # 恢复训练模式

    # 双分支
    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq1.size()

        # 计算填充的高度和宽度
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size

        # 反射填充
        img = F.pad(self.lq1, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        # 进行无填充测试
        self.nonpad_test(img)

        # 确保 `self.output` 只取第一张结果
        if isinstance(self.output, (tuple, list)):
            self.output = self.output[0]

        # 进行裁剪，去掉填充区域
        _, _, h_out, w_out = self.output.size()
        self.output = self.output[:, :, 0:h_out - mod_pad_h * scale, 0:w_out - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq1

        img = img.to(torch.float32)

        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()

        with torch.no_grad():
            # 🔥 直接传 `img`，让 `forward()` 自己决定跑单分支
            pred = net(img)

        # 🔥 适配返回结果，不管是 tuple 还是 list，都取第一个
        if isinstance(pred, (tuple, list)):
            self.output = pred[0]
        else:
            self.output = pred

        net.train()  # 恢复训练模式

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    # 单分支
    # def nondist_validation(self, dataloader, current_iter, tb_logger,
    #                        save_img, rgb2bgr, use_image):
    #     dataset_name = dataloader.dataset.opt['name']
    #     with_metrics = self.opt['val'].get('metrics') is not None
    #     if with_metrics:
    #         self.metric_results = {
    #             metric: 0
    #             for metric in self.opt['val']['metrics'].keys()
    #         }
    #     # pbar = tqdm(total=len(dataloader), unit='image')
    #
    #     window_size = self.opt['val'].get('window_size', 0)
    #
    #     if window_size:
    #         test = partial(self.pad_test, window_size)
    #     else:
    #         test = self.nonpad_test
    #
    #     cnt = 0
    #
    #     for idx, val_data in enumerate(dataloader):
    #         img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
    #         self.feed_data(val_data)
    #         test()
    #
    #         visuals = self.get_current_visuals()
    #         sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
    #         if 'gt' in visuals:
    #             gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
    #             del self.gt
    #
    #         # tentative for out of GPU memory
    #         del self.lq
    #         del self.output
    #         torch.cuda.empty_cache()
    #
    #         if save_img:
    #
    #             if self.opt['is_train']:
    #
    #                 save_img_path = osp.join(self.opt['path']['visualization'],
    #                                          img_name,
    #                                          f'{img_name}_{current_iter}.png')
    #
    #                 save_gt_img_path = osp.join(self.opt['path']['visualization'],
    #                                             img_name,
    #                                             f'{img_name}_{current_iter}_gt.png')
    #             else:
    #
    #                 save_img_path = osp.join(
    #                     self.opt['path']['visualization'], dataset_name,
    #                     f'{img_name}.png')
    #                 save_gt_img_path = osp.join(
    #                     self.opt['path']['visualization'], dataset_name,
    #                     f'{img_name}_gt.png')
    #
    #             imwrite(sr_img, save_img_path)
    #             imwrite(gt_img, save_gt_img_path)
    #
    #         if with_metrics:
    #             # calculate metrics
    #             opt_metric = deepcopy(self.opt['val']['metrics'])
    #             if use_image:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(sr_img, gt_img, **opt_)
    #             else:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
    #
    #         cnt += 1
    #
    #     current_metric = 0.
    #     if with_metrics:
    #         for metric in self.metric_results.keys():
    #             self.metric_results[metric] /= cnt
    #             current_metric = self.metric_results[metric]
    #
    #         self._log_validation_metric_values(current_iter, dataset_name,
    #                                            tb_logger)
    #     return current_metric

    # 双分支
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq1_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
                # if 'gt' in visuals:
                #     # 🚀 先获取 `sr_img` 目标尺寸
                #     target_size = visuals['result'].shape[2:]
                #
                #     # 🛠 确保 `self.gt` 是 4D (B, C, H, W)，然后调整尺寸
                #     if self.gt.dim() == 4 and self.gt.shape[2:] != target_size:
                #         self.gt = F.interpolate(self.gt, size=target_size, mode='bilinear', align_corners=False)
                #
                #     # 🔥 转换成 `gt_img`
                #     gt_img = tensor2img([self.gt], rgb2bgr=rgb2bgr)

                # 💣 释放显存，防止 OOM

                torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # 深拷贝 metrics 配置
                opt_metric = deepcopy(self.opt['val']['metrics'])

                for name, opt_ in opt_metric.items():
                    metric_type = opt_.get('type', 'calculate_psnr')  # 确保默认值

                    # 过滤掉 `type` 参数，避免 `TypeError`
                    valid_opt = {k: v for k, v in opt_.items() if k != 'type'}

                    if use_image:
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **valid_opt)
                    else:
                        self.metric_results[name] += getattr(metric_module, metric_type)(
                            visuals['result'], visuals['gt'], **valid_opt
                        )

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    # 教师/学生
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq1'] = self.lq1.detach().cpu()
        out_dict['lq2'] = self.lq2.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
