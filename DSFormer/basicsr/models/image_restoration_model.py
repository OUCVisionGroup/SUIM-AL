
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class IlluminationSmoothingGradientLoss(nn.Module):
#     def __init__(self, kernel_size=3, sigma=1.0):
#         super(IlluminationSmoothingGradientLoss, self).__init__()
#
#         # 创建高斯核，进行平滑操作
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#
#         # 计算高斯核
#         self.gaussian_kernel = self.get_gaussian_kernel(kernel_size, sigma)
#
#     def get_gaussian_kernel(self, kernel_size, sigma):
#         """生成高斯核"""
#         # 创建一个标准正态分布
#         kernel = torch.arange(0, kernel_size, dtype=torch.float32) - (kernel_size - 1) // 2
#         kernel = torch.exp(-(kernel**2) / (2 * sigma**2))
#         kernel = kernel / kernel.sum()  # 归一化
#         kernel = kernel.view(1, -1).repeat(kernel_size, 1)  # 创建二维高斯核
#
#         return kernel.unsqueeze(0).unsqueeze(0)
#
#     def forward(self, pred, gt):
#         """
#         计算预测图像和真实图像的梯度损失，同时进行光照平滑
#         pred: 预测图像 (B, C, H, W)
#         gt: 真实图像 (B, C, H, W)
#         """
#         # 计算图像的梯度
#         grad_pred_x, grad_pred_y = self.compute_gradients(pred)
#         grad_gt_x, grad_gt_y = self.compute_gradients(gt)
#
#         # 平滑梯度
#         grad_pred_x = self.smooth(grad_pred_x)
#         grad_pred_y = self.smooth(grad_pred_y)
#         grad_gt_x = self.smooth(grad_gt_x)
#         grad_gt_y = self.smooth(grad_gt_y)
#
#         # 计算梯度损失（L2损失）
#         grad_loss_x = F.mse_loss(grad_pred_x, grad_gt_x)
#         grad_loss_y = F.mse_loss(grad_pred_y, grad_gt_y)
#
#         # 返回损失
#         return grad_loss_x + grad_loss_y
#
#     def compute_gradients(self, img):
#         """计算图像的梯度"""
#         # 使用 Sobel 算子计算梯度
#         sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
#             0)
#         sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
#             0)
#
#         sobel_kernel_x = sobel_kernel_x.repeat(3, 1, 1, 1)  # 为每个通道创建sobel核
#         sobel_kernel_y = sobel_kernel_y.repeat(3, 1, 1, 1)
#
#         # 确保sobel核的数据类型与输入相同
#         sobel_kernel_x = sobel_kernel_x.to(img.device).to(img.dtype)
#         sobel_kernel_y = sobel_kernel_y.to(img.device).to(img.dtype)
#
#         grad_x = F.conv2d(img, sobel_kernel_x, padding=1, groups=img.shape[1])
#         grad_y = F.conv2d(img, sobel_kernel_y, padding=1, groups=img.shape[1])
#
#         return grad_x, grad_y
#
#     def smooth(self, grad):
#         # 扩展卷积核，使得它的通道数与grad的通道数相同
#         kernel = self.gaussian_kernel.repeat(grad.shape[1], 1, 1, 1)  # 将卷积核复制到每个通道
#
#         # 确保卷积核和输入的类型一致
#         kernel = kernel.to(grad.dtype)  # 转换卷积核的类型与grad一致
#
#         # 使用扩展后的卷积核进行卷积操作
#         return F.conv2d(grad, kernel, padding=self.kernel_size // 2, groups=grad.shape[1])
#
#
# def get_ab_loss(pred, gt):
#     # 将图像乘以255并转换为uint8
#     pred = (pred * 255).to('cpu', dtype=torch.uint8)
#     pred = pred.numpy()
#     gt = (gt * 255).to('cpu', dtype=torch.uint8)
#     gt = gt.numpy()
#
#     ab_loss = 0.0
#
#     for i in range(pred.shape[0]):
#         # 获取每张图片的a、b通道
#         pred_temp = pred[i].transpose((1, 2, 0))  # 转换为(H, W, C)
#         pred_temp = cv2.cvtColor(pred_temp, cv2.COLOR_RGB2LAB)
#         pred_a, pred_b = pred_temp[:, :, 1], pred_temp[:, :, 2]
#
#         gt_temp = gt[i].transpose((1, 2, 0))
#         gt_temp = cv2.cvtColor(gt_temp, cv2.COLOR_RGB2LAB)
#         gt_a, gt_b = gt_temp[:, :, 1], gt_temp[:, :, 2]
#
#         # 计算a通道和b通道的均值和标准差
#         pred_a_mean, pred_a_std = cv2.meanStdDev(pred_a)
#         pred_b_mean, pred_b_std = cv2.meanStdDev(pred_b)
#         gt_a_mean, gt_a_std = cv2.meanStdDev(gt_a)
#         gt_b_mean, gt_b_std = cv2.meanStdDev(gt_b)
#
#         # 提取每个通道的均值和标准差
#         pred_a_mean, pred_a_std = pred_a_mean.squeeze(), pred_a_std.squeeze()
#         pred_b_mean, pred_b_std = pred_b_mean.squeeze(), pred_b_std.squeeze()
#         gt_a_mean, gt_a_std = gt_a_mean.squeeze(), gt_a_std.squeeze()
#         gt_b_mean, gt_b_std = gt_b_mean.squeeze(), gt_b_std.squeeze()
#
#         # 计算每个通道的误差并累加
#         errors_a_mean = np.abs(pred_a_mean - gt_a_mean)
#         errors_a_std = np.abs(pred_a_std - gt_a_std)
#         errors_b_mean = np.abs(pred_b_mean - gt_b_mean)
#         errors_b_std = np.abs(pred_b_std - gt_b_std)
#
#         ab_loss += errors_a_mean + errors_a_std + errors_b_mean + errors_b_std
#
#     return ab_loss
# class Gradient_Difference_Loss(nn.Module):
#     def __init__(self, alpha=1, chans=3, cuda=True):
#         super(Gradient_Difference_Loss, self).__init__()
#         self.alpha = alpha
#         self.chans = chans
#         Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#         SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#         SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
#         self.Kx = torch.tensor(SobelX, dtype=torch.float32, device='cuda').expand(self.chans, 1, 3, 3)
#         self.Ky = torch.tensor(SobelY, dtype=torch.float32, device='cuda').expand(self.chans, 1, 3, 3)
#
#     def get_gradients(self, im):
#         gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
#         gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
#         return gx, gy
#
#     def forward(self, pred, true):
#         # get graduent of pred and true
#         gradX_true, gradY_true = self.get_gradients(true)
#         grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
#         gradX_pred, gradY_pred = self.get_gradients(pred)
#         grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
#         # compute and return GDL
#         return 0.5 * torch.mean((grad_true - grad_pred_a) ** 2)
#
#
# def edge_aware_loss(pred, gt):
#     B, C, H, W = pred.shape
#
#     # 定义 Sobel 卷积核
#     sobel_kernel_x = torch.tensor([[[[-1, 1]]]], dtype=torch.float32, device=pred.device)
#     sobel_kernel_y = torch.tensor([[[[-1], [1]]]], dtype=torch.float32, device=pred.device)
#
#     # 将通道数扩展到输入通道数
#     sobel_kernel_x = sobel_kernel_x.expand(C, 1, 1, 2)
#     sobel_kernel_y = sobel_kernel_y.expand(C, 1, 2, 1)
#
#     # 计算预测和真实图像的梯度
#     grad_pred_x = F.conv2d(pred, sobel_kernel_x, padding=(0, 1), groups=C)
#     grad_pred_y = F.conv2d(pred, sobel_kernel_y, padding=(1, 0), groups=C)
#
#     grad_gt_x = F.conv2d(gt, sobel_kernel_x, padding=(0, 1), groups=C)
#     grad_gt_y = F.conv2d(gt, sobel_kernel_y, padding=(1, 0), groups=C)
#
#     # 计算损失
#     loss = F.l1_loss(grad_pred_x, grad_gt_x) + F.l1_loss(grad_pred_y, grad_gt_y)
#     return loss
#
# def ssim_loss(pred, target, window_size=11, channel=1, size_average=True):
#     def gaussian(window_size, sigma):
#         _2d = torch.ones(window_size, window_size)
#         gauss = torch.exp(-torch.square(torch.arange(window_size) - (window_size // 2)).float() / (2 * sigma ** 2))
#         kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
#         return kernel / kernel.sum()
#
#     def ssim(img1, img2, window, window_size, channel, size_average=True):
#         mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#         mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2
#         sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=channel) - mu2_sq
#         sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#
#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2
#
#         numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
#         denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#         ssim_map = numerator / denominator
#         return ssim_map.mean() if size_average else ssim_map
#
#     window = gaussian(window_size, 1.5).to(pred.device)
#     window = window.unsqueeze(0).unsqueeze(0)
#     return 1 - ssim(pred, target, window, window_size, channel, size_average)
#
# class GrayscaleWorldLoss(nn.Module):
#     def __init__(self):
#         super(GrayscaleWorldLoss, self).__init__()
#
#     def forward(self, x):
#         """
#         计算灰度世界一致性损失
#         :param x: 输入的图像，形状为 (batch_size, 3, H, W)
#         :return: 灰度世界损失
#         """
#         # 分离出RGB三个通道
#         r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
#
#         # 计算每个通道的平均值
#         mean_r = torch.mean(r)
#         mean_g = torch.mean(g)
#         mean_b = torch.mean(b)
#
#         # 计算颜色一致性损失
#         loss = torch.abs(mean_r - mean_g) + torch.abs(mean_g - mean_b)
#
#         return loss
# class CharbonnierLoss(nn.Module):
#     """Charbonnier Loss (平滑的L1损失)"""
#
#     def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
#         super(CharbonnierLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.reduction = reduction
#         self.eps = eps
#
#     def forward(self, x, y):
#         # 计算差值
#         diff = x - y
#         # Charbonnier 损失公式
#         loss = torch.sqrt(diff * diff + self.eps * self.eps)
#
#         # 根据选择的归约方式来处理
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             pass  # 不进行归约，直接返回每个元素的损失
#
#         # 应用损失加权
#         loss = loss * self.loss_weight
#
#         return loss
#
# # 计算一维的高斯分布向量
# import torch
# import math
#
# def gaussian(window_size, sigma):
#     # 确保 window_size 是标量整数
#     if isinstance(window_size, torch.Tensor):
#         if window_size.numel() != 1:
#             raise ValueError(f"window_size should be a scalar, but got {window_size.shape}")
#         window_size = window_size.item()
#     window_size = int(window_size)
#
#     gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
#                           for x in range(window_size)], dtype=torch.float32)
#     return gauss
#
#
# def create_window(window_size, channel=1):
#     if not isinstance(window_size, int):
#         raise TypeError(f"Expected int for window_size, but got {type(window_size)}")
#
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     window = _1D_window @ _1D_window.T
#     window = window / window.sum()
#     return window.expand(channel, 1, window_size, window_size)
#
# def ssim(img1, img2, window_size=12, window=None, size_average=True, full=False, val_range=None):
#     # 确定像素范围
#     if val_range is None:
#         max_val = 255 if torch.max(img1) > 128 else 1
#         min_val = -1 if torch.min(img1) < -0.5 else 0
#         L = max_val - min_val
#     else:
#         L = val_range
#
#     padd = window_size // 2
#     (_, channel, height, width) = img1.size()
#
#     # 创建高斯窗口
#     if window is None or window.size(-1) != window_size:
#         real_size = min(window_size, height, width)
#         window = create_window(real_size, channel=channel).to(img1.device)
#
#     # 计算均值
#     mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     # 计算方差与协方差
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
#
#     # SSIM公式中的常量
#     C1 = (0.01 * L) ** 2
#     C2 = (0.03 * L) ** 2
#
#     # 对比敏感度
#     v1 = 2.0 * sigma12 + C2
#     v2 = sigma1_sq + sigma2_sq + C2
#     cs = torch.mean(v1 / v2)
#
#     # 计算SSIM
#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
#
#     # 返回结果
#     ret = ssim_map.mean() if size_average else ssim_map.mean([1, 2, 3])
#
#     return (ret, cs) if full else ret
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
# # Classes to re-use window
# class SSIM(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True, val_range=None):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.val_range = val_range
#
#         # Assume 1 channel for SSIM
#         self.channel = 1
#         self.window = create_window(window_size)
#
#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#
#         if channel == self.channel and self.window.dtype == img1.dtype:
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
#             self.window = window
#             self.channel = channel
#
#         return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
#
# # def retinex_decomposition(img):
# #     # 如果 img 是 PyTorch 张量，转换为 NumPy 数组
# #     if isinstance(img, torch.Tensor):
# #         img = img.detach().cpu().numpy()
# #
# #     # 如果图像范围在 [0, 1] 之间，转换为 [0, 255]
# #     if img.max() <= 1.0:
# #         img = (img * 255).astype(np.uint8)
# #
# #     # 如果图像是 (C, H, W) 格式，转换为 (H, W, C)
# #     if img.ndim == 3 and img.shape[0] == 3:
# #         img = np.transpose(img, (1, 2, 0))  # 转换为 (H, W, C) 格式
# #
# #     # 如果是四维张量 (B, C, H, W)
# #     if img.ndim == 4:
# #         B, C, H, W = img.shape  # 解包为批次、通道、高度和宽度
# #         img = img[0]  # 假设我们只处理批次中的第一个图像
# #     else:
# #         H, W, C = img.shape  # 获取图像的高度、宽度和通道数
# #
# #     kernel_size = 5  # 示例的卷积核大小
# #     L_blur = np.zeros_like(img, dtype=np.uint8)  # 初始化空数组用于保存模糊图像
# #
# #     # 对每个通道应用高斯模糊并保存单通道光照图
# #     # 这里使用图像的平均值来创建光照图
# #     L_blur = np.mean(img, axis=2)  # 计算各通道的均值，得到单通道的光照图
# #
# #     # 转换回 PyTorch 张量，并保持形状为 (1, 1, H, W)
# #     illu = torch.tensor(L_blur, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 保持形状为 (1, 1, H, W)
# #
# #     return illu
#
# def gradient(input_tensor, direction):
#     # 如果 input_tensor 是 NumPy 数组，转换为 PyTorch 张量
#     if input_tensor.shape[-2] < 2 or input_tensor.shape[-1] < 2:
#         raise ValueError("Input tensor dimensions are too small for the gradient kernel.")
#
#     # 定义平滑卷积核
#     smooth_kernel_x = torch.tensor([[0, 1]], dtype=torch.float32).view(1, 1, 1, 2)
#     smooth_kernel_y = torch.tensor([[1], [-1]], dtype=torch.float32).view(1, 1, 2, 1)
#
#     # 根据方向选择合适的卷积核
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     else:
#         raise ValueError("Direction must be 'x' or 'y'")
#
#     # 确保 kernel 在相同的设备上（与 input_tensor 相同的设备）
#     kernel = kernel.to(input_tensor.device)
#
#     # 使用 conv2d 计算梯度
#     # 直接对 input_tensor 进行卷积，而不需要增加批次维度
#     gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=0))  # (C, H, W)
#
#     # 归一化梯度
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)
#
#     return grad_norm
#
#
# def rgb_to_lab(image):
#     """将RGB图像转换为Lab颜色空间，返回L通道"""
#     # 转换为 NumPy 数组以便使用 OpenCV 函数
#     image = image.detach().cpu().numpy()
#     # 如果图像范围是 [0, 1]，转换为 [0, 255]
#     if image.max() <= 1.0:
#         image = (image * 255).astype(np.uint8)
#
#     # 将图像从 RGB 转换为 BGR (OpenCV 使用 BGR 格式)
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # 转换为 LAB 色彩空间
#     lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
#
#     # 提取 L 通道
#     L_channel = lab_image[:, :, 0]
#
#     return L_channel
#
#
# def local_consistency_loss_L_channel(enhanced_image, mask, window_size=3):
#     """
#     计算L通道上的局部一致性损失，用于约束增强图像中的极端黑暗区域。
#
#     Args:
#         enhanced_image (torch.Tensor): 增强后的图像 (B, C, H, W)
#         mask (torch.Tensor): 极端黑暗区域的掩码 (B, 1, H, W)
#         window_size (int): 局部窗口大小
#
#     Returns:
#         loss (torch.Tensor): 局部一致性损失
#     """
#     pad = window_size // 2
#     enhanced_padded = F.pad(enhanced_image, (pad, pad, pad, pad), mode='reflect')
#
#     # 将 RGB 图像转换为 L 通道
#     L_channel = rgb_to_lab(enhanced_image)  # 计算增强图像的 L 通道
#
#     # 计算局部窗口内的 L 通道均值
#     local_mean = F.avg_pool2d(enhanced_padded, kernel_size=window_size, stride=1, padding=0)
#
#     # 只对掩码指定区域计算一致性损失
#     consistency_loss = F.l1_loss(L_channel * mask, local_mean * mask)
#
#     return consistency_loss
#
# # class SmoothLoss(nn.Module):
# #     """ Illumination smoothness loss """
# #
# #     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
# #         super(SmoothLoss, self).__init__()
# #         self.loss_weight = loss_weight
# #         self.eps = eps
# #         self.reduction = reduction
# #
# #     def forward(self, illu, img):
# #         # illu: 预测的光照图 b×c×h×w
# #         # img: 输入图像 b×c×h×w
# #
# #         # 计算光照图和图像的梯度
# #         illu_gradient_x = gradient(illu, "x")
# #         img_gradient_x = gradient(img, "x")
# #         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, self.eps)))
# #
# #         illu_gradient_y = gradient(illu, "y")
# #         img_gradient_y = gradient(img, "y")
# #         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, self.eps)))
# #
# #         # 计算总损失
# #         loss = torch.mean(x_loss + y_loss) * self.loss_weight
# #
# #         return loss
#
# class LabContrastiveLoss(nn.Module):
#     def __init__(self, lambd=0.005, temperature=0.001):
#         super(LabContrastiveLoss, self).__init__()
#         self.lambd = lambd  # Regularization strength for off-diagonal loss
#         self.temperature = temperature  # Temperature scaling for contrastive loss
#
#     def rgb_to_lab(self, rgb):
#         """Convert RGB to Lab color space and normalize channels."""
#         rgb = rgb.permute(0, 2, 3, 1) / 255.0  # Convert from [B, C, H, W] to [B, H, W, C]
#         B, H, W, C = rgb.shape
#
#         # Convert RGB to Lab using OpenCV (CPU-based operation)
#         lab_images = []
#         for i in range(B):
#             lab_image = cv2.cvtColor(rgb[i].detach().cpu().numpy(), cv2.COLOR_RGB2Lab)  # RGB to Lab
#             lab_images.append(lab_image)
#
#         lab = np.stack(lab_images, axis=0)  # Shape: [B, H, W, C]
#         lab = torch.tensor(lab, dtype=torch.float32).permute(0, 3, 1, 2).cuda()  # Convert back to [B, C, H, W]
#
#         # Normalize L channel to [0, 1] and A, B channels to [-1, 1]
#         lab[:, 0, :, :] = lab[:, 0, :, :] / 100.0  # Normalize L channel to [0, 1]
#         lab[:, 1:, :, :] = lab[:, 1:, :, :] / 128.0  # Normalize A and B channels to [-1, 1]
#
#         return lab
#
#     def compute_lab_stats(self, dataset):
#         """Compute the mean and std of L channel for the entire dataset."""
#         L_means = []
#         L_stds = []
#
#         for img in dataset:
#             lab = self.rgb_to_lab(img)  # Convert to Lab space
#             L_channel = lab[:, 0, :, :]  # Extract L channel
#
#             # Compute mean and std for each image
#             L_mean = torch.mean(L_channel, dim=(1, 2))
#             L_std = torch.std(L_channel, dim=(1, 2))
#             L_means.append(L_mean)
#             L_stds.append(L_std)
#
#         # Compute mean and std for the entire dataset
#         L_mean = torch.mean(torch.stack(L_means), dim=0)
#         L_std = torch.mean(torch.stack(L_stds), dim=0)
#
#         return L_mean, L_std
#
#     def compute_similarity(self, gen_mean, gen_std, target_mean, target_std):
#         """Compute cosine similarity between generated and target statistics (mean and std)."""
#         # Normalize mean and std
#         gen_mean = F.normalize(gen_mean.unsqueeze(1), p=2, dim=1)
#         gen_std = F.normalize(gen_std.unsqueeze(1), p=2, dim=1)
#         target_mean = F.normalize(target_mean.unsqueeze(1), p=2, dim=1)
#         target_std = F.normalize(target_std.unsqueeze(1), p=2, dim=1)
#
#         # Compute cosine similarity
#         similarity_mean = torch.matmul(gen_mean, target_mean.T) / self.temperature
#         similarity_std = torch.matmul(gen_std, target_std.T) / self.temperature
#
#         return similarity_mean, similarity_std
#
#     def forward(self, generated_img, target_img, dataset):
#         """Compute the contrastive loss based on Lab channel means and stds."""
#         # Compute stats (mean and std) for the entire dataset
#         target_mean, target_std = self.compute_lab_stats(dataset)
#
#         # Compute stats for the generated image
#         generated_lab = self.rgb_to_lab(generated_img)
#         gen_mean = torch.mean(generated_lab[:, 0, :, :], dim=(2, 3))
#         gen_std = torch.std(generated_lab[:, 0, :, :], dim=(2, 3))
#
#         # Compute similarity for mean and std (L)
#         similarity_mean_L, similarity_std_L = self.compute_similarity(gen_mean, gen_std, target_mean, target_std)
#
#         # Diagonal loss (on-diagonal similarity should be 1)
#         on_diag_L_mean = torch.diagonal(similarity_mean_L).add_(-1).pow_(2).sum()
#         on_diag_L_std = torch.diagonal(similarity_std_L).add_(-1).pow_(2).sum()
#
#         # Off-diagonal loss (minimize cross-correlation between different instances)
#         off_diag_L_mean = similarity_mean_L[
#             torch.tril_indices(similarity_mean_L.size(0), similarity_mean_L.size(1), -1)].pow_(2).sum()
#         off_diag_L_std = similarity_std_L[
#             torch.tril_indices(similarity_std_L.size(0), similarity_std_L.size(1), -1)].pow_(2).sum()
#
#         # Combine all loss terms
#         loss = (on_diag_L_mean + on_diag_L_std) + self.lambd * (off_diag_L_mean + off_diag_L_std)
#
#         return loss
#
#
# def split_into_patches(image, patch_size=32, num_patches=4):
#     """假设图像被裁剪成num_patches个小块"""
#     patches = []
#     B, C, H, W = image.shape
#     for _ in range(num_patches):
#         i = torch.randint(0, H - patch_size, (1,)).item()
#         j = torch.randint(0, W - patch_size, (1,)).item()
#         patch = image[:, :, i:i+patch_size, j:j+patch_size]
#         patches.append(patch)
#     return patches
#
#
# def gradient(input_tensor, direction):
#     # **🔥 统一数据类型**
#     dtype = input_tensor.dtype  # **确保 kernel 和 input_tensor 一样的 dtype**
#
#     # Gradient kernel in x-direction
#     smooth_kernel_x = torch.tensor([[[[0, 0], [-1, 1]]]] * input_tensor.shape[1], dtype=dtype,
#                                    device=input_tensor.device)
#
#     # Gradient kernel in y-direction
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)
#
#     kernel = smooth_kernel_x if direction == "x" else smooth_kernel_y
#
#     # **🔥 计算梯度**
#     gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1, groups=input_tensor.shape[1]))
#
#     # **🔥 归一化处理**
#     grad_min = gradient_orig.amin(dim=[1, 2, 3], keepdim=True)  # **`amin` 更高效**
#     grad_max = gradient_orig.amax(dim=[1, 2, 3], keepdim=True)
#
#     grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 1e-4)
#
#     return grad_norm
#
# class MultualLoss(nn.Module):
#     """ Multual Consistency"""
#
#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()
#
#         self.loss_weight = loss_weight
#         self.reduction = reduction
#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")
#
#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)
#
#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss
#
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# class SpectrumLoss(nn.Module):
#     def __init__(self, lambda_high=1.0, lambda_low=1.0, lambda_smooth=0.1):
#         super(SpectrumLoss, self).__init__()
#         self.lambda_high = lambda_high
#         self.lambda_low = lambda_low
#         self.lambda_smooth = lambda_smooth
#
#     def forward(self, input_image, enhanced_image):
#         # 确保输入图像和增强图像在同一设备
#         device = input_image.device
#
#         # 计算频谱
#         fft_input = torch.fft.fft2(input_image, norm='ortho').to(device)
#         fft_enhanced = torch.fft.fft2(enhanced_image, norm='ortho').to(device)
#
#         # 获取频谱幅度
#         amp_input = torch.abs(fft_input).to(device)
#         amp_enhanced = torch.abs(fft_enhanced).to(device)
#
#         # 分割高频和低频部分
#         h, w = amp_input.shape[-2:]
#         center_h, center_w = h // 2, w // 2
#         radius = min(h, w) // 4  # 半径，控制高频和低频区域
#
#         # 创建距离掩码并移动到相同设备
#         y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
#         distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
#
#         high_freq_mask = (distance > radius).float().to(device)
#         low_freq_mask = (distance <= radius).float().to(device)
#
#         # 高频增强损失
#         high_loss = torch.mean(high_freq_mask * (amp_input - amp_enhanced) ** 2)
#
#         # 低频保留损失
#         low_loss = torch.mean(low_freq_mask * (amp_input - amp_enhanced) ** 2)
#
#         # 频谱平滑损失
#         grad_x = torch.abs(amp_enhanced[:, :, 1:, :] - amp_enhanced[:, :, :-1, :])
#         grad_y = torch.abs(amp_enhanced[:, :, :, 1:] - amp_enhanced[:, :, :, :-1])
#         smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)
#
#         # 总损失
#         total_loss = self.lambda_high * high_loss + self.lambda_low * low_loss + self.lambda_smooth * smooth_loss
#
#         return total_loss
#
# import torch
# import torch.nn as nn
# import torch.fft
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.fft
#
# class FourierHighBoostLoss(nn.Module):
#     def __init__(self, radius_ratio=0.35, lambda_high=2.0, lambda_low=0.5, lambda_contrast=0.5):
#         super().__init__()
#         self.radius_ratio = radius_ratio
#         self.lambda_high = lambda_high
#         self.lambda_low = lambda_low
#         self.lambda_contrast = lambda_contrast
#
#     def forward(self, enhanced, reference):
#         B, C, H, W = enhanced.shape
#
#         # 傅里叶变换
#         fft_enh = torch.fft.fft2(enhanced, norm='ortho')
#         fft_ref = torch.fft.fft2(reference, norm='ortho')
#
#         # 幅度谱
#         amp_diff = torch.abs(fft_enh) - torch.abs(fft_ref)
#
#         # 创建高/低频掩码
#         y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
#         center_h, center_w = H // 2, W // 2
#         radius = int(min(H, W) * self.radius_ratio)
#         dist_mask = ((y - center_h) ** 2 + (x - center_w) ** 2) <= radius ** 2
#         low_freq_mask = dist_mask.to(enhanced.device)
#         high_freq_mask = (~dist_mask).to(enhanced.device)
#
#         # 直接平均差异作为损失（无光照权重）
#         def freq_loss(mask, lambda_weight):
#             mask = mask[None, None, :, :].expand(B, C, H, W)
#             return lambda_weight * ((amp_diff ** 2) * mask).mean()
#
#         loss_high = freq_loss(high_freq_mask, self.lambda_high)
#         loss_low = freq_loss(low_freq_mask, self.lambda_low)
#
#         # 对比度惩罚，还是保留一下，这玩意儿干净利索
#         def contrast_loss(img1, img2):
#             mean1 = torch.mean(img1, dim=[2, 3], keepdim=True)
#             mean2 = torch.mean(img2, dim=[2, 3], keepdim=True)
#             contrast = torch.mean((img2 - mean2) ** 2 - (img1 - mean1) ** 2)
#             return F.relu(contrast)
#
#         loss_contrast = contrast_loss(enhanced, reference) * self.lambda_contrast
#
#         return loss_high + loss_low + loss_contrast
#
# def contrast_loss(img):
#     mean = torch.mean(img, dim=[2,3], keepdim=True)
#     contrast = torch.mean((img - mean) ** 2)
#     return -contrast  # 想让它更高，就最小化负对比度
#
# import torch.nn as nn
# import torch.nn.functional as F
# #图像域
# # class SpectrumLoss(nn.Module):
# #     def __init__(self, lambda_high=1.0, lambda_low=1.0, lambda_smooth=0.1):
# #         super(SpectrumLoss, self).__init__()
# #         self.lambda_high = lambda_high
# #         self.lambda_low = lambda_low
# #         self.lambda_smooth = lambda_smooth
# #
# #     def forward(self, input_images, enhanced_images):
# #         """
# #         input_images: 低光图像域 [b, c, h, w]
# #         enhanced_images: 增强图像域 [b, c, h, w]
# #
# #         返回: 总损失
# #         """
# #         device = input_images.device
# #
# #         # 计算输入图像的频谱
# #         fft_input = torch.fft.fft2(input_images, norm='ortho').to(device)
# #         fft_enhanced = torch.fft.fft2(enhanced_images, norm='ortho').to(device)
# #
# #         # 获取频谱幅度
# #         amp_input = torch.abs(fft_input)
# #         amp_enhanced = torch.abs(fft_enhanced)
# #
# #         # 获取图像的高度和宽度
# #         h, w = amp_input.shape[-2:]
# #         center_h, center_w = h // 2, w // 2
# #         radius = min(h, w) // 4  # 半径，控制高频和低频区域
# #
# #         # 创建距离掩码，分割高频和低频区域
# #         y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
# #         distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
# #
# #         high_freq_mask = (distance > radius).float().to(device)
# #         low_freq_mask = (distance <= radius).float().to(device)
# #
# #         # 高频增强损失（高频区域的幅度差异）
# #         high_loss = torch.mean(high_freq_mask * (amp_input - amp_enhanced) ** 2)
# #
# #         # 低频保留损失（低频区域的幅度差异）
# #         low_loss = torch.mean(low_freq_mask * (amp_input - amp_enhanced) ** 2)
# #
# #         # 频谱平滑损失（鼓励频谱的平滑变化）
# #         grad_x = torch.abs(amp_enhanced[:, :, 1:, :] - amp_enhanced[:, :, :-1, :])
# #         grad_y = torch.abs(amp_enhanced[:, :, :, 1:] - amp_enhanced[:, :, :, :-1])
# #         smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)
# #
# #         # 总损失（加权组合各个损失）
# #         total_loss = self.lambda_high * high_loss + self.lambda_low * low_loss + self.lambda_smooth * smooth_loss
# #
# #         return total_loss
#
# class SmoothLoss(nn.Module):
#     """Illumination smoothness"""
#
#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
#
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x = gradient(img, "x")
#
#         # Convert 0.01 to tensor for compatibility with torch.maximum
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, torch.tensor(0.01, dtype=img.dtype,
#                                                                                                  device=img.device))))
#
#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y = gradient(img, "y")
#
#         # Convert 0.01 to tensor for compatibility with torch.maximum
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, torch.tensor(0.01, dtype=img.dtype,
#                                                                                                  device=img.device))))
#
#         loss = torch.mean(x_loss + y_loss) * self.loss_weight
#
#         return loss
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class IlluminationConsistencyLoss(nn.Module):
#     def __init__(self, loss_weight=0.1, patch_size=32):
#         super(IlluminationConsistencyLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.patch_size = patch_size
#
#     def forward(self, illu):
#         b, c, h, w = illu.shape
#
#         # 将图像切割成多个patch
#         patches = illu.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
#
#         # 每个patch计算梯度
#         grad_x = gradient(patches, "x")
#         grad_y = gradient(patches, "y")
#
#         # 计算梯度差异
#         grad_diff_x = torch.var(grad_x, dim=2)
#         grad_diff_y = torch.var(grad_y, dim=2)
#
#         # 总损失
#         loss = torch.mean(grad_diff_x + grad_diff_y)
#
#         return loss * self.loss_weight
#
# import torch.nn as nn
#
# class IlluminationDiscriminator(nn.Module):
#     """判别器：判断增强图的光照是否和伪标签光照一致"""
#
#     def __init__(self, input_nc=6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
#         super(IlluminationDiscriminator, self).__init__()
#
#         # 🌟 判别器主干，基于 PatchGAN，但调整了下采样方式
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4  # 核大小
#         padw = 1  # padding
#         sequence = [nn.Conv2d(6, ndf, kernel_size=kw, stride=2, padding=padw),
#                     nn.LeakyReLU(0.2, True)]
#
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # 逐步增加通道数
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
#             nn.LeakyReLU(0.2, True)
#         ]
#         sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
#
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, enhanced, reference):
#         """输入增强图和伪标签光照，输出光照匹配程度"""
#         input = torch.cat([enhanced, reference], dim=1)  # 拼接输入
#         return self.model(input)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # class IlluminationLoss(nn.Module):
# #     """光照匹配损失（🔥 增强图自身局部一致性对抗损失）"""
# #
# #     def __init__(self, discriminator, lambda_adv=0.1, lambda_local=0.5):
# #         super(IlluminationLoss, self).__init__()
# #         self.discriminator = discriminator
# #         self.lambda_adv = lambda_adv
# #         self.lambda_local = lambda_local
# #         self.bce_loss = nn.BCEWithLogitsLoss()
# #
# #     def _local_patch_loss(self, enhanced):
# #         """🔥 **增强图 `enhanced` 局部光照一致性对抗损失**"""
# #         B, C, H, W = enhanced.size()
# #         size = 32 # 切成 `16x16` 小块
# #         Y, X = H // size, W // size
# #
# #         # 🚀 切割局部 patch（不对比 reference）
# #         enhanced_patches = enhanced.view(B, C, Y, size, X, size).permute(0, 2, 4, 1, 3, 5).reshape(B * Y * X, C, size, size)
# #
# #         # 🔥 `D` 判别增强图自身局部光照是否一致
# #         fake_pred = self.discriminator(enhanced_patches, enhanced_patches)
# #
# #         # 🚀 计算对抗损失（希望 `D` 觉得每个局部 patch 是真实的）
# #         fake_labels = torch.ones_like(fake_pred)  # 让 `D` 觉得所有增强图的 patch 是真实的
# #         local_loss = self.bce_loss(fake_pred, fake_labels)
# #
# #         return local_loss  # ✅ 对抗损失
# #
# #     def forward(self, enhanced, reference):
# #         """🔥 **最终光照损失**"""
# #         # **全局对抗损失**
# #         real_pred = self.discriminator(reference, reference)
# #         fake_pred = self.discriminator(enhanced, reference)
# #         real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
# #         fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
# #         adv_loss = real_loss + fake_loss  # ✅ 整体光照对抗损失
# #
# #         # **局部光照一致性对抗损失**
# #         local_loss = self._local_patch_loss(enhanced)
# #         print("real_pred mean:", real_pred.mean().item(), "std:", real_pred.std().item())
# #         print("fake_pred mean:", fake_pred.mean().item(), "std:", fake_pred.std().item())
# #
# #         # **最终损失**
# #         return self.lambda_adv * adv_loss + self.lambda_local * local_loss
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from torchvision import transforms
# from pytorch_msssim import ssim
#
# class UnsupervisedIlluminationLoss(nn.Module):
#     def __init__(self, lambda_bright=1.0, lambda_dark=1.0, lambda_illum=0.1, lambda_ssim=0.5, dark_threshold=0.5):
#         super(UnsupervisedIlluminationLoss, self).__init__()
#         self.lambda_bright = lambda_bright
#         self.lambda_dark = lambda_dark
#         self.lambda_illum = lambda_illum
#         self.lambda_ssim = lambda_ssim
#         self.dark_threshold = dark_threshold
#
#     def brightness_loss(self, img_in, img_out):
#         """鼓励整体亮度提升"""
#         return 1 - (img_out.mean() / (img_in.mean() + 1e-6))
#
#     def dark_region_loss(self, img_in, img_out):
#         """专门增强低光区域"""
#         mask = (img_in < self.dark_threshold).float()
#         return ((self.dark_threshold - img_out) ** 2 * mask).mean()
#
#     def illumination_consistency_loss(self, img_out):
#         """减少光照不均匀，确保亮度分布均匀"""
#         return torch.var(img_out)
#
#     def structure_preservation_loss(self, img_in, img_out):
#         """防止增强后图像失去结构信息"""
#         return 1 - ssim(img_in, img_out, data_range=1)
#
#     def forward(self, img_in, img_out):
#         """计算最终无监督损失"""
#         loss_bright = self.brightness_loss(img_in, img_out)
#         loss_dark = self.dark_region_loss(img_in, img_out)
#         loss_illum = self.illumination_consistency_loss(img_out)
#         loss_ssim = self.structure_preservation_loss(img_in, img_out)
#
#         return (self.lambda_bright * loss_bright +
#                 self.lambda_dark * loss_dark +
#                 self.lambda_illum * loss_illum +
#                 self.lambda_ssim * loss_ssim)
#
# # 示例：
# import functools
# from torch.nn.utils import spectral_norm as SpectralNorm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import clip
# class L_spa(nn.Module):
#
#     def __init__(self):
#         super(L_spa, self).__init__()
#         # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
#         kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
#         self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
#         self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
#         self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
#         self.pool = nn.AvgPool2d(4)
#
#     def forward(self, org, enhance):
#         b, c, h, w = org.shape
#
#         org_mean = torch.mean(org, 1, keepdim=True)
#         enhance_mean = torch.mean(enhance, 1, keepdim=True)
#
#         org_pool = self.pool(org_mean)
#         enhance_pool = self.pool(enhance_mean)
#
#         weight_diff = torch.max(
#             torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
#                                                               torch.FloatTensor([0]).cuda()),
#             torch.FloatTensor([0.5]).cuda())
#         E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)
#
#         D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
#         D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
#         D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
#         D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
#
#         D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
#         D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
#         D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
#         D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
#
#         D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
#         D_right = torch.pow(D_org_right - D_enhance_right, 2)
#         D_up = torch.pow(D_org_up - D_enhance_up, 2)
#         D_down = torch.pow(D_org_down - D_enhance_down, 2)
#         E = (D_left + D_right + D_up + D_down)
#         # E = 25*(D_left + D_right + D_up +D_down)
#         return E
# class L_TV(nn.Module):
#     def __init__(self, TVLoss_weight=1):
#         super(L_TV, self).__init__()
#         self.TVLoss_weight = TVLoss_weight
#
#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = (x.size()[2] - 1) * x.size()[3]
#         count_w = x.size()[2] * (x.size()[3] - 1)
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
# # class L_exp(nn.Module):
# #
# #     def __init__(self, patch_size, mean_val):
# #         super(L_exp, self).__init__()
# #         # print(1)
# #         self.pool = nn.AvgPool2d(patch_size)
# #         self.mean_val = mean_val
# #
# #     def forward(self, x):
# #         b, c, h, w = x.shape
# #         x = torch.mean(x, 1, keepdim=True)
# #         mean = self.pool(x)
# #
# #         d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
# #         return d
# # class L_exp(nn.Module):
# #     def __init__(self, patch_size=16, mean_val=0.6):
# #         super(L_exp, self).__init__()
# #         self.pool = nn.AvgPool2d(patch_size)
# #         self.mean_val = mean_val
# #
# #     def forward(self, x, light_prior):
# #         """
# #         x: [B, C, H, W] - 输入图像
# #         light_prior: [B, 1, H, W] - 光照先验图（白=亮，黑=暗）
# #         """
# #         x_gray = torch.mean(x, dim=1, keepdim=True)               # 灰度图
# #         local_mean = self.pool(x_gray)                            # 局部平均亮度
# #         light_prior_down = self.pool(light_prior)                # 光照图也下采样到一样的尺寸
# #
# #         # 🎯 权重：亮的地方小惩罚，暗的地方大惩罚
# #         weights = 1.0 - light_prior_down.clamp(0.0, 1.0)          # 归一化+反转，黑=1, 白=0
# #
# #         target = torch.full_like(local_mean, self.mean_val)       # 目标亮度
# #
# #         # 💥 权重加权曝光损失
# #         loss = torch.mean(weights * (local_mean - target) ** 2)
# #
# #         return loss
#
# class L_exp(nn.Module):
#     def __init__(self, patch_size=16, mean_val=0.6):
#         super(L_exp, self).__init__()
#         self.pool = nn.AvgPool2d(patch_size)
#         self.mean_val = mean_val
#
#     def forward(self, x, light_prior):
#         x_gray = torch.mean(x, dim=1, keepdim=True)
#         local_mean = self.pool(x_gray)
#         light_prior_down = self.pool(light_prior)
#
#         # 🌗 动态曝光因子：亮区1，暗区2
#         # 先归一化到 [0,1]
#         light_prior_down = light_prior_down.clamp(0, 1)
#         weight = torch.where(light_prior_down > 0.5, 1.0, 2.0)
#
#         # 🧠 和目标亮度对比
#         target = torch.full_like(local_mean, self.mean_val)
#         diff = (local_mean - target) ** 2
#
#         # 🔥 强制暗区曝光惩罚加倍
#         loss = torch.mean(weight * diff)
#         return loss
#
# class L_color(nn.Module):
#
#     def __init__(self):
#         super(L_color, self).__init__()
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         mean_rgb = torch.mean(x, [2, 3], keepdim=True)
#         mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
#         Drg = torch.pow(mr - mg, 2)
#         Drb = torch.pow(mr - mb, 2)
#         Dgb = torch.pow(mb - mg, 2)
#         k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
#
#         return k
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import clip
# class AdaptiveCLIPLoss(nn.Module):
#     def __init__(self, clip_model_path, lambda_teacher=0.2, update_freq=10, device="cuda"):
#         super(AdaptiveCLIPLoss, self).__init__()
#         self.device = device
#         self.lambda_teacher = lambda_teacher
#         self.update_freq = update_freq
#
#         # 加载并冻结CLIP模型
#         self.clip_model, _ = clip.load(clip_model_path, device=self.device)
#         self.clip_model.eval()
#
#         # 缓存机制：缓存 teacher 和 text 的特征
#         self.teacher_cache = None
#         self.cached_teacher_shape = None
#         self.text_cache = {}
#
#     def _get_text_features(self, text_prompt, batch_size):
#         if text_prompt not in self.text_cache:
#             with torch.no_grad():
#                 text_inputs = clip.tokenize([text_prompt] * batch_size).to(self.device)
#                 text_features = self.clip_model.encode_text(text_inputs)
#                 text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#                 self.text_cache[text_prompt] = text_features
#         return self.text_cache[text_prompt]
#
#     def forward(self, enhanced_img, teacher_img,
#                 text_prompt="The image is clear and bright, the illumination is uniform and smooth, the color is bright, and there is no artifact.",
#                 current_iter=0):
#         batch_size = enhanced_img.shape[0]
#
#         # 只在 update_freq 触发周期执行损失
#         if current_iter % self.update_freq != 0:
#             return torch.tensor(0.0, device=self.device, requires_grad=False)
#
#         # 🔒 教师图像特征缓存机制
#         with torch.no_grad():
#             if (self.teacher_cache is None) or (teacher_img.shape != self.cached_teacher_shape):
#                 teacher_features = self.clip_model.encode_image(teacher_img)
#                 teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
#                 self.teacher_cache = teacher_features
#                 self.cached_teacher_shape = teacher_img.shape
#             else:
#                 teacher_features = self.teacher_cache
#
#         # 📚 文本描述特征缓存机制
#         text_features = self._get_text_features(text_prompt, batch_size)
#
#         # 🧑‍🎓 学生增强图像特征
#         student_features = self.clip_model.encode_image(enhanced_img)
#         student_features = student_features / student_features.norm(dim=-1, keepdim=True)
#
#         # 🧠 相似度计算
#         sim_student_text = torch.cosine_similarity(student_features, text_features, dim=-1)
#         sim_teacher_text = torch.cosine_similarity(teacher_features, text_features, dim=-1)
#         sim_student_teacher = torch.cosine_similarity(student_features, teacher_features, dim=-1)
#         # print(sim_teacher_text.mean())
#         # print(sim_student_text.mean())
#         # 🎯 自适应引导项
#         delta = (sim_teacher_text - sim_student_text).clamp(min=0)  # 只要教师更好，就往上拉
#         adaptive_lambda = self.lambda_teacher * delta.detach()
#
#         contrast_loss = adaptive_lambda * (1 - sim_student_teacher)
#         clip_loss = F.relu(0.8 - sim_student_text).mean()
#
#         # 🍭 总损失
#         return clip_loss.mean() + contrast_loss.mean()
#
# class ContrastiveTextCLIPLoss(nn.Module):
#     def __init__(self, clip_model_path, device="cuda", update_freq=1):
#         super(ContrastiveTextCLIPLoss, self).__init__()
#         self.device = device
#         self.update_freq = update_freq
#         self.clip_model, _ = clip.load(clip_model_path, device=self.device)
#         self.clip_model.eval()
#         self.text_cache = {}
#
#     def _get_text_features(self, prompts, batch_size):
#         features = []
#         for prompt in prompts:
#             if prompt not in self.text_cache:
#                 with torch.no_grad():
#                     tokens = clip.tokenize([prompt] * batch_size).to(self.device)
#                     feat = self.clip_model.encode_text(tokens)
#                     feat = feat / feat.norm(dim=-1, keepdim=True)
#                     self.text_cache[prompt] = feat
#             features.append(self.text_cache[prompt])
#         return torch.stack(features, dim=1)  # (B, 2, D)
#
#     def forward(self, enhanced_img,
#                 prompts=["Uniform and bright images", "Non-uniform and dark images"],
#                 current_iter=0):
#
#         if current_iter % self.update_freq != 0:
#             return torch.tensor(0.0, device=self.device)
#
#         batch_size = enhanced_img.shape[0]
#         assert len(prompts) == 2, "你给我老实点，必须给两个 prompt，一个正一个反 😤"
#
#         # ✨ 获取文本特征
#         with torch.no_grad():
#             text_feats = self._get_text_features(prompts, batch_size)  # (B, 2, D)
#         # 💦 获取图像特征
#         student_feats = self.clip_model.encode_image(enhanced_img)  # (B, D)
#         student_feats = student_feats / student_feats.norm(dim=-1, keepdim=True)
#         # 💥 算余弦相似度
#         sim = torch.cosine_similarity(student_feats.unsqueeze(1), text_feats, dim=-1)  # (B, 2)
#         # 🔥 softmax 后取正向语句的得分
#         sim_softmax = F.softmax(sim, dim=1)  # (B, 2)
#         positive_score = sim_softmax[:, 0]  # 取第一个正向语句得分
#         # 🧨 目标就是它要越靠近 1 越好
#         target = torch.ones_like(positive_score)
#         loss = F.mse_loss(positive_score, target)
#         return loss
# # class PromptContrastiveCLIPLoss(nn.Module):
# #     def __init__(self, clip_model, prompt_path, device="cuda", update_freq=1):
# #         super(PromptContrastiveCLIPLoss, self).__init__()
# #         self.device = device
# #         self.update_freq = update_freq
# #         self.clip_model = clip_model.eval()
# #
# #         # 💥 加载训练好的 prompts
# #         # print(f"💥 Loading prompt from: {prompt_path}")
# #         self.learn_prompt = Prompts([" ".join(["X"] * 16)] * 2).to(device)
# #         self.learn_prompt = torch.nn.DataParallel(self.learn_prompt)
# #
# #         # ⚠️ 尝试加载 state_dict，如果失败就报出问题位置
# #         try:
# #             state_dict = torch.load(prompt_path, map_location=device)
# #             self.learn_prompt.load_state_dict(state_dict)
# #         except RuntimeError as e:
# #             print(f"❌ Failed to load state_dict from {prompt_path}: {e}")
# #             raise e
# #
# #         self.learn_prompt.eval()
# #
# #     def forward(self, student_img, teacher_img, current_iter=0):
# #         if current_iter % self.update_freq != 0:
# #             return torch.tensor(0.0, device=self.device)
# #
# #         batch_size = student_img.shape[0]
# #
# #         with torch.no_grad():
# #             embedding_prompt = self.learn_prompt.module.embedding_prompt  # (2, L, D)
# #             prompt_length = embedding_prompt.shape[1]
# #
# #             # ✅ 限制 token 长度别超 77，CLIP 容量就这么大
# #             token_str = " ".join(["X"] * min(prompt_length, 64))  # 💦 稳妥一点别太长
# #             tokenized_prompts = clip.tokenize([token_str] * 2).to(self.device)
# #
# #             # 💋 文本特征
# #             text_feats = self.clip_model.encode_text(tokenized_prompts)  # (2, D)
# #             text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
# #             text_feats = text_feats[0].unsqueeze(0).expand(batch_size, -1)  # 正向 prompt 拿第一个
# #
# #         # 💦 图像特征
# #         student_feats = self.clip_model.encode_image(student_img)
# #         teacher_feats = self.clip_model.encode_image(teacher_img)
# #
# #         student_feats = student_feats / student_feats.norm(dim=-1, keepdim=True)
# #         teacher_feats = teacher_feats / teacher_feats.norm(dim=-1, keepdim=True)
# #
# #         # ❤️ 相似度来一个
# #         sim_student = torch.cosine_similarity(student_feats, text_feats, dim=-1)  # (B,)
# #         sim_teacher = torch.cosine_similarity(teacher_feats, text_feats, dim=-1)  # (B,)
# #
# #         # 😤 看哪个小坏蛋没学好，干他
# #         mask = sim_student < sim_teacher
# #         if mask.sum() == 0:
# #             return torch.tensor(0.0, device=self.device)
# #
# #         loss = F.mse_loss(sim_student[mask], sim_teacher[mask].detach())
# #
# #         return loss
# class PromptOnlyCLIPRankingLoss(nn.Module):
#     def __init__(self, clip_model, prompt_module, device="cuda", update_freq=1):
#         super().__init__()
#         self.clip_model = clip_model.eval()
#         self.prompt_module = prompt_module.eval()
#         self.device = device
#         self.update_freq = update_freq
#
#     def forward(self, student_img, teacher_img, current_iter=0):
#         if current_iter % self.update_freq != 0:
#             return torch.tensor(0.0, device=self.device)
#
#         batch_size = student_img.shape[0]
#         with torch.no_grad():
#             prompt_embed = self.prompt_module.module.embedding_prompt  # (2, L, D)
#             prompt_length = prompt_embed.shape[1]
#             token_str = " ".join(["X"] * min(prompt_length, 64))
#             tokenized_prompts = clip.tokenize([token_str]).to(self.device)
#             text_feats = self.clip_model.encode_text(tokenized_prompts)
#             text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
#             text_feats = text_feats.expand(batch_size, -1)
#
#         # 提取图像特征
#         student_feats = self.clip_model.encode_image(student_img)
#         teacher_feats = self.clip_model.encode_image(teacher_img)
#         student_feats = student_feats / student_feats.norm(dim=-1, keepdim=True)
#         teacher_feats = teacher_feats / teacher_feats.norm(dim=-1, keepdim=True)
#
#         # 计算余弦相似度
#         sim_student = torch.cosine_similarity(student_feats, text_feats, dim=-1)
#         sim_teacher = torch.cosine_similarity(teacher_feats, text_feats, dim=-1)
#
#         sim = torch.stack([sim_teacher, sim_student], dim=1)  # (B, 2)
#         sim_softmax = F.softmax(sim, dim=1)
#         student_score = sim_softmax[:, 1]
#         target = torch.ones_like(student_score)
#         loss = F.mse_loss(student_score, target)
#
#         return loss
#
# import numpy as np
# import torch.nn as nn
# import math
# from torch.utils import data
# import os
# from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
# import logging
# import yaml
#
# def get_option(opt_path):
#     with open(opt_path, 'r') as f:
#         option = yaml.safe_load(f)
#
#     option.setdefault('seed', 2022)
#
#     # 👉 自动把某些字段的字符串数字列表转成 int
#     def convert_list(val):
#         return [int(x) if isinstance(x, str) and x.isdigit() else x for x in val]
#
#     model = option.get('model', {})
#     if 'embed_dims' in model:
#         model['embed_dims'] = convert_list(model['embed_dims'])
#     if 'mlp_ratios' in model:
#         model['mlp_ratios'] = convert_list(model['mlp_ratios'])
#     if 'serial_depths' in model:
#         model['serial_depths'] = convert_list(model['serial_depths'])
#     if 'in_chans' in model:
#         model['in_chans'] = int(model['in_chans']) if isinstance(model['in_chans'], str) else model['in_chans']
#     if 'patch_size' in model:
#         model['patch_size'] = int(model['patch_size']) if isinstance(model['patch_size'], str) else model['patch_size']
#
#     return option
#
# def build_optimizer(opt, model):
#     optimizer_name = opt['optimizer']
#     try:
#         optimizer_class = getattr(torch.optim, optimizer_name)
#         optimizer = optimizer_class(model.parameters(), lr=opt['lr'])
#     except:
#         raise NotImplementedError('Unable to load optimizer: \'%s\' ' % optimizer_name)
#
#     return optimizer
#
# def build_lr_scheduler(opt, optimizer):
#     lr_scheduler_name = opt['lr_scheduler'] if 'lr_scheduler' in opt.keys() else None
#     if lr_scheduler_name:
#         try:
#             lr_scheduler_class = getattr(getattr(torch.optim, 'lr_scheduler'), lr_scheduler_name)
#         except:
#             raise NotImplementedError(
#                 'Unable to load lr_scheduler: \'%s\', please check if there are any spelling errors ' % lr_scheduler_name)
#         try:
#             lr_scheduler = lr_scheduler_class(optimizer, **opt['lr_scheduler_arg'])
#         except:
#             raise NotImplementedError('Failed to load optimizer')
#         return lr_scheduler
#     else:
#         return None
#
# def build_dataloader(opt, type='train'):
#     dataset_name = opt['dataset_name']
#     module = __import__('dataset.dataset')
#     dataset_class = getattr(module, dataset_name)
#     dataset = dataset_class(opt, type)
#     dataloader = data.DataLoader(dataset,
#                                  batch_size=opt['bs'] if type == 'train' else 1,
#                                  num_workers=opt['num_workers'],
#                                  shuffle=True if type == 'train' else False)
#     return dataloader
#
# def build_model(opt):
#     model_name = opt['model_name']
#     module = __import__('all_model.' + model_name + '.model')
#     model_class = getattr(module, model_name)
#
#     # load model args
#     all_args = list(opt.keys())
#     model_args = {}
#     for i in range(len(all_args) - 4):
#         model_args[all_args[i + 4]] = opt.get(all_args[i + 4])
#     model = model_class(**model_args)
#
#     if opt['cuda']:
#         model = model.cuda()
#     if opt['parallel']:
#         model = torch.nn.DataParallel(model)
#
#     # load pretrained dict
#     if opt['resume_ckpt_path']:
#         ckpt_dict = torch.load(opt['resume_ckpt_path'])['net']
#         model.load_state_dict(ckpt_dict)
#
#     return model
#
# def build_logger(opt):
#     make_dir(os.path.join(opt['save_root'], opt['log']))
#     log_path = os.path.join(opt['save_root'], opt['log'], 'logs.log')
#     log_format = "%(asctime)s - %(message)s"
#     logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#
#     return logger
#
# def make_dir(path):
#     if os.path.exists(path):
#         pass
#     else:
#         paths = path.split('/')
#         now_path = ''
#         for temp_path in paths:
#             now_path = os.path.join(now_path, temp_path)
#             if not os.path.exists(now_path):
#                 os.mkdir(now_path)
#         return
#
# def calc_psnr(pred, gt, is_for_torch=True):
#     if is_for_torch:
#         pred = pred[0].permute(1, 2, 0).detach().numpy()
#         gt = gt[0].premute(1, 2, 0).detach().numpy()
#
#         psnr = peak_signal_noise_ratio(gt, pred)
#     else:
#         psnr = peak_signal_noise_ratio(gt, pred)
#
#     return psnr
#
# def calc_ssim(pred, gt, is_for_torch=True):
#     if is_for_torch:
#         pred = pred[0].permute(1, 2, 0).detach().numpy()
#         gt = gt[0].premute(1, 2, 0).detach().numpy()
#
#         ssim = structural_similarity(gt, pred, multichannel=True)
#     else:
#         ssim = structural_similarity(gt, pred, multichannel=True)
#
#     return ssim
#
# def normalize_img(img):
#     if torch.max(img) > 1 or torch.min(img) < 0:
#         im_max = torch.max(img)
#         im_min = torch.min(img)
#
#         img = (img - im_min) / (im_max - im_min + 1e-7)
#
#     return img
#
# def preprocessing(d_img_org):
#     d_img_org = padding_img(d_img_org)
#     x_his = build_historgram(d_img_org)
#     return {
#         'x': d_img_org,
#         'x_his': x_his
#     }
#
# def padding_img(img):
#     b, c, h, w = img.shape
#     h_out = math.ceil(h / 32) * 32
#     w_out = math.ceil(w / 32) * 32
#
#     left_pad = (w_out - w) // 2
#     right_pad = w_out - w - left_pad
#     top_pad = (h_out - h) // 2
#     bottom_pad = h_out - h - top_pad
#
#     img = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)
#
#     return img
#
# def build_historgram(img):
#     with torch.no_grad():
#         b, _, _, _ = img.shape
#
#         r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
#         g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
#         b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)
#
#         historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
#
#         for i in range(1, b):
#             r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
#             g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
#             b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)
#
#             historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
#             historgram = torch.cat((historgram, historgram_temp), dim=0)
#
#     return historgram
#
# class RankerConditionalLoss(nn.Module):
#     def __init__(self, opt_path, checkpoint_path):
#         super().__init__()
#         self.device = torch.device('cuda')
#         options = get_option(opt_path)
#         options['model']['model_name'] = 'URanker'
#         options['model']['resume_ckpt_path'] = checkpoint_path
#         self.model = build_model(options['model']).to(self.device).eval()
#
#  # 🐾 分数差值：学生必须甩老师 margin 分数才行！
#
#     def forward(self, enhanced_img, teacher_img):
#         # 只取前三通道并resize
#         enhanced_img = F.interpolate(enhanced_img[:, :3], size=(256, 256), mode='bilinear', align_corners=False)
#         teacher_img = F.interpolate(teacher_img[:, :3], size=(256, 256), mode='bilinear', align_corners=False)
#
#         # 预处理
#         enhanced_inputs = preprocessing(enhanced_img)
#         teacher_inputs = preprocessing(teacher_img)
#
#         # 老师分数：不带梯度
#         with torch.no_grad():
#             teacher_score = self.model(**teacher_inputs)['final_result'].view(-1)
#
#         # 学生分数：带梯度
#         enhanced_score = self.model(**enhanced_inputs)['final_result'].view(-1)
#
#         # 🚨 Loss：只要学生比老师低 margin 分就狠狠罚它！
#         loss = F.relu(teacher_score - enhanced_score + 0.3)
#
#
#         return loss.mean()
#
# from turtle import forward
# import torchvision.transforms as transforms
# import torch
# import clip
# import torch.nn as nn
# from torch.nn import functional as F
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
# model.to(device)
# img_resize = transforms.Resize((224,224))
# for para in model.parameters():
# 	para.requires_grad = False
# # def get_clip_score(tensor,words):
# # 	score=0
# # 	for i in range(tensor.shape[0]):
# # 		#image preprocess
# # 		clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# # 		img_resize = transforms.Resize((224,224))
# # 		image2=img_resize(tensor[i])
# # 		image=clip_normalizer(image2).unsqueeze(0)
# # 		#get probabilitis
# # 		text = clip.tokenize(words).to(device)
# # 		logits_per_image, logits_per_text = model(image, text)
# # 		probs = logits_per_image.softmax(dim=-1)
# # 		#2-word-compared probability
# # 		# prob = probs[0][0]/probs[0][1]#you may need to change this line for more words comparison
# # 		prob = probs[0][0]
# # 		score =score + prob
# #
# # 	return score
#
# class Prompts(nn.Module):
#     def __init__(self, initials=None):
#         super(Prompts, self).__init__()
#         self.text_encoder = TextEncoder(model)
#
#         if isinstance(initials, list):
#             # 使用自定义文本列表初始化
#             tokenized = clip.tokenize(initials).cuda()
#             with torch.no_grad():
#                 embed = model.token_embedding(tokenized)
#             self.embedding_prompt = nn.Parameter(embed.clone().detach()).cuda()
#
#         elif isinstance(initials, str):
#             # 加载已有的预训练 embedding
#             state_dict = torch.load(initials)
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 new_state_dict[k.replace('module.', '')] = v
#             self.embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt'].clone().detach()).cuda()
#
#         else:
#             # 默认初始化为 prompt_len 个 "X"
#             prompt_text = " ".join(["X"] * config.length_prompt)
#             tokenized = clip.tokenize([prompt_text]).cuda()
#             with torch.no_grad():
#                 embed = model.token_embedding(tokenized)
#             self.embedding_prompt = nn.Parameter(embed.clone().detach()).cuda()
#
#         self.embedding_prompt.requires_grad = True
#
#     def forward(self, tensor, flag=1):
#         prompt_text = " ".join(["X"] * config.length_prompt)
#         tokenized_prompts = clip.tokenize([prompt_text]).cuda()
#
#         text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts)
#
#         probs = []
#         for i in range(tensor.shape[0]):
#             image_features = tensor[i]
#             nor = torch.norm(text_features, dim=-1, keepdim=True)
#             sim = (100.0 * image_features @ (text_features / nor).T)
#             if flag:
#                 sim = sim.softmax(dim=-1)
#                 probs.append(sim[:, 0])
#             else:
#                 probs.append(sim)
#
#         return torch.stack(probs, dim=0)
#
# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype
#
#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)
#         x = self.ln_final(x).type(self.dtype)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#         return x
#
# class Config:
#     def __init__(self):
#         self.length_prompt = 16
#         self.prompt_pretrain_dir = "checkpoints/pretrained_prompt.pth"
#         self.load_pretrain_prompt = True
#         self.prompt_snapshots_folder = "snapshots/"
#         self.num_clip_pretrained_iters = 8000
#
# config = Config()
#
# clip_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
#                          (0.26862954, 0.26130258, 0.27577711))
# ])
#
# def get_clip_score_from_feature(tensor, text_features):
#     score = 0
#     for i in range(tensor.shape[0]):
#         image = clip_transform(tensor[i]).reshape(1, 3, 224, 224)
#         image_features = model.encode_image(image)
#         image_nor = image_features.norm(dim=-1, keepdim=True)
#         text_nor = text_features.norm(dim=-1, keepdim=True)
#         similarity = (100.0 * (image_features / image_nor) @ (text_features / text_nor).T).softmax(dim=-1)
#         score += similarity[0][0]
#     return score / tensor.shape[0]
#
# class L_clip_from_feature(nn.Module):
#     def __init__(self):
#         super(L_clip_from_feature, self).__init__()
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def forward(self, x, text_features):
#         return get_clip_score_from_feature(x, text_features)
#
# # 初始化你的 prompt 模块
# learn_prompt = Prompts().cuda()
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import resize as img_resize
#
# # 归一化器（你要保证跟你用的 clip 模型匹配）
#
# class ImageCleanModel(BaseModel):
#     """Base Deblur model for single image deblur."""
#
#     def __init__(self, opt):
#         super(ImageCleanModel, self).__init__(opt)
#
#         # define mixed precision
#         self.use_amp = opt.get('use_amp', False) and load_amp
#         self.amp_scaler = GradScaler(enabled=self.use_amp)
#         if self.use_amp:
#             print('Using Automatic Mixed Precision')
#         else:
#             print('Not using Automatic Mixed Precision')
#
#         # define network
#         self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
#         if self.mixing_flag:
#             mixup_beta = self.opt['train']['mixing_augs'].get(
#                 'mixup_beta', 1.2)
#             use_identity = self.opt['train']['mixing_augs'].get(
#                 'use_identity', False)
#             self.mixing_augmentation = Mixing_Augment(
#                 mixup_beta, use_identity, self.device)
#
#         self.net_g = define_network(deepcopy(opt['network_g']))
#         self.net_g = self.model_to_device(self.net_g)
#         # self.print_network(self.net_g)
#
#         # load pretrained models
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         if load_path is not None:
#             self.load_network(self.net_g, load_path,
#                               self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
#
#         if self.is_train:
#             self.init_training_settings()
#
#     # 在模型类 (ImageCleanModel) 内部定义
#     def get_enhanced_result(self):
#         """获取当前模型增强结果"""
#         if hasattr(self, 'output'):
#             return self.output.detach()  # 返回增强后的图像
#         else:
#             raise AttributeError("模型尚未生成增强结果，请检查前向传播过程。")
#
#     def init_training_settings(self):
#         self.net_g.train()
#         train_opt = self.opt['train']
#
#         self.ema_decay = train_opt.get('ema_decay', 0)
#         if self.ema_decay > 0:
#             logger = get_root_logger()
#             logger.info(
#                 f'Use Exponential Moving Average with decay: {self.ema_decay}')
#             # define network net_g with Exponential Moving Average (EMA)
#             # net_g_ema is used only for testing on one GPU and saving
#             # There is no need to wrap with DistributedDataParallel
#             self.net_g_ema = define_network(self.opt['network_g']).to(
#                 self.device)
#             # load pretrained model
#             load_path = self.opt['path'].get('pretrain_network_g', None)
#             if load_path is not None:
#                 self.load_network(self.net_g_ema, load_path,
#                                   self.opt['path'].get('strict_load_g',
#                                                        True), 'params_ema')
#             else:
#                 self.model_ema(0)  # copy net_g weight
#             self.net_g_ema.eval()
#
#         # define losses
#         # if train_opt.get('pixel_opt'):
#         #     pixel_type = train_opt['pixel_opt'].pop('type')
#         #     cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
#         #     self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
#         #         self.device)      #如何写 weighted loss 呢？传参构造Loss函数
#         # else:
#         #     raise ValueError('pixel loss are None.')
#
#         # set up optimizers and schedulers
#         self.setup_optimizers()
#         self.setup_schedulers()
#     #共享权重
# #     def setup_optimizers(self):
# #         train_opt = self.opt['train']
# #         optim_params = []
# #
# #         for k, v in self.net_g.named_parameters():
# #             if v.requires_grad:
# #                 optim_params.append(v)
# #             else:
# #                 logger = get_root_logger()
# #                 logger.warning(f'Params {k} will not be optimized.')
# #
# #         optim_type = train_opt['optim_g'].pop('type')
# #         if optim_type == 'Adam':
# #             self.optimizer_g = torch.optim.Adam(
# #                 optim_params, **train_opt['optim_g'])
# #         elif optim_type == 'AdamW':
# #             self.optimizer_g = torch.optim.AdamW(
# #                 optim_params, **train_opt['optim_g'])
# #         else:
# #             raise NotImplementedError(
# #                 f'optimizer {optim_type} is not supperted yet.')
# #         self.optimizers.append(self.optimizer_g)
# # #学生
#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []
#
#         for k, v in self.net_g.named_parameters():
#             if v.requires_grad:
#                 optim_params.append(v)
#             else:
#                 logger = get_root_logger()
#                 logger.warning(f'⚠️ Params {k} will not be optimized.')
#
#         if not optim_params:
#             raise RuntimeError("❌ 没有可训练的参数！检查 `self.net_g` 是否加载正确！")
#
#         optim_type = train_opt['optim_g'].get('type', 'Adam')  # **❌ 不能用 pop，要用 get**
#
#         optimizer_opts = {k: v for k, v in train_opt['optim_g'].items() if
#                           k != 'type'}  # **❌ 不能直接传 train_opt['optim_g']，因为有 'type' 键**
#
#         if optim_type == 'Adam':
#             self.optimizer_g = torch.optim.Adam(optim_params, **optimizer_opts)
#         elif optim_type == 'AdamW':
#             self.optimizer_g = torch.optim.AdamW(optim_params, **optimizer_opts)
#         else:
#             raise NotImplementedError(f'❌ optimizer {optim_type} 还不支持！')
#
#         self.optimizers.append(self.optimizer_g)
#
#     def feed_train_data(self, lq1,lq2,gt):
#        self.lq1= lq1.to(self.device)
#        self.lq2 = lq2.to(self.device)
#        self.gt = gt.to(self.device)
#        if self.mixing_flag:
#         self.gt, self.lq1 = self.mixing_augmentation(self.gt, self.lq1)
#         # if self.mixing_flag:
#         #     self.gt, self.lq2 = self.mixing_augmentation(self.gt, self.lq2)
#     # def feed_data(self, data):
#     #     self.lq1 = data['lq'].to(self.device)
#     #     self.lq2 = data['lq2'].to(self.device)
#     #     if 'gt' in data:
#     #         self.gt = data['gt'].to(self.device)
#     #     # if 'gt' in data:
#     #     #     self.gt =  self.gt
#     def feed_data(self, data):
#         self.lq1 = data['lq1'].to(self.device)
#         self.lq2 = data['lq2'].to(self.device)
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)
#         if 'gt' in data:
#             self.gt =  self.gt
#     def optimize_parameters(self, current_iter):
#         self.optimizer_g.zero_grad()
#         enhanced_input = self.lq1.clone()
#         device = self.lq1.device
# #共享权重
#         with autocast(enabled=self.use_amp):
#             preds1, preds2, illu1, illu2, input1, input2, fire1, fire2, img1, img2, illu_fea1, illu_fea2 = self.net_g( self.lq1, self.lq2)
# #学生
#         # with autocast(enabled=self.use_amp):
#         #     preds, illu,input,fire,img,mean= self.net_g(self.lq1)
#
#         # **🔥 确保所有输出在同一个 device**
#         device = self.lq1.device
#         self.output1 = preds1.to(device)
#         self.output2 = preds2.to(device)
#         self.illu = illu1.to(device)
#         self.illu = illu2.to(device)
#         # **🔥 确保 `self.gt` 也在同一个 device 和 dtype**
#         dtype = self.output1.dtype
#         self.gt = self.gt.to(device, dtype)
#         loss_dict = OrderedDict()
#         l_pix1 = F.l1_loss(self.output1.to(device), self.gt)
#         # l_pix2 = F.l1_loss(self.output2.to(device), self.gt)
#
#         grad_loss = Gradient_Difference_Loss(alpha=1, chans=3, cuda=True)
#         edgeloss1 = grad_loss(preds1, self.gt)
#         # edgeloss2 = grad_loss(preds2, self.gt)
#
#         spectrum_loss = SpectrumLoss(lambda_high=1.0, lambda_low=1.0, lambda_smooth=0.1)
#
#         smooth_loss_fn = SmoothLoss(loss_weight=0.3, reduction='mean', eps=1e-2)
#         # `preds1` 和 `preds2` 之间的 smooth loss
#         multual_loss = MultualLoss(loss_weight=0.20, reduction='mean')
#         lossml1 = multual_loss(preds)
#         # illum_loss_fn = IlluminationLoss(lambda_bright=0.5, lambda_uniform=1, lambda_smooth=0.1, lambda_reg=0.1)
#         # # 经过增强的图像
#         # illum_loss1 = illum_loss_fn(input, illu)
#         Lloss1 = L_spa()
#         # ✅ 创建 L_exp 的时候就传好参数：
#         Lloss2 = L_exp(patch_size=16, mean_val=0.6).to(device)
#         Lloss3 = L_TV()
#         Lloss4 = L_color()
#         # L1 = Lloss1(img,input).mean()
#         L21 = Lloss2(input,mean).mean()
#         L22 = Lloss2(preds, mean).mean()
#         L42 = Lloss2(preds, mean).mean()
#         # L3 = Lloss3(input).mean()
#         L4 = Lloss4(preds).mean()
#         # lossml2 = multual_loss(preds2)
#         wavelet_loss = FourierHighBoostLoss()
#         illu_pectrumloss1 = wavelet_loss(enhanced=preds, reference=self.gt)
#             # 加载 TextEncoder 和 Prompt
#         # switch_iter = 20*300  # 前 10000 次用 loss1，之后用 loss2
#         # if current_iter < switch_iter:
#         #     total_loss = l_pix1 + 0.1 * illu_pectrumloss1 + edgeloss1 + 0.1 * L21 +  L22 + lossml1  # loss1
#         #     loss_dict['l_pix1'] = l_pix1.detach()
#         #     loss_dict['edgeloss1'] = edgeloss1.detach()
#         #     loss_dict['illu_pectrumloss1'] = illu_pectrumloss1.detach()
#         #     loss_dict['L21'] = L21.detach()
#         #     loss_dict['L22'] = L21.detach()
#         #     loss_dict['total_loss'] = total_loss.detach()
#         # else:
#         #     text_encoder = TextEncoder(model)
#         #     learn_prompt = Prompts([" ".join(["X"] * config.length_prompt)] * 2).cuda()
#         #     learn_prompt = torch.nn.DataParallel(learn_prompt)
#         #     learn_prompt.load_state_dict(
#         #         torch.load("/data/gez/Project/CLIP-LIT-main/train0/snapshots_prompt_train0/best_prompt_round0.pth",
#         #                    map_location=device))
#         #     learn_prompt.eval()
#         #
#         #     embedding_prompt = learn_prompt.module.embedding_prompt
#         #     tokenized_prompts = torch.cat([clip.tokenize(" ".join(["X"] * config.length_prompt))])
#         #     tokenized_prompts = tokenized_prompts.cuda()
#         #
#         #     text_features = text_encoder(embedding_prompt, tokenized_prompts)
#         #     loss_fn = CLIPSoftmaxContrastiveLoss(model)
#         #     clip_loss = loss_fn(preds, self.gt, text_features)
#         #
#         #     ranker_loss_fn = RankerConditionalLoss(
#         #         opt_path="/data/gez/Project/UnderwaterRanker-master/options/URanker.yaml",
#         #         checkpoint_path="/data/gez/Project/UnderwaterRanker-master/checkpoints/URanker_ckpt.pth",
#         #     )
#         #     rank_loss = ranker_loss_fn(preds, self.gt)
#         #     total_loss = l_pix1 + 0.1 * illu_pectrumloss1 + edgeloss1 + 0.1 * L21 +0.05 * rank_loss +  L22 + 0.005 * clip_loss + 0.1 * lossml1 # loss2，可以改为你想要的loss2形式\
#         #     loss_dict['l_pix1'] = l_pix1.detach()
#         #     loss_dict['edgeloss1'] = edgeloss1.detach()
#         #     loss_dict['illu_pectrumloss1'] = illu_pectrumloss1.detach()
#         #     loss_dict['L21'] = L21.detach()
#         #     loss_dict['clip_loss'] = clip_loss.detach()
#         #     loss_dict['rank_loss'] = rank_loss.detach()
#         #     loss_dict['L22'] = L22.detach()
#         #     loss_dict['lossml1'] = lossml1.detach()
#         #     loss_dict['total_loss'] = total_loss.detach()
# #逐渐增大权重
#         max_iter = 2429*50  # 总迭代次数，自己定义一下，比如 max_iter = 20000
#         # rank_weight = 0.001 * (current_iter / max_iter)
#         # clip_weight = 0.00001 * (current_iter / max_iter)
#
#         # 限制最大值，防止超过
#         # text_encoder = TextEncoder(model)
#         # learn_prompt = Prompts([" ".join(["X"] * config.length_prompt)] * 2).cuda()
#         # learn_prompt = torch.nn.DataParallel(learn_prompt)
#         # learn_prompt.load_state_dict(
#         #     torch.load("/data/gez/Project/CLIP-LIT-main/train0/snapshots_prompt_train0/best_prompt_round0.pth",
#         #                map_location=device))
#         # learn_prompt.eval()
#         #
#         # embedding_prompt = learn_prompt.module.embedding_prompt
#         # tokenized_prompts = torch.cat([clip.tokenize(" ".join(["X"] * config.length_prompt))])
#         # tokenized_prompts = tokenized_prompts.cuda()
#         #
#         # text_features = text_encoder(embedding_prompt, tokenized_prompts)
#         # L_clip = L_clip_from_feature()
#         # clip_loss =L_clip(preds, text_features)
#
#         ranker_loss_fn = RankerConditionalLoss(
#             opt_path="/data/gez/Project/UnderwaterRanker-master/options/URanker.yaml",
#             checkpoint_path="/data/gez/Project/UnderwaterRanker-master/checkpoints/URanker_ckpt.pth",
#         )
#         rank_loss = ranker_loss_fn(preds, self.gt)
#
#         total_loss = l_pix1 +edgeloss1+ 0.1*illu_pectrumloss1+0.1*L21 +0.001*rank_loss
#         loss_dict['l_pix1'] = l_pix1.detach()
#         loss_dict['edgeloss1'] = edgeloss1.detach()
#         loss_dict['illu_pectrumloss1'] = illu_pectrumloss1.detach()
#         # loss_dict['L21'] = L21.detach()
#         # loss_dict['clip_loss'] = clip_loss.detach()
#         loss_dict['rank_loss'] = rank_loss.detach()
#         loss_dict['L22'] = L22.detach()
#         loss_dict['lossml1'] = lossml1.detach()
#         loss_dict['total_loss'] = total_loss.detach()
#         self.optimizer_g.zero_grad()
#         total_loss.backward()
#         if self.opt['train']['use_grad_clip']:
#             torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
#         self.optimizer_g.step()
#
#         if self.ema_decay > 0:
#             self.model_ema(decay=self.ema_decay)
#
#         self.log_dict = self.reduce_loss_dict(loss_dict)
#
# #共享权重训练
#     def pad_test(self, window_size):
#         scale = self.opt.get('scale', 1)
#         mod_pad_h, mod_pad_w = 0, 0
#         _, _, h, w = self.lq1.size()
#
#         # 计算填充的高度和宽度
#         if h % window_size != 0:
#             mod_pad_h = window_size - h % window_size
#         if w % window_size != 0:
#             mod_pad_w = window_size - w % window_size
#
#         # 反射填充
#         img = F.pad(self.lq1, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#
#         # 进行无填充测试
#         self.nonpad_test(img)
#
#         # 确保 `self.output` 只取第一张结果
#         if isinstance(self.output, (tuple, list)):
#             self.output = self.output[0]
#
#         # 进行裁剪，去掉填充区域
#         _, _, h_out, w_out = self.output.size()
#         self.output = self.output[:, :, 0:h_out - mod_pad_h * scale, 0:w_out - mod_pad_w * scale]
#
#     def nonpad_test(self, img=None):
#         if img is None:
#             img = self.lq1
#
#         img = img.to(torch.float32)
#
#         net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
#         net.eval()
#
#         with torch.no_grad():
#             # 🔥 直接传 `img`，让 `forward()` 自己决定跑单分支
#             pred = net(img)
#
#         # 🔥 适配返回结果，不管是 tuple 还是 list，都取第一个
#         if isinstance(pred, (tuple, list)):
#             self.output = pred[0]
#         else:
#             self.output = pred
#
#         net.train()  # 恢复训练模式
# #单分支/学生
# #     def pad_test(self, window_size):
# #         scale = self.opt.get('scale', 1)
# #         mod_pad_h, mod_pad_w = 0, 0
# #         _, _, h, w = self.lq1.size()
# #
# #         # 计算需要填充的高度和宽度
# #         if h % window_size != 0:
# #             mod_pad_h = window_size - h % window_size
# #         if w % window_size != 0:
# #             mod_pad_w = window_size - w % window_size
# #
# #         # 使用反射填充图像
# #         img = F.pad(self.lq1, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
# #         self.nonpad_test(img)
# #
# #         # 解包 self.output 如果它是元组
# #         if isinstance(self.output, tuple):
# #             self.output = self.output[0]  # 获取第一个张量部分
# #
# #         # 对 output 做裁剪，去掉填充部分
# #         _, _, h, w = self.output.size()
# #         self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
# # #学生
# #     def nonpad_test(self, img=None):
# #         if img is None:
# #             img = self.lq1  # 默认使用 `self.lq` 作为输入图像
# #
# #         # 将输入图像转换为 float32 类型
# #         img = img.to(torch.float32)
# #
# #         # 使用 ema 网络进行推理
# #         if hasattr(self, 'net_g_ema'):
# #             self.net_g_ema.eval()
# #             with torch.no_grad():
# #                 # 仅传递图像给网络
# #                 pred = self.net_g_ema(img)
# #             if isinstance(pred, list):
# #                 pred = pred[-1]
# #             # 只保留增强图的输出
# #             if isinstance(pred, tuple):
# #                 self.output = pred[0]  # 增强图像
# #             else:
# #                 self.output = pred  # 增强图像
# #         else:
# #             self.net_g.eval()
# #             with torch.no_grad():
# #                 # 仅传递图像给网络
# #                 pred = self.net_g(img)
# #             if isinstance(pred, list):
# #                 pred = pred[-1]
# #             # 只保留增强图的输出
# #             if isinstance(pred, tuple):
# #                 self.output = pred[0]  # 增强图像
# #             else:
# #                 self.output = pred  # 增强图像
# #
# #         self.net_g.train()  # 恢复训练模式
#     #
#     def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
#         if os.environ['LOCAL_RANK'] == '0':
#             return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
#         else:
#             return 0.
# #学生
#     # def nondist_validation(self, dataloader, current_iter, tb_logger,
#     #                        save_img, rgb2bgr, use_image):
#     #     dataset_name = dataloader.dataset.opt['name']
#     #     with_metrics = self.opt['val'].get('metrics') is not None
#     #     if with_metrics:
#     #         self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
#     #
#     #     window_size = self.opt['val'].get('window_size', 0)
#     #     test = partial(self.pad_test, window_size) if window_size else self.nonpad_test
#     #     cnt = 0
#     #
#     #     for idx, val_data in enumerate(dataloader):
#     #         img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
#     #         self.feed_data(val_data)
#     #         test()
#     #
#     #         visuals = self.get_current_visuals()
#     #         sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
#     #
#     #         # 💥 解决 gt 丢失问题：确保 `gt` 一直存在
#     #         if 'gt' in visuals and isinstance(visuals['gt'], torch.Tensor):
#     #             gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
#     #         else:
#     #             visuals['gt'] = self.gt  # 用教师模型伪标签填充
#     #             gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
#     #
#     #         # 🚀 释放不必要的变量，防止显存溢出
#     #         del self.lq1, self.output
#     #         torch.cuda.empty_cache()
#     #
#     #         # 📸 保存增强图和 gt
#     #         if save_img:
#     #             save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
#     #             save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')
#     #             imwrite(sr_img, save_img_path)
#     #             imwrite(gt_img, save_gt_img_path)
#     #
#     #         # 🎯 计算度量指标
#     #         if with_metrics:
#     #             opt_metric = deepcopy(self.opt['val']['metrics'])
#     #             for name, opt_ in opt_metric.items():
#     #                 metric_type = opt_.pop('type')
#     #
#     #                 # 💡 保证 `gt` 和 `sr` 形状一致
#     #                 if visuals['result'].shape != visuals['gt'].shape:
#     #                     visuals['gt'] = F.interpolate(visuals['gt'], size=visuals['result'].shape[2:], mode='bilinear',
#     #                                                   align_corners=False)
#     #
#     #                 self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'],
#     #                                                                                  **opt_)
#     #         cnt += 1
#     #
#     #     # 📊 计算最终度量分数
#     #     current_metric = 0.
#     #     if with_metrics:
#     #         for metric in self.metric_results.keys():
#     #             self.metric_results[metric] /= cnt
#     #             current_metric = self.metric_results[metric]
#     #         self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
#     #
#     #     return current_metric
# #共享教师
#     def nondist_validation(self, dataloader, current_iter, tb_logger,
#                            save_img, rgb2bgr, use_image):
#         dataset_name = dataloader.dataset.opt['name']
#         with_metrics = self.opt['val'].get('metrics') is not None
#         if with_metrics:
#             self.metric_results = {
#                 metric: 0
#                 for metric in self.opt['val']['metrics'].keys()
#             }
#         # pbar = tqdm(total=len(dataloader), unit='image')
#
#         window_size = self.opt['val'].get('window_size', 0)
#
#         if window_size:
#             test = partial(self.pad_test, window_size)
#         else:
#             test = self.nonpad_test
#
#         cnt = 0
#
#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data['lq1_path'][0]))[0]
#             self.feed_data(val_data)
#             test()
#
#             visuals = self.get_current_visuals()
#             sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
#             if 'gt' in visuals:
#                 gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
#                 del self.gt
#             # if 'gt' in visuals:
#             #     # 🚀 先获取 `sr_img` 目标尺寸
#             #     target_size = visuals['result'].shape[2:]
#             #
#             #     # 🛠 确保 `self.gt` 是 4D (B, C, H, W)，然后调整尺寸
#             #     if self.gt.dim() == 4 and self.gt.shape[2:] != target_size:
#             #         self.gt = F.interpolate(self.gt, size=target_size, mode='bilinear', align_corners=False)
#             #
#             #     # 🔥 转换成 `gt_img`
#             #     gt_img = tensor2img([self.gt], rgb2bgr=rgb2bgr)
#
#                 # 💣 释放显存，防止 OOM
#
#                 torch.cuda.empty_cache()
#
#             if save_img:
#
#                 if self.opt['is_train']:
#
#                     save_img_path = osp.join(self.opt['path']['visualization'],
#                                              img_name,
#                                              f'{img_name}_{current_iter}.png')
#
#                     save_gt_img_path = osp.join(self.opt['path']['visualization'],
#                                                 img_name,
#                                                 f'{img_name}_{current_iter}_gt.png')
#                 else:
#
#                     save_img_path = osp.join(
#                         self.opt['path']['visualization'], dataset_name,
#                         f'{img_name}.png')
#                     save_gt_img_path = osp.join(
#                         self.opt['path']['visualization'], dataset_name,
#                         f'{img_name}_gt.png')
#
#                 imwrite(sr_img, save_img_path)
#                 imwrite(gt_img , save_gt_img_path)
#
#             if with_metrics:
#                 # 深拷贝 metrics 配置
#                 opt_metric = deepcopy(self.opt['val']['metrics'])
#
#                 for name, opt_ in opt_metric.items():
#                     metric_type = opt_.get('type', 'calculate_psnr')  # 确保默认值
#
#                     # 过滤掉 `type` 参数，避免 `TypeError`
#                     valid_opt = {k: v for k, v in opt_.items() if k != 'type'}
#
#                     if use_image:
#                         self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **valid_opt)
#                     else:
#                         self.metric_results[name] += getattr(metric_module, metric_type)(
#                             visuals['result'], visuals['gt'], **valid_opt
#                         )
#
#             cnt += 1
#
#         current_metric = 0.
#         if with_metrics:
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= cnt
#                 current_metric = self.metric_results[metric]
#
#             self._log_validation_metric_values(current_iter, dataset_name,
#                                                tb_logger)
#         return current_metric
#
#     def _log_validation_metric_values(self, current_iter, dataset_name,
#                                       tb_logger):
#         log_str = f'Validation {dataset_name},\t'
#         for metric, value in self.metric_results.items():
#             log_str += f'\t # {metric}: {value:.4f}'
#         logger = get_root_logger()
#         logger.info(log_str)
#         if tb_logger:
#             for metric, value in self.metric_results.items():
#                 tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
#
#     # 教师/学生
#     def get_current_visuals(self):
#         out_dict = OrderedDict()
#         out_dict['lq'] = self.lq1.detach().cpu()
#         # out_dict['lq2'] = self.lq2.detach().cpu()
#         out_dict['result'] = self.output.detach().cpu()
#         if hasattr(self, 'gt'):
#             out_dict['gt'] = self.gt.detach().cpu()
#         return out_dict
#
#     def save(self, epoch, current_iter, **kwargs):
#         if self.ema_decay > 0:
#             self.save_network([self.net_g, self.net_g_ema],
#                               'net_g',
#                               current_iter,
#                               param_key=['params', 'params_ema'])
#         else:
#             self.save_network(self.net_g, 'net_g', current_iter)
#         self.save_training_state(epoch, current_iter, **kwargs)
#
#     def save_best(self, best_metric, param_key='params'):
#         psnr = best_metric['psnr']
#         cur_iter = best_metric['iter']
#         save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
#         exp_root = self.opt['path']['experiments_root']
#         save_path = os.path.join(
#             self.opt['path']['experiments_root'], save_filename)
#
#         if not os.path.exists(save_path):
#             for r_file in glob.glob(f'{exp_root}/best_*'):
#                 os.remove(r_file)
#             net = self.net_g
#
#             net = net if isinstance(net, list) else [net]
#             param_key = param_key if isinstance(
#                 param_key, list) else [param_key]
#             assert len(net) == len(
#                 param_key), 'The lengths of net and param_key should be the same.'
#
#             save_dict = {}
#             for net_, param_key_ in zip(net, param_key):
#                 net_ = self.get_bare_model(net_)
#                 state_dict = net_.state_dict()
#                 for key, param in state_dict.items():
#                     if key.startswith('module.'):  # remove unnecessary 'module.'
#                         key = key[7:]
#                     state_dict[key] = param.cpu()
#                 save_dict[param_key_] = state_dict
#
#             torch.save(save_dict, save_path)


# # import importlib
# # import torch
# # from collections import OrderedDict
# # from copy import deepcopy
# # from os import path as osp
# # from tqdm import tqdm
# # import glob
# # import torch.nn as nn
# # from basicsr.models.archs import define_network
# # from basicsr.models.base_model import BaseModel
# # from basicsr.utils import get_root_logger, imwrite, tensor2img

# # loss_module = importlib.import_module('basicsr.models.losses')
# # metric_module = importlib.import_module('basicsr.metrics')

# # import os
# # import random
# # import numpy as np
# # import cv2
# # import torch.nn.functional as F
# # from functools import partial

# # try :
# #     from torch.cuda.amp import autocast, GradScaler
# #     load_amp = True
# # except:
# #     load_amp = False


# # class Mixing_Augment:
# #     def __init__(self, mixup_beta, use_identity, device):
# #         self.dist = torch.distributions.beta.Beta(
# #             torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
# #         self.device = device

# #         self.use_identity = use_identity

# #         self.augments = [self.mixup]

# #     def mixup(self, target, input_):
# #         lam = self.dist.rsample((1, 1)).item()

# #         r_index = torch.randperm(target.size(0)).to(self.device)

# #         target = lam * target + (1 - lam) * target[r_index, :]
# #         input_ = lam * input_ + (1 - lam) * input_[r_index, :]

# #         return target, input_

# #     def __call__(self, target, input_):
# #         if self.use_identity:
# #             augment = random.randint(0, len(self.augments))
# #             if augment < len(self.augments):
# #                 target, input_ = self.augments[augment](target, input_)
# #         else:
# #             augment = random.randint(0, len(self.augments) - 1)
# #             target, input_ = self.augments[augment](target, input_)
# #         return target, input_

# # class Gradient_Difference_Loss(nn.Module):
# #     def __init__(self, alpha=1, chans=3, cuda=True):
# #         super(Gradient_Difference_Loss, self).__init__()
# #         self.alpha = alpha
# #         self.chans = chans
# #         Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# #         SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
# #         SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
# #         self.Kx = torch.tensor(SobelX, dtype=torch.float32, device='cuda').expand(self.chans, 1, 3, 3)
# #         self.Ky = torch.tensor(SobelY, dtype=torch.float32, device='cuda').expand(self.chans, 1, 3, 3)

# #     def get_gradients(self, im):
# #         gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
# #         gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
# #         return gx, gy

# #     def forward(self, pred, true):
# #         # get graduent of pred and true
# #         gradX_true, gradY_true = self.get_gradients(true)
# #         grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
# #         gradX_pred, gradY_pred = self.get_gradients(pred)
# #         grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
# #         # compute and return GDL
# #         return 0.5 * torch.mean((grad_true - grad_pred_a) ** 2)
# # import torch
# # import torch.nn as nn
# # import torch.fft

# # import torch
# # import torch.nn as nn


# # class SpectrumLoss(nn.Module):
# #     def __init__(self, lambda_high=1.0, lambda_low=1.0, lambda_smooth=0.1):
# #         super(SpectrumLoss, self).__init__()
# #         self.lambda_high = lambda_high
# #         self.lambda_low = lambda_low
# #         self.lambda_smooth = lambda_smooth

# #     def forward(self, input_image, enhanced_image):
# #         # 确保输入图像和增强图像在同一设备
# #         device = input_image.device

# #         # 计算频谱
# #         fft_input = torch.fft.fft2(input_image, norm='ortho').to(device)
# #         fft_enhanced = torch.fft.fft2(enhanced_image, norm='ortho').to(device)

# #         # 获取频谱幅度
# #         amp_input = torch.abs(fft_input).to(device)
# #         amp_enhanced = torch.abs(fft_enhanced).to(device)

# #         # 分割高频和低频部分
# #         h, w = amp_input.shape[-2:]
# #         center_h, center_w = h // 2, w // 2
# #         radius = min(h, w) // 4  # 半径，控制高频和低频区域

# #         # 创建距离掩码并移动到相同设备
# #         y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
# #         distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)

# #         high_freq_mask = (distance > radius).float().to(device)
# #         low_freq_mask = (distance <= radius).float().to(device)

# #         # 高频增强损失
# #         high_loss = torch.mean(high_freq_mask * (amp_input - amp_enhanced) ** 2)

# #         # 低频保留损失
# #         low_loss = torch.mean(low_freq_mask * (amp_input - amp_enhanced) ** 2)

# #         # 频谱平滑损失
# #         grad_x = torch.abs(amp_enhanced[:, :, 1:, :] - amp_enhanced[:, :, :-1, :])
# #         grad_y = torch.abs(amp_enhanced[:, :, :, 1:] - amp_enhanced[:, :, :, :-1])
# #         smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)

# #         # 总损失
# #         total_loss = self.lambda_high * high_loss + self.lambda_low * low_loss + self.lambda_smooth * smooth_loss

# #         return total_loss


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import cv2
# # import numpy as np

# # import torch
# # import cv2
# # import numpy as np

# # class LabContrastiveLoss(nn.Module):
# #     def __init__(self, lambd=0.005, temperature=0.001):
# #         super(LabContrastiveLoss, self).__init__()
# #         self.lambd = lambd  # Regularization strength for off-diagonal loss
# #         self.temperature = temperature  # Temperature scaling for contrastive loss

# #     def rgb_to_lab(self, rgb):
# #         """Convert RGB to Lab color space and normalize channels."""
# #         rgb = rgb.permute(0, 2, 3, 1) / 255.0  # Convert from [B, C, H, W] to [B, H, W, C]
# #         B, H, W, C = rgb.shape

# #         # Convert RGB to Lab using OpenCV (CPU-based operation)
# #         lab_images = []
# #         for i in range(B):
# #             lab_image = cv2.cvtColor(rgb[i].detach().cpu().numpy(), cv2.COLOR_RGB2Lab)  # RGB to Lab
# #             lab_images.append(lab_image)

# #         lab = np.stack(lab_images, axis=0)  # Shape: [B, H, W, C]
# #         lab = torch.tensor(lab, dtype=torch.float32).permute(0, 3, 1, 2).cuda()  # Convert back to [B, C, H, W]

# #         # Normalize L channel to [0, 1] and A, B channels to [-1, 1]
# #         lab[:, 0, :, :] = lab[:, 0, :, :] / 100.0  # Normalize L channel to [0, 1]
# #         lab[:, 1:, :, :] = lab[:, 1:, :, :] / 128.0  # Normalize A and B channels to [-1, 1]

# #         return lab

# #     def compute_lab_stats(self, dataset):
# #         """Compute the mean and std of L channel for the entire dataset."""
# #         L_means = []
# #         L_stds = []

# #         for img in dataset:
# #             lab = self.rgb_to_lab(img)  # Convert to Lab space
# #             L_channel = lab[:, 0, :, :]  # Extract L channel

# #             # Compute mean and std for each image
# #             L_mean = torch.mean(L_channel, dim=(1, 2))
# #             L_std = torch.std(L_channel, dim=(1, 2))
# #             L_means.append(L_mean)
# #             L_stds.append(L_std)

# #         # Compute mean and std for the entire dataset
# #         L_mean = torch.mean(torch.stack(L_means), dim=0)
# #         L_std = torch.mean(torch.stack(L_stds), dim=0)

# #         return L_mean, L_std

# #     def compute_similarity(self, gen_mean, gen_std, target_mean, target_std):
# #         """Compute cosine similarity between generated and target statistics (mean and std)."""
# #         # Normalize mean and std
# #         gen_mean = F.normalize(gen_mean.unsqueeze(1), p=2, dim=1)
# #         gen_std = F.normalize(gen_std.unsqueeze(1), p=2, dim=1)
# #         target_mean = F.normalize(target_mean.unsqueeze(1), p=2, dim=1)
# #         target_std = F.normalize(target_std.unsqueeze(1), p=2, dim=1)

# #         # Compute cosine similarity
# #         similarity_mean = torch.matmul(gen_mean, target_mean.T) / self.temperature
# #         similarity_std = torch.matmul(gen_std, target_std.T) / self.temperature

# #         return similarity_mean, similarity_std

# #     def forward(self, generated_img, target_img, dataset):
# #         """Compute the contrastive loss based on Lab channel means and stds."""
# #         # Compute stats (mean and std) for the entire dataset
# #         target_mean, target_std = self.compute_lab_stats(dataset)

# #         # Compute stats for the generated image
# #         generated_lab = self.rgb_to_lab(generated_img)
# #         gen_mean = torch.mean(generated_lab[:, 0, :, :], dim=(2, 3))
# #         gen_std = torch.std(generated_lab[:, 0, :, :], dim=(2, 3))

# #         # Compute similarity for mean and std (L)
# #         similarity_mean_L, similarity_std_L = self.compute_similarity(gen_mean, gen_std, target_mean, target_std)

# #         # Diagonal loss (on-diagonal similarity should be 1)
# #         on_diag_L_mean = torch.diagonal(similarity_mean_L).add_(-1).pow_(2).sum()
# #         on_diag_L_std = torch.diagonal(similarity_std_L).add_(-1).pow_(2).sum()

# #         # Off-diagonal loss (minimize cross-correlation between different instances)
# #         off_diag_L_mean = similarity_mean_L[torch.tril_indices(similarity_mean_L.size(0), similarity_mean_L.size(1), -1)].pow_(2).sum()
# #         off_diag_L_std = similarity_std_L[torch.tril_indices(similarity_std_L.size(0), similarity_std_L.size(1), -1)].pow_(2).sum()

# #         # Combine all loss terms
# #         loss = (on_diag_L_mean + on_diag_L_std) + self.lambd * (off_diag_L_mean + off_diag_L_std)

# #         return loss


# # def gradient(input_tensor, direction):
# #     # Create gradient kernels for x and y directions (2x2 kernels)
# #     # For 3 input channels, we need the kernel to have 3 input channels as well.
# #     # The output channels are 1 (we only need one gradient per direction)

# #     # Gradient kernel in x-direction (for each of the 3 channels)
# #     smooth_kernel_x = torch.tensor([[[[0, 0], [-1, 1]]]] * input_tensor.shape[1], dtype=torch.float32).to(
# #         input_tensor.device)  # Shape: [3, 1, 2, 2]

# #     # Gradient kernel in y-direction (transpose of x-direction kernel)
# #     smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)  # Swap x and y for gradient in y-direction

# #     if direction == "x":
# #         kernel = smooth_kernel_x
# #     elif direction == "y":
# #         kernel = smooth_kernel_y

# #     # Apply convolution with padding=1 to account for the kernel size (2x2)
# #     gradient_orig = torch.abs(
# #         F.conv2d(input_tensor, kernel, stride=1, padding=1, groups=input_tensor.shape[1]))  # padding=1 for 2x2 kernel

# #     # Normalize the gradients
# #     grad_min = torch.min(gradient_orig)
# #     grad_max = torch.max(gradient_orig)
# #     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))  # Normalize to [0, 1]

# #     return grad_norm


# # class SmoothLoss(nn.Module):
# #     """Illumination smoothness"""

# #     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
# #         super(SmoothLoss, self).__init__()
# #         self.loss_weight = loss_weight
# #         self.eps = eps
# #         self.reduction = reduction

# #     def forward(self, illu, img):
# #         # illu: b×c×h×w   illumination map
# #         # img:  b×c×h×w   input image

# #         illu_gradient_x = gradient(illu, "x")
# #         img_gradient_x = gradient(img, "x")

# #         # Convert 0.01 to tensor for compatibility with torch.maximum
# #         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, torch.tensor(0.01, dtype=img.dtype,
# #                                                                                                  device=img.device))))

# #         illu_gradient_y = gradient(illu, "y")
# #         img_gradient_y = gradient(img, "y")

# #         # Convert 0.01 to tensor for compatibility with torch.maximum
# #         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, torch.tensor(0.01, dtype=img.dtype,
# #                                                                                                  device=img.device))))

# #         loss = torch.mean(x_loss + y_loss) * self.loss_weight

# #         return loss

# # class MultualLoss(nn.Module):
# #     """ Multual Consistency"""

# #     def __init__(self, loss_weight=0.20, reduction='mean'):
# #         super(MultualLoss,self).__init__()

# #         self.loss_weight = loss_weight
# #         self.reduction = reduction


# #     def forward(self, illu):
# #         # illu: b x c x h x w
# #         gradient_x = gradient(illu,"x")
# #         gradient_y = gradient(illu,"y")

# #         x_loss = gradient_x * torch.exp(-10*gradient_x)
# #         y_loss = gradient_y * torch.exp(-10*gradient_y)

# #         loss = torch.mean(x_loss+y_loss) * self.loss_weight
# #         return loss

# # class ImageCleanModel(BaseModel):
# #     """Base Deblur model for single image deblur."""

# #     def __init__(self, opt):
# #         super(ImageCleanModel, self).__init__(opt)

# #         # define mixed precision
# #         self.use_amp = opt.get('use_amp', False) and load_amp
# #         self.amp_scaler = GradScaler(enabled=self.use_amp)
# #         if self.use_amp:
# #             print('Using Automatic Mixed Precision')
# #         else:
# #             print('Not using Automatic Mixed Precision')

# #         # define network
# #         self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
# #         if self.mixing_flag:
# #             mixup_beta = self.opt['train']['mixing_augs'].get(
# #                 'mixup_beta', 1.2)
# #             use_identity = self.opt['train']['mixing_augs'].get(
# #                 'use_identity', False)
# #             self.mixing_augmentation = Mixing_Augment(
# #                 mixup_beta, use_identity, self.device)

# #         self.net_g = define_network(deepcopy(opt['network_g']))
# #         self.net_g = self.model_to_device(self.net_g)
# #         # self.print_network(self.net_g)

# #         # load pretrained models
# #         load_path = self.opt['path'].get('pretrain_network_g', None)
# #         if load_path is not None:
# #             self.load_network(self.net_g, load_path,
# #                               self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

# #         if self.is_train:
# #             self.init_training_settings()

# #     def init_training_settings(self):
# #         self.net_g.train()
# #         train_opt = self.opt['train']

# #         self.ema_decay = train_opt.get('ema_decay', 0)
# #         if self.ema_decay > 0:
# #             logger = get_root_logger()
# #             logger.info(
# #                 f'Use Exponential Moving Average with decay: {self.ema_decay}')
# #             # define network net_g with Exponential Moving Average (EMA)
# #             # net_g_ema is used only for testing on one GPU and saving
# #             # There is no need to wrap with DistributedDataParallel
# #             self.net_g_ema = define_network(self.opt['network_g']).to(
# #                 self.device)
# #             # load pretrained model
# #             load_path = self.opt['path'].get('pretrain_network_g', None)
# #             if load_path is not None:
# #                 self.load_network(self.net_g_ema, load_path,
# #                                   self.opt['path'].get('strict_load_g',
# #                                                        True), 'params_ema')
# #             else:
# #                 self.model_ema(0)  # copy net_g weight
# #             self.net_g_ema.eval()

# #         # define losses
# #         if train_opt.get('pixel_opt'):
# #             pixel_type = train_opt['pixel_opt'].pop('type')
# #             cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
# #             self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
# #                 self.device)      #如何写 weighted loss 呢？传参构造Loss函数
# #         else:
# #             raise ValueError('pixel loss are None.')

# #         # set up optimizers and schedulers
# #         self.setup_optimizers()
# #         self.setup_schedulers()

# #     def setup_optimizers(self):
# #         train_opt = self.opt['train']
# #         optim_params = []

# #         for k, v in self.net_g.named_parameters():
# #             if v.requires_grad:
# #                 optim_params.append(v)
# #             else:
# #                 logger = get_root_logger()
# #                 logger.warning(f'Params {k} will not be optimized.')

# #         optim_type = train_opt['optim_g'].pop('type')
# #         if optim_type == 'Adam':
# #             self.optimizer_g = torch.optim.Adam(
# #                 optim_params, **train_opt['optim_g'])
# #         elif optim_type == 'AdamW':
# #             self.optimizer_g = torch.optim.AdamW(
# #                 optim_params, **train_opt['optim_g'])
# #         else:
# #             raise NotImplementedError(
# #                 f'optimizer {optim_type} is not supperted yet.')
# #         self.optimizers.append(self.optimizer_g)

# #     def feed_train_data(self, data):
# #         self.lq = data['lq'].to(self.device)
# #         if 'gt' in data:
# #             self.gt = data['gt'].to(self.device)

# #         if self.mixing_flag:
# #             self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

# #     def feed_data(self, data):
# #         self.lq = data['lq'].to(self.device)
# #         if 'gt' in data:
# #             self.gt = data['gt'].to(self.device)

# #     def optimize_parameters(self, current_iter):
# #         self.optimizer_g.zero_grad()
# #         with autocast(enabled=self.use_amp):
# #             preds = self.net_g(self.lq)[0]
# #             illu= self.net_g(self.lq)[2]
# #             img= self.net_g(self.lq)[1]
# #             if not isinstance(preds, list):
# #                 preds = [preds]

# #             self.output = preds[-1]

# #             loss_dict = OrderedDict()
# #             # pixel loss
# #             l_pix = 0.
# #             for pred in preds:
# #                 l_pix += self.cri_pix(pred, self.gt) #此处统计batch的loss
# #             grad_loss = Gradient_Difference_Loss(alpha=1, chans=3, cuda=True)
# #             # contrastive_loss_fn = LabContrastiveLoss(lambd=0.005, temperature=0.07)
# #             smooth_loss_fn = SmoothLoss(loss_weight=0.15, reduction='mean', eps=1e-2)
# #             lossSM = smooth_loss_fn(illu, img)
# #             multual_loss = MultualLoss(loss_weight=0.20, reduction='mean')
# #             lossml = multual_loss(illu)
# #             spectrum_loss = SpectrumLoss(lambda_high=1.0, lambda_low=1.0, lambda_smooth=0.1)
# #             lossspectrum = spectrum_loss(self.gt, pred)
# #             # lossCL = contrastive_loss_fn(pred, self.gt)
# #             # losssmooth = loss_fn(pred, self.gt)
# #             edgeloss= grad_loss(pred, self.gt)
# #             loss_dict[' edgeloss'] = edgeloss
# #             # loss_dict[' lossCL'] = lossCL
# #             loss_dict['l_pix'] = l_pix
# #             loss_dict['lossSM '] = l_pix
# #             loss_dict[' lossml '] =  lossml
# #             loss_dict[' lossspectrum '] =  lossspectrum
# #             total_loss = l_pix + 0.2*edgeloss + 0.15*lossSM + 0.2*lossml + 0.1* lossspectrum
# #             loss_dict['total_loss'] = total_loss
# #         self.amp_scaler.scale(total_loss ).backward()
# #         self.amp_scaler.unscale_(self.optimizer_g) # 在梯度裁剪前先unscale梯度
# #         # l_pix.backward()

# #         if self.opt['train']['use_grad_clip']:
# #             torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
# #         # self.optimizer_g.step()
# #         self.amp_scaler.step(self.optimizer_g)
# #         self.amp_scaler.update()

# #         self.log_dict = self.reduce_loss_dict(loss_dict)

# #         if self.ema_decay > 0:
# #             self.model_ema(decay=self.ema_decay)
# #     def pad_test(self, window_size):
# #         scale = self.opt.get('scale', 1)
# #         mod_pad_h, mod_pad_w = 0, 0
# #         _, _, h, w = self.lq.size()

# #         # 计算需要填充的高度和宽度
# #         if h % window_size != 0:
# #             mod_pad_h = window_size - h % window_size
# #         if w % window_size != 0:
# #             mod_pad_w = window_size - w % window_size

# #         # 使用反射填充图像
# #         img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
# #         self.nonpad_test(img)

# #         # 解包 self.output 如果它是元组
# #         if isinstance(self.output, tuple):
# #             self.output = self.output[0]  # 获取第一个张量部分（增强图）

# #         # 对 output 做裁剪，去掉填充部分
# #         _, _, h, w = self.output.size()
# #         self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

# #     def nonpad_test(self, img=None):
# #         if img is None:
# #             img = self.lq  # 默认使用 `self.lq` 作为输入图像

# #         # 将输入图像转换为 float32 类型
# #         img = img.to(torch.float32)

# #         # 使用 ema 网络进行推理
# #         if hasattr(self, 'net_g_ema'):
# #             self.net_g_ema.eval()
# #             with torch.no_grad():
# #                 # 仅传递图像给网络
# #                 pred = self.net_g_ema(img)
# #             if isinstance(pred, list):
# #                 pred = pred[-1]
# #             # 如果模型输出是元组，提取第一个（增强图），丢弃第二个（深度图）
# #             if isinstance(pred, tuple):
# #                 self.output = pred[0]  # 增强图像
# #                 self.depth_map = pred[1]  # 深度图像（如果需要）
# #             else:
# #                 self.output = pred  # 增强图像
# #                 self.depth_map = None  # 如果没有深度图，可以设置为 None
# #         else:
# #             self.net_g.eval()
# #             with torch.no_grad():
# #                 # 仅传递图像给网络
# #                 pred = self.net_g(img)
# #             if isinstance(pred, list):
# #                 pred = pred[-1]
# #             # 如果模型输出是元组，提取第一个（增强图），丢弃第二个（深度图）
# #             if isinstance(pred, tuple):
# #                 self.output = pred[0]  # 增强图像
# #                 self.depth_map = pred[1]  # 深度图像（如果需要）
# #             else:
# #                 self.output = pred  # 增强图像
# #                 self.depth_map = None  # 如果没有深度图，可以设置为 None

# #         # 恢复模型为训练模式
# #         self.net_g.train()  # 恢复训练模式

# #     # def pad_test(self, window_size):
# #     #     scale = self.opt.get('scale', 1)
# #     #     mod_pad_h, mod_pad_w = 0, 0
# #     #     _, _, h, w = self.lq.size()
# #     #     if h % window_size != 0:
# #     #         mod_pad_h = window_size - h % window_size
# #     #     if w % window_size != 0:
# #     #         mod_pad_w = window_size - w % window_size
# #     #     img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
# #     #     self.nonpad_test(img)
# #     #     _, _, h, w = self.output.size()
# #     #     self.output = self.output[:, :, 0:h -
# #     #                               mod_pad_h * scale, 0:w - mod_pad_w * scale]

# #     # def nonpad_test(self, img=None):
# #     #     if img is None:
# #     #         img = self.lq
# #     #     if hasattr(self, 'net_g_ema'):
# #     #         self.net_g_ema.eval()
# #     #         with torch.no_grad():
# #     #             pred = self.net_g_ema(img)
# #     #         if isinstance(pred, list):
# #     #             pred = pred[-1]
# #     #         self.output = pred
# #     #     else:
# #     #         self.net_g.eval()
# #     #         with torch.no_grad():
# #     #             pred = self.net_g(img)
# #     #         if isinstance(pred, list):
# #     #             pred = pred[-1]
# #     #         self.output = pred
# #     #         self.net_g.train()

# #     def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
# #         if os.environ['LOCAL_RANK'] == '0':
# #             return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
# #         else:
# #             return 0.

# #     def nondist_validation(self, dataloader, current_iter, tb_logger,
# #                            save_img, rgb2bgr, use_image):
# #         dataset_name = dataloader.dataset.opt['name']
# #         with_metrics = self.opt['val'].get('metrics') is not None
# #         if with_metrics:
# #             self.metric_results = {
# #                 metric: 0
# #                 for metric in self.opt['val']['metrics'].keys()
# #             }
# #         # pbar = tqdm(total=len(dataloader), unit='image')

# #         window_size = self.opt['val'].get('window_size', 0)

# #         if window_size:
# #             test = partial(self.pad_test, window_size)
# #         else:
# #             test = self.nonpad_test

# #         cnt = 0

# #         for idx, val_data in enumerate(dataloader):
# #             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
# #             self.feed_data(val_data)
# #             test()

# #             visuals = self.get_current_visuals()
# #             sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
# #             if 'gt' in visuals:
# #                 gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
# #                 del self.gt

# #             # tentative for out of GPU memory
# #             del self.lq
# #             del self.output
# #             torch.cuda.empty_cache()
# #             if sr_img.shape != gt_img.shape:
# #                gt_img = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]))  # Resize GT to match SR
# #             if save_img:

# #                 if self.opt['is_train']:

# #                     save_img_path = osp.join(self.opt['path']['visualization'],
# #                                              img_name,
# #                                              f'{img_name}_{current_iter}.png')

# #                     save_gt_img_path = osp.join(self.opt['path']['visualization'],
# #                                                 img_name,
# #                                                 f'{img_name}_{current_iter}_gt.png')
# #                 else:

# #                     save_img_path = osp.join(
# #                         self.opt['path']['visualization'], dataset_name,
# #                         f'{img_name}.png')
# #                     save_gt_img_path = osp.join(
# #                         self.opt['path']['visualization'], dataset_name,
# #                         f'{img_name}_gt.png')

# #                 imwrite(sr_img, save_img_path)
# #                 imwrite(gt_img, save_gt_img_path)

# #             if with_metrics:
# #                 # calculate metrics
# #                 opt_metric = deepcopy(self.opt['val']['metrics'])
# #                 if use_image:
# #                     for name, opt_ in opt_metric.items():
# #                         metric_type = opt_.pop('type')
# #                         self.metric_results[name] += getattr(
# #                             metric_module, metric_type)(sr_img, gt_img, **opt_)
# #                 else:
# #                     for name, opt_ in opt_metric.items():
# #                         metric_type = opt_.pop('type')
# #                         self.metric_results[name] += getattr(
# #                             metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

# #             cnt += 1

# #         current_metric = 0.
# #         if with_metrics:
# #             for metric in self.metric_results.keys():
# #                 self.metric_results[metric] /= cnt
# #                 current_metric = self.metric_results[metric]

# #             self._log_validation_metric_values(current_iter, dataset_name,
# #                                                tb_logger)
# #         return current_metric

# #     def _log_validation_metric_values(self, current_iter, dataset_name,
# #                                       tb_logger):
# #         log_str = f'Validation {dataset_name},\t'
# #         for metric, value in self.metric_results.items():
# #             log_str += f'\t # {metric}: {value:.4f}'
# #         logger = get_root_logger()
# #         logger.info(log_str)
# #         if tb_logger:
# #             for metric, value in self.metric_results.items():
# #                 tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

# #     def get_current_visuals(self):
# #         out_dict = OrderedDict()
# #         out_dict['lq'] = self.lq.detach().cpu()
# #         out_dict['result'] = self.output.detach().cpu()
# #         if hasattr(self, 'gt'):
# #             out_dict['gt'] = self.gt.detach().cpu()
# #         return out_dict

# #     def save(self, epoch, current_iter, **kwargs):
# #         if self.ema_decay > 0:
# #             self.save_network([self.net_g, self.net_g_ema],
# #                               'net_g',
# #                               current_iter,
# #                               param_key=['params', 'params_ema'])
# #         else:
# #             self.save_network(self.net_g, 'net_g', current_iter)
# #         self.save_training_state(epoch, current_iter, **kwargs)

# #     def save_best(self, best_metric, param_key='params'):
# #         psnr = best_metric['psnr']
# #         cur_iter = best_metric['iter']
# #         save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
# #         exp_root = self.opt['path']['experiments_root']
# #         save_path = os.path.join(
# #             self.opt['path']['experiments_root'], save_filename)

# #         if not os.path.exists(save_path):
# #             for r_file in glob.glob(f'{exp_root}/best_*'):
# #                 os.remove(r_file)
# #             net = self.net_g

# #             net = net if isinstance(net, list) else [net]
# #             param_key = param_key if isinstance(
# #                 param_key, list) else [param_key]
# #             assert len(net) == len(
# #                 param_key), 'The lengths of net and param_key should be the same.'

# #             save_dict = {}
# #             for net_, param_key_ in zip(net, param_key):
# #                 net_ = self.get_bare_model(net_)
# #                 state_dict = net_.state_dict()
# #                 for key, param in state_dict.items():
# #                     if key.startswith('module.'):  # remove unnecessary 'module.'
# #                         key = key[7:]
# #                     state_dict[key] = param.cpu()
# #                 save_dict[param_key_] = state_dict

# #             torch.save(save_dict, save_path)
# import importlib
# import torch
# from collections import OrderedDict
# from copy import deepcopy
# from os import path as osp
# from tqdm import tqdm
# import glob

# from basicsr.models.archs import define_network
# from basicsr.models.base_model import BaseModel
# from basicsr.utils import get_root_logger, imwrite, tensor2img

# loss_module = importlib.import_module('basicsr.models.losses')
# metric_module = importlib.import_module('basicsr.metrics')

# import os
# import random
# import numpy as np
# import cv2
# import torch.nn.functional as F
# from functools import partial

# try :
#     from torch.cuda.amp import autocast, GradScaler
#     load_amp = True
# except:
#     load_amp = False


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


# class ImageCleanModel(BaseModel):
#     """Base Deblur model for single image deblur."""

#     def __init__(self, opt):
#         super(ImageCleanModel, self).__init__(opt)

#         # define mixed precision
#         self.use_amp = opt.get('use_amp', False) and load_amp
#         self.amp_scaler = GradScaler(enabled=self.use_amp)
#         if self.use_amp:
#             print('Using Automatic Mixed Precision')
#         else:
#             print('Not using Automatic Mixed Precision')

#         # define network
#         self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
#         if self.mixing_flag:
#             mixup_beta = self.opt['train']['mixing_augs'].get(
#                 'mixup_beta', 1.2)
#             use_identity = self.opt['train']['mixing_augs'].get(
#                 'use_identity', False)
#             self.mixing_augmentation = Mixing_Augment(
#                 mixup_beta, use_identity, self.device)

#         self.net_g = define_network(deepcopy(opt['network_g']))
#         self.net_g = self.model_to_device(self.net_g)
#         # self.print_network(self.net_g)

#         # load pretrained models
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         if load_path is not None:
#             self.load_network(self.net_g, load_path,
#                               self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

#         if self.is_train:
#             self.init_training_settings()

#     def init_training_settings(self):
#         self.net_g.train()
#         train_opt = self.opt['train']

#         self.ema_decay = train_opt.get('ema_decay', 0)
#         if self.ema_decay > 0:
#             logger = get_root_logger()
#             logger.info(
#                 f'Use Exponential Moving Average with decay: {self.ema_decay}')
#             # define network net_g with Exponential Moving Average (EMA)
#             # net_g_ema is used only for testing on one GPU and saving
#             # There is no need to wrap with DistributedDataParallel
#             self.net_g_ema = define_network(self.opt['network_g']).to(
#                 self.device)
#             # load pretrained model
#             load_path = self.opt['path'].get('pretrain_network_g', None)
#             if load_path is not None:
#                 self.load_network(self.net_g_ema, load_path,
#                                   self.opt['path'].get('strict_load_g',
#                                                        True), 'params_ema')
#             else:
#                 self.model_ema(0)  # copy net_g weight
#             self.net_g_ema.eval()

#         # define losses
#         if train_opt.get('pixel_opt'):
#             pixel_type = train_opt['pixel_opt'].pop('type')
#             cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
#             self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
#                 self.device)      #如何写 weighted loss 呢？传参构造Loss函数
#         else:
#             raise ValueError('pixel loss are None.')

#         # set up optimizers and schedulers
#         self.setup_optimizers()
#         self.setup_schedulers()

#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []

#         for k, v in self.net_g.named_parameters():
#             if v.requires_grad:
#                 optim_params.append(v)
#             else:
#                 logger = get_root_logger()
#                 logger.warning(f'Params {k} will not be optimized.')

#         optim_type = train_opt['optim_g'].pop('type')
#         if optim_type == 'Adam':
#             self.optimizer_g = torch.optim.Adam(
#                 optim_params, **train_opt['optim_g'])
#         elif optim_type == 'AdamW':
#             self.optimizer_g = torch.optim.AdamW(
#                 optim_params, **train_opt['optim_g'])
#         else:
#             raise NotImplementedError(
#                 f'optimizer {optim_type} is not supperted yet.')
#         self.optimizers.append(self.optimizer_g)

#     def feed_train_data(self, data):
#         self.lq = data['lq'].to(self.device)
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)

#         if self.mixing_flag:
#             self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

#     def feed_data(self, data):
#         self.lq = data['lq'].to(self.device)
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)

#     def optimize_parameters(self, current_iter):
#         self.optimizer_g.zero_grad()

#         with autocast(enabled=self.use_amp):
#             preds = self.net_g(self.lq)
#             if not isinstance(preds, list):
#                 preds = [preds]

#             self.output = preds[-1]

#             loss_dict = OrderedDict()
#             # pixel loss
#             l_pix = 0.
#             for pred in preds:
#                 l_pix += self.cri_pix(pred, self.gt) #此处统计batch的loss

#             loss_dict['l_pix'] = l_pix

#         self.amp_scaler.scale(l_pix).backward()
#         self.amp_scaler.unscale_(self.optimizer_g) # 在梯度裁剪前先unscale梯度
#         # l_pix.backward()

#         if self.opt['train']['use_grad_clip']:
#             torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
#         # self.optimizer_g.step()
#         self.amp_scaler.step(self.optimizer_g)
#         self.amp_scaler.update()

#         self.log_dict = self.reduce_loss_dict(loss_dict)

#         if self.ema_decay > 0:
#             self.model_ema(decay=self.ema_decay)

#     def pad_test(self, window_size):
#         scale = self.opt.get('scale', 1)
#         mod_pad_h, mod_pad_w = 0, 0
#         _, _, h, w = self.lq.size()
#         if h % window_size != 0:
#             mod_pad_h = window_size - h % window_size
#         if w % window_size != 0:
#             mod_pad_w = window_size - w % window_size
#         img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         self.nonpad_test(img)
#         _, _, h, w = self.output.size()
#         self.output = self.output[:, :, 0:h -
#                                   mod_pad_h * scale, 0:w - mod_pad_w * scale]

#     def nonpad_test(self, img=None):
#         if img is None:
#             img = self.lq
#         if hasattr(self, 'net_g_ema'):
#             self.net_g_ema.eval()
#             with torch.no_grad():
#                 pred = self.net_g_ema(img)
#             if isinstance(pred, list):
#                 pred = pred[-1]
#             self.output = pred
#         else:
#             self.net_g.eval()
#             with torch.no_grad():
#                 pred = self.net_g(img)
#             if isinstance(pred, list):
#                 pred = pred[-1]
#             self.output = pred
#             self.net_g.train()

#     def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
#         if os.environ['LOCAL_RANK'] == '0':
#             return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
#         else:
#             return 0.

#     def nondist_validation(self, dataloader, current_iter, tb_logger,
#                            save_img, rgb2bgr, use_image):
#         dataset_name = dataloader.dataset.opt['name']
#         with_metrics = self.opt['val'].get('metrics') is not None
#         if with_metrics:
#             self.metric_results = {
#                 metric: 0
#                 for metric in self.opt['val']['metrics'].keys()
#             }
#         # pbar = tqdm(total=len(dataloader), unit='image')

#         window_size = self.opt['val'].get('window_size', 0)

#         if window_size:
#             test = partial(self.pad_test, window_size)
#         else:
#             test = self.nonpad_test

#         cnt = 0

#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
#             self.feed_data(val_data)
#             test()

#             visuals = self.get_current_visuals()
#             sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
#             if 'gt' in visuals:
#                 gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
#                 del self.gt

#             # tentative for out of GPU memory
#             del self.lq
#             del self.output
#             torch.cuda.empty_cache()

#             if save_img:

#                 if self.opt['is_train']:

#                     save_img_path = osp.join(self.opt['path']['visualization'],
#                                              img_name,
#                                              f'{img_name}_{current_iter}.png')

#                     save_gt_img_path = osp.join(self.opt['path']['visualization'],
#                                                 img_name,
#                                                 f'{img_name}_{current_iter}_gt.png')
#                 else:

#                     save_img_path = osp.join(
#                         self.opt['path']['visualization'], dataset_name,
#                         f'{img_name}.png')
#                     save_gt_img_path = osp.join(
#                         self.opt['path']['visualization'], dataset_name,
#                         f'{img_name}_gt.png')

#                 imwrite(sr_img, save_img_path)
#                 imwrite(gt_img, save_gt_img_path)

#             if with_metrics:
#                 # calculate metrics
#                 opt_metric = deepcopy(self.opt['val']['metrics'])
#                 if use_image:
#                     for name, opt_ in opt_metric.items():
#                         metric_type = opt_.pop('type')
#                         self.metric_results[name] += getattr(
#                             metric_module, metric_type)(sr_img, gt_img, **opt_)
#                 else:
#                     for name, opt_ in opt_metric.items():
#                         metric_type = opt_.pop('type')
#                         self.metric_results[name] += getattr(
#                             metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

#             cnt += 1

#         current_metric = 0.
#         if with_metrics:
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= cnt
#                 current_metric = self.metric_results[metric]

#             self._log_validation_metric_values(current_iter, dataset_name,
#                                                tb_logger)
#         return current_metric

#     def _log_validation_metric_values(self, current_iter, dataset_name,
#                                       tb_logger):
#         log_str = f'Validation {dataset_name},\t'
#         for metric, value in self.metric_results.items():
#             log_str += f'\t # {metric}: {value:.4f}'
#         logger = get_root_logger()
#         logger.info(log_str)
#         if tb_logger:
#             for metric, value in self.metric_results.items():
#                 tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

#     def get_current_visuals(self):
#         out_dict = OrderedDict()
#         out_dict['lq'] = self.lq.detach().cpu()
#         out_dict['result'] = self.output.detach().cpu()
#         if hasattr(self, 'gt'):
#             out_dict['gt'] = self.gt.detach().cpu()
#         return out_dict

#     def save(self, epoch, current_iter, **kwargs):
#         if self.ema_decay > 0:
#             self.save_network([self.net_g, self.net_g_ema],
#                               'net_g',
#                               current_iter,
#                               param_key=['params', 'params_ema'])
#         else:
#             self.save_network(self.net_g, 'net_g', current_iter)
#         self.save_training_state(epoch, current_iter, **kwargs)

#     def save_best(self, best_metric, param_key='params'):
#         psnr = best_metric['psnr']
#         cur_iter = best_metric['iter']
#         save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
#         exp_root = self.opt['path']['experiments_root']
#         save_path = os.path.join(
#             self.opt['path']['experiments_root'], save_filename)

#         if not os.path.exists(save_path):
#             for r_file in glob.glob(f'{exp_root}/best_*'):
#                 os.remove(r_file)
#             net = self.net_g

#             net = net if isinstance(net, list) else [net]
#             param_key = param_key if isinstance(
#                 param_key, list) else [param_key]
#             assert len(net) == len(
#                 param_key), 'The lengths of net and param_key should be the same.'

#             save_dict = {}
#             for net_, param_key_ in zip(net, param_key):
#                 net_ = self.get_bare_model(net_)
#                 state_dict = net_.state_dict()
#                 for key, param in state_dict.items():
#                     if key.startswith('module.'):  # remove unnecessary 'module.'
#                         key = key[7:]
#                     state_dict[key] = param.cpu()
#                 save_dict[param_key_] = state_dict

#             torch.save(save_dict, save_path)


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


def local_consistency_loss_L_channel(enhanced_image, mask, window_size=3):
    """
    计算L通道上的局部一致性损失，用于约束增强图像中的极端黑暗区域。

    Args:
        enhanced_image (torch.Tensor): 增强后的图像 (B, C, H, W)
        mask (torch.Tensor): 极端黑暗区域的掩码 (B, 1, H, W)
        window_size (int): 局部窗口大小

    Returns:
        loss (torch.Tensor): 局部一致性损失
    """
    pad = window_size // 2
    enhanced_padded = F.pad(enhanced_image, (pad, pad, pad, pad), mode='reflect')

    # 将 RGB 图像转换为 L 通道
    L_channel = rgb_to_lab(enhanced_image)  # 计算增强图像的 L 通道

    # 计算局部窗口内的 L 通道均值
    local_mean = F.avg_pool2d(enhanced_padded, kernel_size=window_size, stride=1, padding=0)

    # 只对掩码指定区域计算一致性损失
    consistency_loss = F.l1_loss(L_channel * mask, local_mean * mask)

    return consistency_loss


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


# def calc_psnr(pred, gt, is_for_torch=True):
#     if is_for_torch:
#         pred = pred[0].permute(1, 2, 0).detach().numpy()
#         gt = gt[0].premute(1, 2, 0).detach().numpy()

#         psnr = peak_signal_noise_ratio(gt, pred)
#     else:
#         psnr = peak_signal_noise_ratio(gt, pred)

#     return psnr

# def calc_ssim(pred, gt, is_for_torch=True):
#     if is_for_torch:
#         pred = pred[0].permute(1, 2, 0).detach().numpy()
#         gt = gt[0].premute(1, 2, 0).detach().numpy()

#         ssim = structural_similarity(gt, pred, multichannel=True)
#     else:
#         ssim = structural_similarity(gt, pred, multichannel=True)

#     return ssim

def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        im_max = torch.max(img)
        im_min = torch.min(img)

        img = (img - im_min) / (im_max - im_min + 1e-7)

    return img


def preprocessing(d_img_org):
    d_img_org = padding_img(d_img_org)
    x_his = build_historgram(d_img_org)
    return {
        'x': d_img_org,
        'x_his': x_his
    }


def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32

    left_pad = (w_out - w) // 2
    right_pad = w_out - w - left_pad
    top_pad = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad

    img = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)

    return img


def build_historgram(img):
    with torch.no_grad():
        b, _, _, _ = img.shape

        r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
        g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
        b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)

        historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)

        for i in range(1, b):
            r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
            g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
            b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)

            historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
            historgram = torch.cat((historgram, historgram_temp), dim=0)

    return historgram


class RankerConditionalLoss(nn.Module):
    def __init__(self, opt_path, checkpoint_path):
        super().__init__()
        self.device = torch.device('cuda')
        options = get_option(opt_path)
        options['model']['model_name'] = 'URanker'
        options['model']['resume_ckpt_path'] = checkpoint_path
        self.model = build_model(options['model']).to(self.device).eval()

    # 🐾 分数差值：学生必须甩老师 margin 分数才行！

    def forward(self, enhanced_img, teacher_img=None):  # teacher_img 不再需要啦宝贝~
        # 取前三通道并 resize
        enhanced_img = F.interpolate(enhanced_img[:, :3], size=(256, 256), mode='bilinear', align_corners=False)

        # 预处理
        enhanced_inputs = preprocessing(enhanced_img)

        # 学生分数（带梯度）
        enhanced_score = self.model(**enhanced_inputs)['final_result'].view(-1)

        # ⚠️ 目标分数是 0.5，鼓励学生比这个高！
        margin = 0.5
        loss = F.relu(margin - enhanced_score)  # 你低于 0.5？那就罚你丫的！

        return loss.mean()

import torch.nn as nn
from torch.autograd import Function
class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 1)  # 最后输出一个 map，用于 PatchGAN 风格
        )

    def forward(self, x):
        return self.net(x)
class Discriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int = 512):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1)  # 输出一个标量，用于 Wasserstein loss
        )
        self._init_params()

    def forward(self, x):
        return self.net(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 10}]
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)

    # 安全判断维度：图像 or 特征
    if real_samples.dim() == 4:  # [B, C, H, W]
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    elif real_samples.dim() == 2:  # [B, F]
        alpha = torch.rand(batch_size, 1, device=device)
    else:
        raise ValueError(f"Unsupported input dim: {real_samples.dim()}")

    # 插值样本
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    # 判别器输出
    d_interpolates = D(interpolates)

    # 判别器输出必须是 scalar 或 [B, 1]
    if d_interpolates.dim() > 1:
        d_interpolates = d_interpolates.view(batch_size, -1).mean(1, keepdim=True)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 展平梯度
    gradients = gradients.view(batch_size, -1)

    # 加 clamp，防止过大
    gradients = torch.clamp(gradients, -10, 10)

    # 计算 norm + 稳定项
    gradient_norm = gradients.norm(2, dim=1) + 1e-6

    # 梯度惩罚
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    del interpolates, gradients, d_interpolates
    torch.cuda.empty_cache()
    return gradient_penalty

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_domain = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)  # 直接三通道输入
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img):
        # 输入 img 形状应为 [B, 3, H, W]，数值范围最好归一化到0~1，不然卷积吃不消，懂吧？
        features = self.conv_domain(img)
        features = self.global_avg_pool(features)
        features = features.view(features.size(0), -1)  # [B, 256]
        return features
def extract_feature(model, input_image, layer='encoder'):
    """
    从增强模型中提取某一层的中间特征。
    默认从 'encoder' 或 'backbone' 输出。
    """
    if hasattr(model, layer):
        feature_extractor = getattr(model, layer)
        feature = feature_extractor(input_image)
        return feature
    else:
        raise ValueError(f"Model does not have layer '{layer}'")
def grl_hook(coeff):
    def hook(grad):
        return -coeff * grad.clone()
    return hook

import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
class VGGContentLoss(nn.Module):
    def __init__(self, layer='relu3_3'):
        super().__init__()
        self.layer_name = layer
        self.layer_idx = self._get_layer_index(layer)

        vgg_all = vgg19(weights=VGG19_Weights.DEFAULT).features[:self.layer_idx + 1]
        self.vgg = vgg_all.eval().to(torch.float32).to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False
    def _get_layer_index(self, layer_name):
        layer_dict = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13,
            'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22,
            'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31,
            'relu5_3': 33, 'relu5_4': 35,
        }
        return layer_dict[layer_name]
    def forward(self, input, target):
        input = input.float()
        target = target.float()
        # 保证 batch size 一致
        min_batch = min(input.size(0), target.size(0))
        input = input[:min_batch]
        target = target[:min_batch]
        # 🔥 只取到目标层的特征图
        feat_input = self.vgg(input)
        feat_target = self.vgg(target)
        return F.l1_loss(feat_input, feat_target)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------------
# VGG 特征提取
# -------------------------------
STYLE_LAYERS = [3, 8, 15, 22]  # VGG16 conv1_2, conv2_2, conv3_3, conv4_3

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleDict()
        last = 0
        for layer in STYLE_LAYERS:
            idx = int(layer)
            block = nn.Sequential(*[vgg[i] for i in range(last, idx + 1)])
            self.slices[str(layer)] = block  # 将层名转换为字符串
            last = idx + 1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = {}
        h = x
        for ln in STYLE_LAYERS:
            h = self.slices[str(ln)](h)  # 将层名转换为字符串
            feats[ln] = h
        return feats

# -------------------------------
# Gram 矩阵
# -------------------------------
import torch
import torch.nn.functional as F


def gram_matrix(feat: torch.Tensor, epsilon=1e-6):
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    G = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
    # 归一化到合理范围
    G = G / (torch.norm(G, dim=(1,2), keepdim=True) + epsilon)
    return G

def normalize_vgg_output(feat):
    """
    Normalize the VGG feature output by channel-wise mean and std.
    feat: dict of {layer_name: feature}, where feature is a tensor
    """
    normalized_feats = {}
    for layer_name, feature in feat.items():
        # 计算均值和标准差
        mean = feature.mean(dim=[2, 3], keepdim=True)
        std = feature.std(dim=[2, 3], keepdim=True)

        # 标准化
        normalized_feats[layer_name] = (feature - mean) / (std + 1e-6)

    return normalized_feats


def check_vgg_output(vgg_output):
    # 如果 vgg_output 是 list 类型, 直接遍历
    for i, output in enumerate(vgg_output):
        print(f"Layer {i} - min: {output.min()}, max: {output.max()}")
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            print(f"NaN or Inf detected in layer {i}")
def check_input_validity(img):
    if torch.any(torch.isnan(img)) or torch.any(torch.isinf(img)):
        print("Invalid input detected (NaN or Inf)!")
    else:
        print("Input is valid.")


def vgg_loss(enh_ref1, enh_ref2, enh_aux, vgg_model):
    """
    enh_ref1, enh_ref2: (B,C,H,W) 有参考图增强结果
    enh_aux: (B,C,H,W) 无参考图增强结果
    vgg_model: VGGFeatures实例
    """
    # 提取 VGG 特征
    feats_ref1 = vgg_model(enh_ref1)
    feats_ref2 = vgg_model(enh_ref2)
    feats_aux = vgg_model(enh_aux)

    # 归一化 VGG 特征
    feats_ref1 = normalize_vgg_output(feats_ref1)
    feats_ref2 = normalize_vgg_output(feats_ref2)
    feats_aux = normalize_vgg_output(feats_aux)
    # 检查输入图像范围
    # print(enh_ref1.min(), enh_ref1.max())
    # print(enh_ref2.min(), enh_ref2.max())
    print(enh_aux.min(), enh_aux.max())

    loss = 0.0
    for f_ref1, f_ref2, f_aux in zip(feats_ref1.values(), feats_ref2.values(), feats_aux.values()):
        # Gram矩阵
        G_ref1 = gram_matrix(f_ref1)
        G_ref2 = gram_matrix(f_ref2)
        G_aux = gram_matrix(f_aux)
        B_aux = G_aux.shape[0]

        # 参考图的平均 Gram
        G_target = 0.5 * (G_ref1 + G_ref2)
        if G_target.shape[0] != B_aux:
            G_target = G_target.expand(B_aux, -1, -1)  # 重复到同样的 batch

        # 均方误差
        loss += F.mse_loss(G_aux, G_target)
    return loss


# -------------------------------
# 测试示例
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg = VGGFeatures().to(device).eval()

# 假设输入增强图
B, C, H, W = 2, 3, 256, 256
enh_ref1 = torch.randn(B, C, H, W).to(device)
enh_ref2 = torch.randn(B, C, H, W).to(device)
enh_aux  = torch.randn(B, C, H, W).to(device)

loss = vgg_loss(enh_ref1, enh_ref2, enh_aux, vgg)


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""
    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)
        self.feature_extractor = FeatureExtractor().cuda()
        self.net_d = Discriminator(in_feature=256, hidden_size=512).cuda()
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=1e-4)
        self.net_d_pix = PixelDiscriminator().to(self.device)  # 或别的判别器，按你的配置来
        self.net_d_pix.train()
        self.net_d_feat = Discriminator(in_feature=256, hidden_size=512).to(self.device)
        self.net_d_feat.train()
        self.optimizer_d_feat = torch.optim.Adam(self.net_d_feat.parameters(), lr=1e-4)
        self.optimizer_d_pix = torch.optim.Adam(self.net_d_pix.parameters(), lr=1e-4)
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

    # 学生
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

    # def optimize_parameters(self, current_iter):
    #     self.optimizer_g.zero_grad()
    #     # enhanced_input = self.lq1.clone()
    #     # preds = self.net_g(self.lq,self.depth)
    #     # 共享权重
    #     sim_per_epoch = []
    #     with autocast(enabled=self.use_amp):
    #         preds1, preds2, illu1, illu2, input1, input2, fire1, fire2, img1, img2, illu_fea1, illu_fea2 = self.net_g(self.lq1, self.lq2)
    #
    #         device = self.lq1.device  # 假设lq是输入图像，pred和gt会在相同设备上
    #         self.gt = self.gt.to(device)
    #         # device = self.lq2.device
    #         loss_dict = OrderedDict()
    #         # 开启梯度计算
    #         self.gt.requires_grad_()
    #         l_pix1 = F.l1_loss(preds1.to(device), self.gt)
    #         l_pix2 = F.l1_loss(preds2.to(device), self.gt)
    #         # pixel loss
    #         loss_fn = IlluminationSmoothingGradientLoss(kernel_size=3, sigma=1.0)
    #         # # 计算损失
    #         grad_loss = Gradient_Difference_Loss(alpha=1, chans=3, cuda=True)
    #         # # losssmooth = loss_fn(pred, self.gt)
    #         edgeloss1 = grad_loss(preds1, self.gt)
    #         edgeloss2 = grad_loss(preds2, self.gt)
    #         edgeloss3= grad_loss(preds2, preds1)
    #         # contrastive_loss_fn = LabContrastiveLoss(lambd=0.005, temperature=0.07)
    #         wavelet_loss = FourierIlluminationLoss()
    #         multual_loss = MultualLoss(loss_weight=0.20, reduction='mean')
    #         # lossml1 = multual_loss(preds1)
    #         # lossml2= multual_loss(preds2)
    #         lossval = F.l1_loss(preds1, preds2)
    #
    #         Lloss1 = L_spa()
    #         # ✅ 创建 L_exp 的时候就传好参数：
    #         Lloss2 = L_exp(patch_size=16, mean_val=0.6).to(device)
    #         Lloss3 = L_TV()
    #         Lloss4 = L_color()
    #         # L11 = Lloss1(img1,input1).mean()
    #         L21 = Lloss2(input1).mean()
    #         # loss_val=edgelossl(input1,input2)
    #         L22 =Lloss2(input2).mean()
    #         # loss_exp = L_exp_dual(patch_size=16)
    #         # loss_val = loss_exp(preds1, preds2)
    #         loss_function = DualBranchLossWithReference(self.gt)
    #         # 经过模型增强后的图像
    #         # lossml2 = multual_loss(preds2)
    #         multual_loss = MultualLoss(loss_weight=0.20, reduction='mean')
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #         text_encoder = TextEncoder(model)
    #         learn_prompt = Prompts([" ".join(["X"] * config.length_prompt)] * 2).cuda()
    #         learn_prompt = torch.nn.DataParallel(learn_prompt)
    #         learn_prompt.load_state_dict(
    #             torch.load("/data/gez/Project/CLIP-LIT-main/train6.12/snapshots_prompt_train0/negitive.pth",
    #                        map_location=device))
    #         learn_prompt.eval()
    #
    #         embedding_prompt = learn_prompt.module.embedding_prompt
    #         tokenized_prompts = torch.cat([clip.tokenize(" ".join(["X"] * config.length_prompt))])
    #         tokenized_prompts = tokenized_prompts.cuda()
    #
    #         text_features = text_encoder(embedding_prompt, tokenized_prompts)
    #
    #         # L_clip = L_clip_neg()
    #         L_clip = L_clip_from_feature()
    #         cliploss1 = L_clip(preds1, text_features)
    #         cliploss2 = L_clip(preds2, text_features)
    #         # ranker_loss_fn = RankerConditionalLoss(
    #         #     opt_path="/data/gez/Project/UnderwaterRanker-master/options/URanker.yaml",
    #         #     checkpoint_path="/data/gez/Project/UnderwaterRanker-master/checkpoints/URanker_ckpt.pth",
    #         # )
    #         # rankloss1 = ranker_loss_fn(preds1, self.gt)
    #         # rankloss2 = ranker_loss_fn(preds2, self.gt)
    #         # illu_pectrumloss1 = wavelet_loss(enhanced=preds1, reference=self.gt)
    #         # illu_pectrumloss2 = wavelet_loss(enhanced=preds2, reference=self.gt)
    #         # total_loss1 = 10*l_pix1+10*edgeloss1+0.03*cliploss1+L21
    #         # total_loss2 = 10*l_pix2+10*edgeloss2+0.03*cliploss2+L22
    #         total_loss1 = l_pix1 + edgeloss1 + 0.1*L21+0.05*cliploss1
    #         total_loss2 = l_pix2 + edgeloss2 + 0.1*L22+0.05*cliploss2
    #         total_loss = total_loss1 + total_loss2 + 0.1*lossval + 0.1*edgeloss3
    #         loss_dict['l_pix1'] = l_pix1.detach()
    #         loss_dict['l_pix2'] = l_pix2.detach()
    #         loss_dict['L21'] = L21.detach()
    #         # loss_dict['illu_pectrumloss1'] = illu_pectrumloss1.detach()
    #         loss_dict['clip_loss1'] = cliploss1.detach()
    #         loss_dict['clip_loss2'] = cliploss2.detach()
    #         loss_dict['edgeloss1'] = edgeloss1.detach()
    #         loss_dict['edgeloss3'] = edgeloss3.detach()
    #         loss_dict['lossval'] = lossval.detach()
    #         loss_dict['total_loss'] = total_loss.detach()
    #         total_loss.backward()
    #
    #         # 如果启用梯度裁剪
    #         if self.opt['train']['use_grad_clip']:
    #             torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
    #
    #         # 执行优化器步骤
    #         self.optimizer_g.step()
    #
    #         # 清零梯度
    #         self.optimizer_g.zero_grad()
    #
    #         # 更新 EMA（如果需要）
    #         if self.ema_decay > 0:
    #             self.model_ema(decay=self.ema_decay)
    #
    #         # 打印日志或其他操作
    #         self.log_dict = self.reduce_loss_dict(loss_dict)
    #     # 清零梯度
    #     self.optimizer_g.zero_grad()
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
    # 学生
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

    # 共享权重训练
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

    # 学生
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

    # 共享教师
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
