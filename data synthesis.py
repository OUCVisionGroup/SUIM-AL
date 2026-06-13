import cv2
import numpy as np
import os
import random

# 输入和输出路径
image_dir = ""#清晰图
depth_dir = ""#深度图
light_source_dir = ""#随机光源图
output_dir =  ""#输出位置
depth_info_file = ""#深度范围文件

# 读取深度范围信息
depth_ranges = {}
with open(depth_info_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            filename, min_depth, max_depth = parts
            depth_ranges[filename] = (float(min_depth), float(max_depth))

# 获取所有光源图
light_source_files = [os.path.join(light_source_dir, f) for f in os.listdir(light_source_dir) if f.endswith(".png")]

# 定义相关参数
def calculate_Br(Bg, br, bg, beta_g, beta_r):
    return (beta_g * br) / (beta_r * bg) * Bg

def calculate_Bb(Bg, bb, bg, beta_g, beta_b):
    return (beta_g * bb) / (beta_b * bg) * Bg

def smooth_depth_map(depth_map, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(depth_map, kernel_size, sigma)

#自定义水体类型
# jelov1
# beta_b_r, beta_b_g, beta_b_b = 0.000899, 0.00205, 0.00381
# alpha_r, alpha_g, alpha_b = 0.334, 0.046, 0.018
# beta_d_r, beta_d_g, beta_d_b = 0.341, 0.049, 0.022
# jelov1A
# beta_b_r, beta_b_g, beta_b_b = 0.00234, 0.00402, 0.00631
# alpha_r, alpha_g, alpha_b = 0.334, 0.0468, 0.0221
# beta_d_r, beta_d_g, beta_d_b = 0.342, 0.0503, 0.0264
# jelov1B
# beta_b_r, beta_b_g, beta_b_b = 0.0450, 0.0565, 0.0680
# alpha_r, alpha_g, alpha_b = 0.334, 0.0469, 0.0235
# beta_d_r, beta_d_g, beta_d_b = 0.349, 0.0572, 0.0342
# jelov1C
# beta_b_r, beta_b_g, beta_b_b = 0.274, 0.395, 0.514
# alpha_r, alpha_g, alpha_b = 0.344, 0.068, 0.105
# beta_d_r, beta_d_g, beta_d_b = 0.439, 0.121, 0.179
# jelov2
# beta_b_r, beta_b_g, beta_b_b = 0.270, 0.387, 0.504
# alpha_r, alpha_g, alpha_b = 0.334, 0.0469, 0.0241
# beta_d_r, beta_d_g, beta_d_b = 0.375, 0.0845, 0.0620
# jelov3
# beta_b_r, beta_b_g, beta_b_b = 0.737, 1.06, 1.38
# alpha_r, alpha_g, alpha_b = 0.336, 0.0507, 0.0388
# beta_d_r, beta_d_g, beta_d_b = 0.426, 0.129, 0.124
# jelov3C
# beta_b_r, beta_b_g, beta_b_b = 0.800, 1.15, 1.5
# alpha_r, alpha_g, alpha_b = 0.346, 0.078, 0.154
# beta_d_r, beta_d_g, beta_d_b = 0.498, 0.187, 0.319
# jelov5C
# beta_b_r, beta_b_g, beta_b_b = 1.23, 1.44, 1.87
# alpha_r, alpha_g, alpha_b = 0.119, 0.127, 0.297
# beta_d_r, beta_d_g, beta_d_b = 0.252, 0.277, 0.535
# jelov7C
# beta_b_r, beta_b_g, beta_b_b = 1.77, 2.54, 3.3
# alpha_r, alpha_g, alpha_b = 0.403, 0.233, 0.542
# beta_d_r, beta_d_g, beta_d_b = 0.635, 0.470, 0.924
# jelov9C
beta_b_r, beta_b_g, beta_b_b = 2.35, 3.38, 4.39
alpha_r, alpha_g, alpha_b = 0.456, 0.430, 0.943
beta_d_r, beta_d_g, beta_d_b = 0.775, 0.826, 1.56

D = 50
E_B = 0.1

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 批量处理
for image_file in os.listdir(image_dir):
    if not image_file.endswith(".png"):
        continue

    image_path = os.path.join(image_dir, image_file)
    depth_path = os.path.join(depth_dir, image_file.replace(".png", ".png"))

    # 随机选择一个光源图
    light_source_file = random.choice(light_source_files)

    # 读取图像和光源图
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    light_source = cv2.imread(light_source_file).astype(np.float32) / 255.0

    # 调整光源图大小
    light_source_resized = cv2.resize(light_source, (image.shape[1], image.shape[0]))

    # 处理光源图
    light_source_blurred = cv2.GaussianBlur(light_source_resized, (21, 21), sigmaX=10, sigmaY=10)
    gray_light_source = cv2.cvtColor((light_source_blurred * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_light_source, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    light_source_no_halo = cv2.inpaint((light_source_blurred * 255).astype(np.uint8), mask, inpaintRadius=7,
                                       flags=cv2.INPAINT_TELEA)
    light_source_no_halo = light_source_no_halo.astype(np.float32) / 255.0
    light_source_no_halo_resized = cv2.resize(light_source_no_halo, (image.shape[1], image.shape[0]))

    imageL = image * light_source_no_halo_resized
    # 读取深度图
    relative_depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    relative_depth = smooth_depth_map(relative_depth)

    min_depth, max_depth = depth_ranges.get(image_file, (0.35, 0.77))

    relative_depth = 1 - relative_depth
    depth = relative_depth * (max_depth - min_depth) + min_depth

    # **随机化 E_A 和 Bg**
    E_A = random.uniform(1, 2)
    Bg = random.uniform(0.4, 0.6)

    # 重新计算 Br 和 Bb
    Br = calculate_Br(Bg, beta_d_r, beta_d_g, beta_b_r, beta_b_g)
    Bb = calculate_Bb(Bg, beta_d_b, beta_d_g, beta_b_g, beta_b_b)
    # 重新计算 Br 和 Bb

    # 计算图像通道
    image[..., 0] = ((image[..., 0] * np.exp(-alpha_b * D) +
                      E_A * imageL[..., 0] *  np.exp(-beta_d_b * depth)) * np.exp(-beta_d_b * depth) +
                     E_B * Bb * (1 - np.exp(-beta_b_b * depth)))

    image[..., 1] = ((image[..., 1] * np.exp(-alpha_g * D) +
                      E_A * imageL[..., 1]* np.exp(-beta_d_g * depth)) * np.exp(-beta_d_g * depth) +
                     E_B * Bg * (1 - np.exp(-beta_b_g * depth)))

    image[..., 2] = ((image[..., 2] * np.exp(-alpha_r * D) +
                      E_A * imageL[..., 2]* np.exp(-beta_d_r * depth)) * np.exp(-beta_d_r * depth) +
                     E_B * Br * (1 - np.exp(-beta_b_r * depth)))

    # 转换回0-255范围并保存
    output_image = np.clip(image * 255, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.png")
    cv2.imwrite(output_path, output_image)

    print(f"处理完成: {output_path}")