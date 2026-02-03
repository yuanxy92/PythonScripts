import cv2
import numpy as np
import os

def compute_weight(image):
    """
    计算单张图像的权重图
    权重由三部分组成：对比度、饱和度、曝光度，加权求和
    """
    # 转换为浮点数格式，方便计算
    img = image.astype(np.float32) / 255.0
    
    # 1. 计算对比度（使用拉普拉斯算子的绝对值）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    contrast = np.abs(laplacian)
    
    # 2. 计算饱和度（各通道标准差）
    saturation = np.std(img, axis=2)
    
    # 3. 计算曝光度（高斯函数，理想曝光值为0.5）
    exposure = np.exp(-((img - 0.5) ** 2) / (2 * 0.2 ** 2))
    exposure = np.mean(exposure, axis=2)  # 三通道平均
    
    # 权重融合（可调参数）
    weight = (contrast * 0.2 + saturation * 0.3 + exposure * 0.5)
    
    # 防止权重为0
    weight = np.maximum(weight, 1e-8)
    
    return weight

def exposure_fusion(images):
    """
    曝光融合主函数
    参数:
        images: 不同曝光的图像列表（BGR格式）
    返回:
        fused_image: 融合后的图像
    """
    # 获取图像尺寸
    h, w = images[0].shape[:2]
    num_images = len(images)
    
    # 初始化权重图列表
    weights = []
    
    # 计算每张图像的权重图
    for img in images:
        weight = compute_weight(img)
        # 高斯滤波平滑权重图，避免融合后出现拼接痕迹
        weight = cv2.GaussianBlur(weight, (5, 5), 0)
        weights.append(weight)
    
    # 权重归一化：每个像素的权重总和为1
    weight_sum = np.sum(weights, axis=0)
    normalized_weights = [w / weight_sum for w in weights]
    
    # 初始化融合图像
    fused = np.zeros_like(images[0], dtype=np.float32)
    
    # 按权重融合图像
    for i in range(num_images):
        # 将权重图扩展到三通道
        weight_3d = np.repeat(normalized_weights[i][:, :, np.newaxis], 3, axis=2)
        # 加权求和
        fused += images[i].astype(np.float32) * weight_3d
    
    # 转换为8位图像并裁剪到有效范围
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    
    return fused

def load_exposure_images(folder_path):
    """
    从文件夹加载不同曝光的图像
    要求图像按曝光度命名，如: exp_0.jpg, exp_1.jpg, exp_2.jpg
    """
    image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = [cv2.imread(path) for path in image_paths]
    
    # 检查图像是否加载成功
    for i, img in enumerate(images):
        if img is None:
            raise ValueError(f"无法加载图像: {image_paths[i]}")
    
    # 确保所有图像尺寸相同
    base_shape = images[0].shape
    for img in images[1:]:
        if img.shape != base_shape:
            raise ValueError("所有图像必须具有相同的尺寸")
    
    return images

if __name__ == "__main__":
    # ====================== 配置参数 ======================
    # 替换为你的图像文件夹路径（存放不同曝光的图像）
    input_folder = "./images/2015_01082"
    # 融合后图像的保存路径
    output_path = "./images/2015_01082.png"
    
    # ====================== 执行融合 ======================
    try:
        # 加载图像
        print("正在加载曝光图像...")
        exposure_images = load_exposure_images(input_folder)
        print(f"成功加载 {len(exposure_images)} 张曝光图像")
        
        # 执行曝光融合
        print("正在执行曝光融合...")
        fused_img = exposure_fusion(exposure_images)
        
        # 保存结果
        cv2.imwrite(output_path, fused_img)
        print(f"融合完成！结果已保存至: {output_path}")
        
        # 显示结果（可选）
        cv2.imshow("Fused Image", fused_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"出错了: {e}")