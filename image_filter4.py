import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import skimage.draw

def create_image_with_center_circle(radius, image_size=(256, 256)):
    # 创建空白图像
    image = np.zeros(image_size)
    # 计算图像中心
    center_y, center_x = image_size[0] // 2, image_size[1] // 2
    # 使用 skimage.draw.disk 创建圆
    rr, cc = skimage.draw.disk((center_y, center_x), radius, shape=image_size)
    image[rr, cc] = 1

    return image

def create_bright_bar_image(angle, length, width, image_size=(256, 256)):
    # 创建空白图像
    image = np.zeros(image_size)
    # 计算图像中心
    center_y, center_x = image_size[0] // 2, image_size[1] // 2
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)
    # 计算条的半长和半宽
    half_length = length // 2
    half_width = width // 2
    # 计算条的四个角点
    x1, y1 = center_x - half_length * np.cos(angle_rad) + half_width * np.sin(angle_rad), center_y - half_length * np.sin(angle_rad) - half_width * np.cos(angle_rad)
    x2, y2 = center_x + half_length * np.cos(angle_rad) + half_width * np.sin(angle_rad), center_y + half_length * np.sin(angle_rad) - half_width * np.cos(angle_rad)
    x3, y3 = center_x + half_length * np.cos(angle_rad) - half_width * np.sin(angle_rad), center_y + half_length * np.sin(angle_rad) + half_width * np.cos(angle_rad)
    x4, y4 = center_x - half_length * np.cos(angle_rad) - half_width * np.sin(angle_rad), center_y - half_length * np.sin(angle_rad) + half_width * np.cos(angle_rad)
    # 使用多边形填充函数绘制条
    rr, cc = skimage.draw.polygon([y1, y2, y3, y4], [x1, x2, x3, x4], image_size)
    image[rr, cc] = 1

    return image

# 读取图像
image_path = 'shanghai2.png'  # 替换为你的图像路径
# image_path = 'The-original-cameraman-image.png'  # 替换为你的图像路径
image = cv2.imread(image_path)
image = cv2.resize(image, [1024, 1024])

image_path2 = 'shanghai.png'  # 替换为你的图像路径
# image_path = 'The-original-cameraman-image.png'  # 替换为你的图像路径
image2 = cv2.imread(image_path2)
image2 = cv2.resize(image2, [1024, 1024])

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# noise = np.random.normal(0, 10, [350, 350]).astype(np.uint8)
# noise = np.random.randint(0, 256, (400, 400), dtype=np.uint8)
# gray_image = noise

# 计算二维傅里叶变换
f_transform = np.fft.fft2(gray_image)
f_transform2 = np.fft.fft2(gray_image2)

# 创建图像展示
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 显示灰度图像
axs[0].imshow(gray_image, cmap='gray')
# axs[0].set_title('Gray Image')
axs[0].axis('off')

amp = np.abs(f_transform)
pha = np.angle(f_transform)
amp2 = np.abs(f_transform2)
image2_f = amp2 * np.cos(pha) + amp2 * np.sin(pha) * 1j
# image2 = np.fft.ifft2(image2_f * 1250).astype(np.uint8)
image2 = np.fft.ifft2(image2_f).astype(np.uint8)
img = axs[1].imshow(image2, cmap='gray')
axs[1].axis('off')

image3_f = amp
image3 = np.fft.ifft2(image3_f).astype(np.uint8)
img = axs[2].imshow(image3, cmap='gray')
axs[2].axis('off')

# 添加 colorbar
# cbar = fig.colorbar(img, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
# cbar.set_label('')

# plt.show()
plt.tight_layout()
plt.savefig(f"temp.png")
plt.close(fig)