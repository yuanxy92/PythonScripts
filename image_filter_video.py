import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import skimage.draw
import os

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

def save_images_to_video(images_folder, video_filename, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    img_files = sorted([f"{images_folder}/{img}" for img in os.listdir(images_folder) if img.endswith('.png')])
    frame = cv2.imread(img_files[0])
    height, width, layers = frame.shape

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for img_file in img_files:
        frame = cv2.imread(img_file)
        video.write(frame)

    video.release()

# 读取图像
image_path = 'licensed-image.jpeg'  # 替换为你的图像路径
image = cv2.imread(image_path)
image = cv2.resize(image, [1024, 1024])

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


for i in range(36):
    # 自定义参数
    angle = 10 * i  # 角度
    length = 200  # 长度
    width = 20  # 宽度
    img_size = (1024, 1024)  # 图像大小
    # 生成图像
    gray_image = create_bright_bar_image(angle, length, width, img_size)
    # # gray_image = create_image_with_center_circle(length / 2, img_size)

    # 计算二维傅里叶变换
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 0.1)

    # 创建图像展示
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 显示灰度图像
    axs[0].imshow(gray_image, cmap='gray')
    # axs[0].set_title('Gray Image')
    axs[0].axis('off')

    # 显示二维傅里叶变换结果
    # 使用黄红色的 colorbar
    norm = mcolors.Normalize(vmin=np.min(magnitude_spectrum), vmax=np.max(magnitude_spectrum))
    cmap = plt.get_cmap('hot')

    img = axs[1].imshow(magnitude_spectrum, cmap=cmap, norm=norm)
    # axs[1].set_title('DFT')
    axs[1].axis('off')

    # 添加 colorbar
    cbar = fig.colorbar(img, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    # cbar.set_label('')

    # plt.show()
    plt.tight_layout()
    plt.savefig(f"temp/frame_{i:03d}.png")
    plt.close(fig)

video_filename = 'fft_animation.mp4'
save_images_to_video('temp', video_filename)