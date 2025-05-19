import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import convolve2d
import cv2
import skimage.draw
from matplotlib.cm import ScalarMappable

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

def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2)/(2*sigma**2)),
                            (kernel_size, kernel_size))
    return kernel / np.sum(kernel)

# 读取图像
image_path = 'licensed-image.jpeg'  # 替换为你的图像路径
image = cv2.imread(image_path)
image = cv2.resize(image, [1024, 1024])

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 生成kernel
angle = 150  # 角度
length = 140  # 长度
width = 30  # 宽度
img_size = (1024, 1024)  # 图像大小
# 生成图像
gpsf = create_bright_bar_image(angle, length, width, img_size)
gpsf = gpsf / np.sum(gpsf)

# gpsf = gaussian_kernel(1024, 49)
gpsf2 = gpsf / np.max(gpsf)
filtered_image_cv = cv2.filter2D(gray_image, -1, gpsf)

# 计算二维傅里叶变换
f_transform = np.fft.fft2(gray_image)
magnitude_spectrum = np.log(np.abs(np.fft.fftshift(f_transform)))
cmap = plt.get_cmap('hot')
sm = ScalarMappable(cmap=cmap)
color_magnitude_spectrum = (sm.to_rgba(magnitude_spectrum)[:, :, :3] * 255).astype(np.uint8)
colored_magnitude_spectrum = cv2.cvtColor(color_magnitude_spectrum, cv2.COLOR_RGB2BGR)

# apply filter
gpsf_transform = np.fft.fft2(np.fft.ifftshift(gpsf2))
gpsf_transform_shifted = np.fft.fftshift(gpsf_transform)

f_transform_filter = f_transform * gpsf_transform
magnitude_spectrum2 = np.log(np.abs(np.fft.fftshift(f_transform_filter)) + 1e-6)
color_magnitude_spectrum2 = (sm.to_rgba(magnitude_spectrum2)[:, :, :3] * 255).astype(np.uint8)
color_magnitude_spectrum2 = cv2.cvtColor(color_magnitude_spectrum2, cv2.COLOR_RGB2BGR)

magnitude_spectrum_psf = np.log(np.abs(gpsf_transform_shifted) + 1e-6)
color_magnitude_spectrum_psf = (sm.to_rgba(magnitude_spectrum_psf)[:, :, :3] * 255).astype(np.uint8)
color_magnitude_spectrum_psf = cv2.cvtColor(color_magnitude_spectrum_psf, cv2.COLOR_RGB2BGR)

filtered_image = np.fft.ifft2(f_transform_filter)
filtered_image = np.abs(filtered_image)
filtered_image = filtered_image / np.max(filtered_image) * 255
filtered_image = filtered_image.astype(np.uint8)

gpsf2 = (gpsf / np.max(gpsf) * 255).astype(np.uint8)

cv2.imwrite('2_gray.png',gray_image)
cv2.imwrite('2_2ddft.png',colored_magnitude_spectrum)
cv2.imwrite('2_gray_filtered_cv.png',filtered_image_cv)
cv2.imwrite('2_gray_filtered.png',filtered_image)
cv2.imwrite('2_2ddft_filtered.png',color_magnitude_spectrum2)
cv2.imwrite('2_2ddft_psf.png',color_magnitude_spectrum_psf)
cv2.imwrite('2_psf.png',gpsf2)
# cv2.imwrite('1_gray.png',gray_image)
# cv2.imwrite('1_gray.png',gray_image)