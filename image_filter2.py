import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# 读取图像
image_path = 'licensed-image.jpeg'  # 替换为你的图像路径
image = cv2.imread(image_path)
image = cv2.resize(image, [1024, 1024])

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算二维傅里叶变换
f_transform = np.fft.fft2(gray_image)
f_transform_shifted = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_transform_shifted))
cmap = plt.get_cmap('hot')
# 创建一个ScalarMappable对象
sm = ScalarMappable(cmap=cmap)
# 将灰度图映射到颜色映射上
color_magnitude_spectrum = (sm.to_rgba(magnitude_spectrum)[:, :, :3] * 255).astype(np.uint8)
colored_magnitude_spectrum = cv2.cvtColor(color_magnitude_spectrum, cv2.COLOR_RGB2BGR)

# apply filter
mask = create_image_with_center_circle(30, image_size=(1024, 1024))
f_transform_shifted_f = f_transform_shifted * (1 - mask)
magnitude_spectrum2 = np.log(np.abs(f_transform_shifted_f) + 1e-6)
color_magnitude_spectrum2 = (sm.to_rgba(magnitude_spectrum2)[:, :, :3] * 255).astype(np.uint8)
color_magnitude_spectrum2 = cv2.cvtColor(color_magnitude_spectrum2, cv2.COLOR_RGB2BGR)

filtered_image = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted_f))
filtered_image = np.abs(filtered_image)
filtered_image = filtered_image / np.max(filtered_image) * 255
filtered_image = filtered_image.astype(np.uint8)

cv2.imwrite('1_gray.png',gray_image)
cv2.imwrite('1_2ddft.png',colored_magnitude_spectrum)
cv2.imwrite('1_gray_filtered.png',filtered_image)
cv2.imwrite('1_2ddft_filtered.png',color_magnitude_spectrum2)
# cv2.imwrite('1_gray.png',gray_image)
# cv2.imwrite('1_gray.png',gray_image)