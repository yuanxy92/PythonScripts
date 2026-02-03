import numpy as np
import matplotlib.pyplot as plt

# 定义 soft clipping 函数
def soft_clipping(x, alpha=1.0):
    return x / (1 + alpha * np.abs(x))

# 创建输入数据
x = np.linspace(-10, 10, 1000)

# 计算不同 alpha 值下的函数值
y1 = soft_clipping(x, alpha=1.0)
y2 = soft_clipping(x, alpha=1.5)
y3 = soft_clipping(x, alpha=2.0)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='alpha = 0.5', color='blue')
plt.plot(x, y2, label='alpha = 1.0', color='green')
plt.plot(x, y3, label='alpha = 2.0', color='red')
plt.axhline(1, color='gray', linestyle='--', linewidth=0.5)
plt.axhline(-1, color='gray', linestyle='--', linewidth=0.5)
plt.title('Soft Clipping Function')
plt.xlabel('x')
plt.ylabel('f(x) = x / (1 + alpha * |x|)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
