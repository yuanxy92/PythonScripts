import json
import matplotlib.pyplot as plt
import numpy as np

# 读取JSON数据（假设文件与代码在同一目录）
with open('cfg_ev_eval.json', 'r') as f:
    data = json.load(f)

# 提取cfg_scales和对应的values数据
cfg_scales = data['cfg_scales']
violin_data = []
for scale in cfg_scales:
    # 转换为字符串作为键（因为JSON中键是字符串类型）
    scale_str = str(scale)
    violin_data.append(data['results'][scale_str]['values'])

# 创建画布
plt.figure(figsize=(12, 6))

# 绘制小提琴图（使用索引作为位置，解决非均匀问题）
# positions = range(len(cfg_scales))
positions = np.array(cfg_scales, dtype=float) * 6
parts = plt.violinplot(violin_data, positions=positions, 
                      showmeans=False, showmedians=False, showextrema=False)

colors = [
    '#E8B7B7',  # soft red
    '#BFD3E6',  # soft blue
    '#C9E2C3',  # soft green
    '#D6C8E0',  # soft purple
    '#EED6B8',  # soft orange
    '#F2F2C2',  # soft yellow
    '#DDD6C9',  # soft gray
]


# 美化小提琴图样式
idx = 0
for pc in parts['bodies']:
    pc.set_facecolor(colors[idx])
    idx = idx + 1
    pc.set_edgecolor('none')
    pc.set_alpha(0.7)

# 设置横轴刻度和标签（显示实际的cfg scale值）
plt.xticks(positions, [str(scale) for scale in cfg_scales], fontsize=10)
plt.xlabel('CFG Scale')
plt.ylabel('Exposure Value Change')

median_values = np.array([np.median(v) for v in violin_data])
plt.scatter(
    positions,
    median_values,
    s=20,
    color='dimgray',
    zorder=3
)
coef = np.polyfit(positions, median_values, deg=1)
fit_x = np.linspace(positions.min(), positions.max(), 200)
fit_y = np.polyval(coef, fit_x)
# 预测值（在原始 positions 上）
pred = np.polyval(coef, positions)
# R^2
ss_res = np.sum((median_values - pred) ** 2)
ss_tot = np.sum((median_values - np.mean(median_values)) ** 2)
r2 = 1 - ss_res / ss_tot
print(r2)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,          # 全局基准字号
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

plt.plot(
    fit_x,
    fit_y,
    color='dimgray',
    linewidth=1,
    alpha=0.9,
    zorder=2
)

# 添加网格线提升可读性
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig(
    'cfg_ev_violin.svg',
    format='svg',
    dpi=300,              # 对 SVG 不关键，但无害
    bbox_inches='tight'
)

# 显示图像
plt.show()