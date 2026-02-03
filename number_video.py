import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# =====================
# 参数（PPT 友好）
# =====================
start_num = 1
end_num = 50
interval = 2.7      # 每个数字持续时间（秒）
fps = 30

width, height = 640, 360   # PPT 局部动画尺寸
dpi = 100

bg_color = "white"         # "white" or "black"
text_color = "black"       # 白底黑字 / 黑底白字

font_size = 80             # 根据尺寸调

output_file = "counter_ppt.mp4"

# =====================
# Figure
# =====================
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
fig.patch.set_facecolor(bg_color)

ax = plt.axes()
ax.set_facecolor(bg_color)
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

text = ax.text(
    0.5, 0.5, "",
    fontsize=font_size,
    color=text_color,
    ha="center",
    va="center"
)

# =====================
# Writer
# =====================
writer = FFMpegWriter(
    fps=fps,
    codec="libx264",
    extra_args=["-pix_fmt", "yuv420p"]
)

frames_per_step = int(interval * fps)

# =====================
# Render
# =====================
with writer.saving(fig, output_file, dpi):
    for num in range(start_num, end_num + 1):
        text.set_text(str(num))
        for _ in range(frames_per_step):
            writer.grab_frame()

plt.close(fig)
