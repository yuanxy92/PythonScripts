from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, concatenate_audioclips
from pathlib import Path

image_dir='/Users/yuanxy/Backup/sq_cgx/shuqi/4K修复写真--古装宫廷'
output_path="/Users/yuanxy/Backup/sq_cgx/shuqi/4K修复写真--古装宫廷.mp4"
music_paths=["/Users/yuanxy/Music/QQ音乐/舒淇-天下第一奇书.ogg", "/Users/yuanxy/Music/QQ音乐/舒淇-解放.ogg", "/Users/yuanxy/Music/QQ音乐/舒淇-迷惑.ogg",
            "/Users/yuanxy/Music/QQ音乐/舒淇-撕杀.ogg", "/Users/yuanxy/Music/QQ音乐/舒淇-出浴.ogg", "/Users/yuanxy/Music/QQ音乐/舒淇-新欢.ogg", "/Users/yuanxy/Music/QQ音乐/舒淇-墓前-结尾.ogg"]  # 可以混合不同格式

# 配置
image_folder = image_dir   # 图片文件夹
output_video = output_path # 输出视频
fps = 30                          # 视频帧率
duration_per_image = 3            # 每张图停留秒数
video_size = (3840, 2160)         # 4K分辨率
music_list = music_paths # 背景音乐列表

# 获取图片列表（按文件名排序）
image_paths = sorted(Path(image_folder).glob("*.*"))

clips = []
for img_path in image_paths:
    clip = ImageClip(str(img_path)).set_duration(duration_per_image)
    
    # 保持比例缩放到 4K 范围内
    clip = clip.resize(height=video_size[1]) if clip.w/clip.h < video_size[0]/video_size[1] else clip.resize(width=video_size[0])
    
    # 居中填充黑边
    clip = clip.on_color(size=video_size, color=(0,0,0), pos=("center","center"))
    
    clips.append(clip)

# 拼接视频
final_clip = concatenate_videoclips(clips, method="compose").set_fps(fps)

# 添加背景音乐（多个文件顺序播放或循环）
if music_list:
    audio_clips = [AudioFileClip(m).subclip(0) for m in music_list]
    full_audio = concatenate_audioclips(audio_clips)
    
    # 如果音乐比视频短，循环填充
    if full_audio.duration < final_clip.duration:
        n_loops = int(final_clip.duration // full_audio.duration) + 1
        full_audio = concatenate_audioclips(audio_clips * n_loops)
    
    full_audio = full_audio.set_duration(final_clip.duration)
    final_clip = final_clip.set_audio(full_audio)

# 输出视频
final_clip.write_videofile(output_video, codec="h264", audio_codec="aac")