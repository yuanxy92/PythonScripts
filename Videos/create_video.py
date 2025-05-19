import os
import subprocess

def generate_video_from_images(image_folder, output_video, fps=30):
    """
    Generate a video from a folder of images using ffmpeg.

    Parameters:
    image_folder (str): Path to the folder containing images.
    output_video (str): Path to the output video file (e.g., 'output.mp4').
    fps (int): Frames per second for the video (default is 30).
    """
    # Check if the folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: The folder '{image_folder}' does not exist.")
        return
    
    # Make sure there are images in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        print("Error: No image files found in the folder.")
        return
    
    # Generate the video using ffmpeg
    input_pattern = os.path.join(image_folder, '%*.jpg')  # Change '%*.png' to match your file format
    command = [
        'ffmpeg', '-y', '-framerate', str(fps), '-i', input_pattern,
        '-c:v', 'libx264', '-r', '15', '-pix_fmt', 'yuv420p', output_video
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(command, check=True)
        print(f"Video successfully generated: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while generating the video: {e}")

# Example usage
image_folder = '/Users/yuanxy/Downloads/LocalSend/capture_server_saved_Aurora/capture_20240929_4/continuous/UVC_cam'
output_video = '/Users/yuanxy/Downloads/LocalSend/capture_server_saved_Aurora/capture_20240929_4/continuous.mp4'
generate_video_from_images(image_folder, output_video)