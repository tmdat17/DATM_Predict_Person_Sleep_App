from moviepy.editor import VideoFileClip, concatenate_videoclips

def merge_clip(video1, video2, new_name):
    clip1 = VideoFileClip(video1)
    clip2 = VideoFileClip(video2)
    final_video = concatenate_videoclips([clip1, clip2])
    final_video.write_videofile(f'{new_name}.mp4', codec='libx264', preset='slow', threads=4)

if __name__ == '__main__':
    video1 = './converted_mp4_code/at_home/sit_sleep_at_home_P1.mp4'
    video2 = './converted_mp4_code/at_home/sit_sleep_at_home_P2.mp4'
    merge_clip(video1, video2, './converted_mp4_code/at_home/full_sit_sleep_at_home')