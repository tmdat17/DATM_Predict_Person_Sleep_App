import time
import os
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def cut_one_video(segment_duration, input_video_path, output_directory):
    """
    cut one video 
    
    Parameters:
        segment_duration(int): time(second) you want cut (ex: 10)
        input_video_path(string): name file .mp4  (required: .mp4) want cut (ex: './sample_img_pose/sleep.mp4')
        output_directory (string): folder save when cutted completed (ex: 'clip_cropped/')
    """
    # Tạo thư mục lưu các đoạn video nếu chưa tồn tại
    os.makedirs(output_directory, exist_ok=True)
    output_directory = output_directory + '/'
    # Lấy thời lượng của video gốc
    video = VideoFileClip(input_video_path)
    total_duration = video.duration
    # Bắt đầu cắt và lưu các đoạn video
    start_time = 0
    segment_number = 1
    print('Prepare Cropped:  ')
    cur = time.time()
    while start_time < total_duration:
        cur_segment = time.time()
        end_time = min(start_time + segment_duration, total_duration)
        output_video_name = f'clip_{segment_number}.mp4'
        ffmpeg_extract_subclip(input_video_path, start_time, end_time, targetname=os.path.join(output_directory, output_video_name))
        print(f'Video {segment_number} đã được cắt và lưu tại: {os.path.join(output_directory, output_video_name)}')
        start_time = end_time
        segment_number += 1
        print('End Time For segment {}:   {}'.format(segment_number-1, time.time() - cur_segment))
    print('Full End Time: ', time.time() - cur)

if __name__ == '__main__':
    # Độ dài mỗi đoạn video cắt (đơn vị: giây)
    segment_duration = 10
    # Đường dẫn đến file video gốc
    input_video_path = './converted_mp4_code/at_home/full_lie_wake_at_home.mp4'
    # Đường dẫn đến thư mục lưu các đoạn video đã cắt
    
    folder_ori = './converted_mp4_code/at_home/'
    for filename in os.listdir(folder_ori):
        output_directory = 'clip_cropped'
        # make sure it is file not folder
        if os.path.isfile(os.path.join(folder_ori, filename)):
            output_directory = output_directory + '/' + filename.split('.')[0] + '/' + str(segment_duration) + 's'
            os.makedirs(output_directory, exist_ok=True)
            filename = folder_ori + filename
            cut_one_video(segment_duration, filename, output_directory)
    