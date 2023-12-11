import time
from moviepy.editor import VideoFileClip
import os

def convert_all_file_from_folder(folder_ori, directory_path):
    """
    convert all file in folder from MOV to MP4
    Parameters:
        folder_ori (string): folder contain file .MOV (ex: './Data_Train_Pose/')
        directory_path (string): folder save when convert completed (ex: 'converted_mp4_code')
    """
    # Kiểm tra xem thư mục đã tồn tại chưa
    if not os.path.exists(directory_path):
        # Nếu thư mục chưa tồn tại, tạo thư mục mới
        os.makedirs(directory_path)
        print(f'Folder "{directory_path}" is created')
    else:
        print(f'Folder "{directory_path}" already existed')

    # resolution of iPhone 11
    resolution = (828, 1792)
    # frame rate expect (FPS)
    frame_rate = 30
    cur = time.time()
    for filename in os.listdir(folder_ori):
        # make sure it is file not folder
        if os.path.isfile(os.path.join(folder_ori, filename)):
            new_name = filename.split('.')[0]
            print(f'new name {new_name}')
            cur_file = time.time()
            input_file = folder_ori + '/{}'.format(filename)
            # implement here
            print(f'Prepare handle file: {filename}')
            output_file = directory_path +  '/{}.mp4'.format(new_name)
            video = VideoFileClip(input_file)
            video = video.resize(resolution)  # Cài đặt độ phân giải
            video = video.set_duration(video.duration)  # Đảm bảo tỷ lệ khung hình không thay đổi
            video = video.set_fps(frame_rate)  # Cài đặt tỷ lệ khung hình
            video.write_videofile(output_file, codec='libx264', preset='slow', threads=4)
            print(f'Convert Success: {input_file} -> {output_file}')
            print('FULL TIME FOR FILE {}:   {}'.format(filename, time.time() - cur_file))
            print('=========================================== END ==========================================')
    print('FULL END TIME:   {}'.format(time.time() - cur))


def convert_one_file_from_folder(input_file, directory_path):
    """
    convert one file from folder
    Parameters:
        input_file(string): name file .mov need convert (ex: 'video_input.MOV')
        directory_path (string): folder save when convert completed (ex: 'converted_mp4_code')
    """
    if not os.path.exists(directory_path):
        # Nếu thư mục chưa tồn tại, tạo thư mục mới
        os.makedirs(directory_path)
        print(f'Folder "{directory_path}" is created')
    else:
        print(f'Folder "{directory_path}" already existed')
    new_name = input_file.split('.')[0]
    output_file = '{}.mp4'.format(new_name)
    # resolution of iPhone 11
    resolution = (828, 1792)
    # frame rate expect (FPS)
    frame_rate = 30
    print(f'Prepare handle file: {input_file}')
    cur_file = time.time()
    # Sử dụng moviepy để chuyển đổi định dạng video và cài đặt độ phân giải và tỷ lệ khung hình
    video = VideoFileClip(input_file)
    video = video.resize(resolution)  # Cài đặt độ phân giải
    video = video.set_duration(video.duration)  # Đảm bảo tỷ lệ khung hình không thay đổi
    video = video.set_fps(frame_rate)  # Cài đặt tỷ lệ khung hình
    video.write_videofile(output_file, codec='libx264', preset='slow', threads=4)
    print(f'Convert Success: {input_file} -> {output_file}')
    print('FULL TIME FOR FILE {}:   {}'.format(input_file, time.time() - cur_file))


if __name__ == '__main__':
    folder_ori = './Data_Train_Pose/at_home'
    directory_path = 'converted_mp4_code/at_home'
    convert_all_file_from_folder(folder_ori, directory_path)