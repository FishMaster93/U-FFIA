import os
import glob


def get_video_file(audio_files):
    dir, audio_name = os.path.split(audio_files)
    audio_id =(audio_name.split('.')[0])[9:]
    dir1, audio_name1 = os.path.split(dir)
    dir2, audio_name2 = os.path.split(dir1)
    dir3, audio_name3 = os.path.split(dir2)
    video_files = glob.glob(f'/vol/research/Fish_tracking_master/Fish_av_dataset/video_dataset/{audio_name3}/{audio_name2}/{audio_name1}/*.mp4')
    video_file = get_video_index(video_files, audio_id)
    return video_file

def get_video_index(video_files, audio_id):
    for video_file in video_files:
        dir4, video_name = os.path.split(video_file)
        video_id = (video_name.split('.')[0])[9:]
        if str(video_id) == str(audio_id):
            return os.path.join(dir4, video_name)


