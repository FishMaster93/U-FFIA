import os
import glob


def get_audio_file(video_files):
    dir, audio_name = os.path.split(video_files)
    audio_id = (audio_name.split('.')[0])[9:]
    dir1, audio_name1 = os.path.split(dir)
    dir2, audio_name2 = os.path.split(dir1)
    dir3, audio_name3 = os.path.split(dir2)
    audio_files = glob.glob(f'/mnt/fast/nobackup/users/mc02229/Fish_av_dataset/audio_dataset/{audio_name3}/{audio_name2}/{audio_name1}/*.wav')
    for audio_file in audio_files:
        dir4, audio_name = os.path.split(audio_file)
        video_id = (audio_name.split('.')[0])[9:]
        if str(video_id) == str(audio_id):
            return os.path.join(dir4, audio_name)
