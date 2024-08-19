import os
import math
from pydub import AudioSegment
# you can use this to format your music files into a suitable form for the format
# these have already been processed, so don't run this script

path = 'C:/Users/saraa/Desktop/music_generation/train_music/'
segment_length = 4 # 4 seconds

def make_segments(index: int):
    name = path+str(index)
    audio = AudioSegment.from_wav(name+'.wav')
    total_time = math.floor(audio.duration_seconds)
    for i in range(0, total_time, segment_length):
        clip = audio[i*1000:(i+segment_length)*1000]
        clip.export(name+'x'+str(i+1)+'.wav', format='wav')
    os.remove(name+'.wav')

def rename_files(files: list[str]):
    for index, name in enumerate(files, start=0):
        old_file_path = os.path.join(path, name)
        new_file_path = os.path.join(path, f'{index}.wav')
        os.rename(old_file_path, new_file_path)

def process_data():
    for i in range(1, 165):
        make_segments(i)
    files = os.listdir(path)
    rename_files(files)

process_data()