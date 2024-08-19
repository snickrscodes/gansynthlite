import tensorflow as tf
import numpy as np
import os

path = 'C:/Users/saraa/Desktop/music_generation/train_music/'
out_path = 'C:/Users/saraa/Desktop/music_generation/generated_music/'

def get_data(index: int):
    waveform = tf.io.read_file(path+str(index)+'.wav')
    waveform, _ = tf.audio.decode_wav(contents=waveform, desired_channels=1, desired_samples=64000)
    return waveform.numpy()
# cutoff is 41 (F2) to 89 (F6)
def get_dataset() -> np.ndarray:
    files = sorted(os.listdir(path), key=lambda x: int(x[:-4]))
    num_samples = len(files)
    data = np.zeros(shape=(num_samples,64000,1), dtype=np.float32)
    for i in range(num_samples):
        data[i] = get_data(i)
    print(f'\033[93mmemory used by training data: {data.nbytes / (1024 ** 3)} GB\033[00m')
    return data

get_dataset()