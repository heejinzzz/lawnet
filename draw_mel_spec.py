import os
import librosa
from matplotlib import pyplot as plt

hop_length = 512
n_fft = 2048
n_mels = 120

data_root_path = "./Datasets/CREMA-D"
audios_path = data_root_path + "/audios"
mel_images_path = data_root_path + "/mel_spectrogram"

plt.figure(figsize=[500, 500], dpi=1)
for filename in os.listdir(audios_path):
    sample_id = filename.split(".")[0]
    audio, rate = librosa.load(f"{audios_path}/{filename}", sr=16000)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec)
    plt.clf()
    librosa.display.specshow(mel_spec_db)
    plt.savefig(f"{mel_images_path}/{sample_id}.jpg", bbox_inches="tight", pad_inches=0)
