import json
import random
import torch
import torch.utils.data.dataset
from transformers import AutoFeatureExtractor
import librosa
from PIL import Image
import cv2


def do_encode_audio(audio_file, audio_processor, encode_audio_length, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    encode_audio = audio_processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values
    encode_audio = torch.squeeze(encode_audio, 0)
    if encode_audio_length > len(encode_audio):
        audio_padding = torch.full([encode_audio_length-len(encode_audio)], fill_value=audio_processor.padding_value)
        encode_audio = torch.hstack([encode_audio, audio_padding])
    encode_audio = encode_audio[:encode_audio_length]
    return encode_audio


def do_encode_image(image_file, image_processor):
    image = Image.open(image_file)
    encode_image = image_processor(image, return_tensors="pt").pixel_values
    encode_image = torch.squeeze(encode_image, 0)
    return encode_image


def do_encode_video(video_file, image_processor, sample_frame_num, mode):
    assert mode in ["random", "uniform"]
    video = cv2.VideoCapture(video_file)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num <= 0:
        return torch.zeros([sample_frame_num, 3, image_processor.size["height"], image_processor.size["width"]])
    if frame_num < sample_frame_num:
        sample_frame_ids = list(range(sample_frame_num))
        for i in range(len(sample_frame_ids)):
            sample_frame_ids[i] = min(sample_frame_ids[i], frame_num - 1)
    elif mode == "random":
        sample_frame_ids = random.sample(range(frame_num), sample_frame_num)
        list.sort(sample_frame_ids)
    else:
        interval = frame_num // sample_frame_num
        sample_frame_ids = list(range(0, frame_num, interval))
    encode_frames = []
    while video.isOpened():
        if len(encode_frames) == sample_frame_num:
            break
        frame_id = sample_frame_ids[len(encode_frames)]
        video.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id))
        ok, frame = video.read()
        if not ok:
            if len(encode_frames) > 0:
                encode_frames.append(torch.clone(encode_frames[-1]))
            else:
                encode_frames.append(torch.zeros([3, image_processor.size["height"], image_processor.size["width"]]))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame, mode="RGB")
        encode_frame = image_processor(image, return_tensors="pt").pixel_values
        encode_frame = torch.squeeze(encode_frame, 0)
        encode_frames.append(encode_frame)
    video.release()
    encode_frames = torch.stack(encode_frames)
    return encode_frames


class RAVDESS_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split, encode_audio_length, sample_frame_num, data_root_path="./Datasets/RAVDESS/speech"):
        assert split in ["train", "val", "test"]
        split_data = json.load(open(data_root_path + "/split.json"))
        sample_ids = split_data[split]

        self.sample_frame_num = sample_frame_num
        self.sample_mode = "random" if split == "train" else "uniform"

        self.audios_path = data_root_path + "/audios"
        self.mel_images_path = data_root_path + "/mel_spectrogram"
        self.videos_path = data_root_path + "/videos"

        wav2vec_path = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        swin_path = "microsoft/swin-base-patch4-window7-224-in22k"

        self.encode_audio_length = encode_audio_length
        self.audio_processor = AutoFeatureExtractor.from_pretrained(wav2vec_path, local_files_only=False)
        self.image_processor = AutoFeatureExtractor.from_pretrained(swin_path, local_files_only=False)

        self.encode_audios = []
        self.encode_mel_images = []
        self.video_files = []
        self.emotion_ids = []

        for sample_id in sample_ids:
            emotion_id = int(sample_id.split("-")[1]) - 1
            encode_audio = do_encode_audio(f"{self.audios_path}/03-{sample_id}.wav", self.audio_processor, self.encode_audio_length)
            encode_mel_image = do_encode_image(f"{self.mel_images_path}/03-{sample_id}.jpg", self.image_processor)
            video_file = f"{self.videos_path}/02-{sample_id}.mp4"

            self.encode_audios.append(encode_audio)
            self.encode_mel_images.append(encode_mel_image)
            self.video_files.append(video_file)
            self.emotion_ids.append(emotion_id)

    def __len__(self):
        return len(self.emotion_ids)

    def __getitem__(self, index):
        encode_audio, encode_mel_image, emotion_id = self.encode_audios[index], self.encode_mel_images[index], self.emotion_ids[index]
        video_file = self.video_files[index]
        encode_frames = do_encode_video(video_file, self.image_processor, self.sample_frame_num, self.sample_mode)
        return encode_audio, encode_mel_image, encode_frames, emotion_id


class CREMA_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split, encode_audio_length, sample_frame_num, data_root_path="./Datasets/CREMA-D"):
        self.emotion2id = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
        assert split in ["train", "val", "test"]
        split_data = json.load(open(data_root_path + "/split.json"))
        sample_ids = split_data[split]

        self.sample_frame_num = sample_frame_num
        self.sample_mode = "random" if split == "train" else "uniform"

        self.audios_path = data_root_path + "/audios"
        self.mel_images_path = data_root_path + "/mel_spectrogram"
        self.videos_path = data_root_path + "/videos"

        wav2vec_path = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        swin_path = "microsoft/swin-base-patch4-window7-224-in22k"

        self.encode_audio_length = encode_audio_length
        self.audio_processor = AutoFeatureExtractor.from_pretrained(wav2vec_path, local_files_only=False)
        self.image_processor = AutoFeatureExtractor.from_pretrained(swin_path, local_files_only=False)

        self.encode_audios = []
        self.encode_mel_images = []
        self.video_files = []
        self.emotion_ids = []

        for sample_id in sample_ids:
            emotion_id = int(self.emotion2id[sample_id.split("_")[2]])
            encode_audio = do_encode_audio(f"{self.audios_path}/{sample_id}.wav", self.audio_processor, self.encode_audio_length)
            encode_mel_image = do_encode_image(f"{self.mel_images_path}/{sample_id}.jpg", self.image_processor)
            video_file = f"{self.videos_path}/{sample_id}.flv"

            self.encode_audios.append(encode_audio)
            self.encode_mel_images.append(encode_mel_image)
            self.video_files.append(video_file)
            self.emotion_ids.append(emotion_id)

    def __len__(self):
        return len(self.emotion_ids)

    def __getitem__(self, index):
        encode_audio, encode_mel_image, emotion_id = self.encode_audios[index], self.encode_mel_images[index], self.emotion_ids[index]
        video_file = self.video_files[index]
        encode_frames = do_encode_video(video_file, self.image_processor, self.sample_frame_num, self.sample_mode)
        return encode_audio, encode_mel_image, encode_frames, emotion_id
