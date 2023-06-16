import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, SwinModel


class FusionModule(nn.Module):
    def __init__(self, d_model):
        super(FusionModule, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=16, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(p=0.1),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        y = self.cross_attention(x, memory, memory)[0]
        y = self.dropout(y)
        y = self.norm(x + y)
        y = self.norm2(y + self.ffn(y))
        return y


class LaWNet(nn.Module):
    def __init__(self, num_classes, sample_frame_num):
        super(LaWNet, self).__init__()
        wav2vec_path = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        swin_path = "microsoft/swin-base-patch4-window7-224-in22k"

        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_path, local_files_only=False)
        self.mel_swin = SwinModel.from_pretrained(swin_path, local_files_only=False)
        self.video_swin = SwinModel.from_pretrained(swin_path, local_files_only=False)

        self.embedding_size = self.wav2vec.config.hidden_size

        self.audio_mel_position_embedding = nn.Embedding(num_embeddings=249+49, embedding_dim=self.embedding_size)
        self.audio_mel_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=16, batch_first=True)
        self.audio_mel_encoder = nn.TransformerEncoder(self.audio_mel_encoder_layer, num_layers=2)

        self.sample_frame_num = sample_frame_num
        self.frames_position_embedding = nn.Embedding(num_embeddings=sample_frame_num, embedding_dim=self.embedding_size)
        self.frames_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=16, batch_first=True)
        self.frames_encoder = nn.TransformerEncoder(self.frames_encoder_layer, num_layers=2)

        self.audio_to_visual_fusion = FusionModule(self.embedding_size)
        self.visual_to_audio_fusion = FusionModule(self.embedding_size)

        self.classifier = nn.Linear(self.embedding_size, num_classes)

    def forward(self, x_audios, x_mel_images, x_frames):
        # audio
        encode_audios = self.wav2vec(x_audios).last_hidden_state
        encode_mel_images = self.mel_swin(x_mel_images).last_hidden_state
        encode_audio_mel = torch.hstack([encode_audios, encode_mel_images])
        audio_mel_position_ids = torch.arange(encode_audio_mel.size(1), dtype=torch.long, device=encode_audio_mel.device)
        audio_mel_position_ids = audio_mel_position_ids.unsqueeze(0).expand([encode_audio_mel.size(0), encode_audio_mel.size(1)])
        audio_mel_position_embed = self.audio_mel_position_embedding(audio_mel_position_ids)
        encode_audio_mel = encode_audio_mel + audio_mel_position_embed
        encode_audio_mel = self.audio_mel_encoder(encode_audio_mel)

        # visual
        batch_size = len(x_frames)
        x_frames = torch.reshape(x_frames, [batch_size * self.sample_frame_num, 3, self.video_swin.config.image_size, self.video_swin.config.image_size])
        encode_frames = self.video_swin(x_frames).last_hidden_state
        encode_frames = torch.reshape(encode_frames, [batch_size, self.sample_frame_num, -1, self.embedding_size])
        encode_frames = torch.mean(encode_frames, dim=2)
        frames_position_ids = torch.arange(self.sample_frame_num, dtype=torch.long, device=x_frames.device)
        frames_position_ids = frames_position_ids.unsqueeze(0).expand([batch_size, self.sample_frame_num])
        frames_position_embed = self.frames_position_embedding(frames_position_ids)
        encode_frames = encode_frames + frames_position_embed
        encode_frames = self.frames_encoder(encode_frames)

        # fusion
        fused_audio_mel = self.visual_to_audio_fusion(encode_audio_mel, encode_frames)
        fused_audio_mel = torch.mean(fused_audio_mel, dim=1)
        fused_frames = self.audio_to_visual_fusion(encode_frames, encode_audio_mel)
        fused_frames = torch.mean(fused_frames, dim=1)
        encode_av = fused_audio_mel + fused_frames

        pred = self.classifier(encode_av)
        return pred
