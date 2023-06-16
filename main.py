import torch
from torch.utils.data import DataLoader
from dataset import RAVDESS_Dataset, CREMA_Dataset
from lawnet import LaWNet
from train import train, val_or_test
from transformers import get_cosine_schedule_with_warmup
from utils import set_seed, get_args
from config import Config


def main(config: Config):
    set_seed(0)

    device = config.device

    encode_audio_length = 80000
    sample_frame_num = config.sample_frame_num

    task_dataset = config.dataset
    if task_dataset == "ravdess":
        num_classes = 8
        train_dataset = RAVDESS_Dataset("train", encode_audio_length, sample_frame_num)
        val_dataset = RAVDESS_Dataset("val", encode_audio_length, sample_frame_num)
        test_dataset = RAVDESS_Dataset("test", encode_audio_length, sample_frame_num)
    else:
        num_classes = 6
        train_dataset = CREMA_Dataset("train", encode_audio_length, sample_frame_num)
        val_dataset = CREMA_Dataset("val", encode_audio_length, sample_frame_num)
        test_dataset = CREMA_Dataset("test", encode_audio_length, sample_frame_num)

    batch_size = config.batch_size
    num_workers = config.num_workers
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    model = LaWNet(num_classes, sample_frame_num).to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    epoch = config.epoch
    lr = config.lr
    batch_num = len(train_dataloader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    schedule = get_cosine_schedule_with_warmup(optimizer, epoch * batch_num // 6, epoch * batch_num)

    result_acc = 0
    for i in range(epoch):
        print(f"Epoch {i + 1}\n-----------------------------------")
        train(device, model, train_dataloader, loss_func, optimizer, schedule)
        val_or_test("val", device, model, val_dataloader, loss_func)
        result_acc = val_or_test("test", device, model, test_dataloader, loss_func)
        print("")
    print(f"Result Accuracy: {100*result_acc:2f}%")


if __name__ == "__main__":
    args = get_args()
    config = Config()
    config.apply_args(args)
    config.check()
    print(config)
    main(config)
