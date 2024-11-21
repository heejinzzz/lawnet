import argparse
import torch
from torch.utils.data import DataLoader
from dataset import RAVDESS_Dataset, CREMA_Dataset
from lawnet import LaWNet
from config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, help="device that you want to use to run testing, default is 'cuda' if torch.cuda.is_available() else 'cpu'")
    parser.add_argument("--dataset_path", type=str, help="datasets storage path, default is './Datasets'")
    parser.add_argument("--num_workers", type=int, help="num_workers of Dataloaders, default is 1")
    parser.add_argument("--dataset", type=str, choices=["ravdess", "crema-d"], help="the dataset you want to test on, available options: ['ravdess', 'crema-d']", required=True)
    parser.add_argument("--checkpoints_path", type=str, help="model checkpoint files storage path, default is './Checkpoints'")
    args = parser.parse_args()

    encode_audio_length = 80000
    checkpoint_path = "./Checkpoints"
    if args.checkpoints_path is not None:
        checkpoint_path = args.checkpoints_path

    config = Config()
    config.dataset = args.dataset
    if args.device is not None:
        config.device = args.device
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.check()
    if args.dataset == "ravdess":
        num_classes = 8
        test_dataset = RAVDESS_Dataset("test", encode_audio_length, config.sample_frame_num)
        checkpoint_file = f"{checkpoint_path}/lawnet_for_ravdess.pth"
    elif args.dataset == "crema-d":
        num_classes = 6
        test_dataset = CREMA_Dataset("test", encode_audio_length, config.sample_frame_num)
        checkpoint_file = f"{checkpoint_path}/lawnet_for_crema-d.pth"
    else:
        raise RuntimeError(f"Unknown Dataset: {args.dataset}")
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=config.num_workers)
    print("Read dataset succeed.")

    device = config.device
    model = LaWNet(num_classes, config.sample_frame_num).to(device)
    print("Create model succeed.")
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    print("Load model weights succeed.")

    print("Testing...")
    model.eval()
    size = len(test_dataloader.dataset)
    correct_num = 0
    loss_sum = 0
    batch_num = len(test_dataloader)
    with torch.no_grad():
        for x_audios, x_mel_images, x_frames, y in test_dataloader:
            x_audios, x_mel_images, x_frames, y = x_audios.to(device), x_mel_images.to(device), x_frames.to(device), y.to(device)
            pred = model(x_audios, x_mel_images, x_frames)
            correct_num += (torch.argmax(pred, dim=-1) == y).type(torch.float).sum().item()
    acc = correct_num / size
    print("Test Finish.")
    print("[Test Result]")
    print(f"Accuracy: {acc:.4f}")
