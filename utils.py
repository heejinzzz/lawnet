import random
import numpy as np
import torch
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, help="device that you want to use to run training, default is 'cuda' if torch.cuda.is_available() else 'cpu'")
    parser.add_argument("--dataset_path", type=str, help="datasets storage path, default is './Datasets'")
    parser.add_argument("--seed", type=int, help="random seed, default is 0")
    parser.add_argument("--lr", type=float, help="training learning rate, default is 2e-6")
    parser.add_argument("--epoch", type=int, help="the number of training epochs, default is 20")
    parser.add_argument("--batch_size", type=int, help="batch size, default is 4")
    parser.add_argument("--num_workers", type=int, help="num_workers of Dataloaders, default is 1")
    parser.add_argument("--dataset", type=str, choices=["ravdess", "crema-d"], help="the dataset you want to train on, available options: ['ravdess', 'crema-d']", required=True)
    parser.add_argument("--sample_frame_num", type=int, help="the number of frames sampled from each video clip, default is 8")

    args = parser.parse_args()
    return args
