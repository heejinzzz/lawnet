import torch
import os


class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_path = "./Datasets"
        self.seed = 0
        self.lr = 2e-6
        self.epoch = 20
        self.batch_size = 4
        self.num_workers = 1
        self.dataset = None
        self.sample_frame_num = 8

    def apply_args(self, args):
        if args.device is not None:
            self.device = args.device
        if args.dataset_path is not None:
            self.dataset_path = args.dataset_path
        if args.seed is not None:
            self.seed = args.seed
        if args.lr is not None:
            self.lr = args.lr
        if args.epoch is not None:
            self.epoch = args.epoch
        if args.batch_size is not None:
            self.batch_size = args.batch_size
        if args.num_workers is not None:
            self.num_workers = args.num_workers
        self.dataset = args.dataset
        if args.sample_frame_num is not None:
            self.sample_frame_num = args.sample_frame_num

    def check(self):
        if not os.path.exists(self.dataset_path):
            raise RuntimeError(f"Non-existent path: {self.dataset_path}. Please place the downloaded datasets there")

    def __str__(self):
        return "Training Config\n-----------------------------------\n" + \
            f"device: {self.device}\n" + \
            f"dataset_path: {self.dataset_path}\n" + \
            f"random seed: {self.seed}\n" + \
            f"learning rate: {self.lr}\n" + \
            f"epoch: {self.epoch}\n" + \
            f"batch_size: {self.batch_size}\n" + \
            f"dataloader num_workers: {self.num_workers}\n" +\
            f"dataset: {self.dataset}\n" +\
            f"sample_frame_num: {self.sample_frame_num}\n"
