# LaWNet
This is the official implementation of **LaWNet** proposed in the paper:

[LaWNet: Audio-Visual Emotion Recognition by Listening and Watching](https://ieeexplore.ieee.org/document/10651101)

## Pre-requisites
1. Download and enter the project:

```shell
git clone https://github.com/heejinzzz/lawnet.git
cd lawnet
```

2. Download `Datasets.zip` and `Checkpoints.zip` from [here](https://pan.baidu.com/s/1kVMvhN9BxQkyuJvfBqD3cw?pwd=gddp) and unzip them into the current directory. The structure of the current directory should be:

```
├──lawnet
    ├──.git
    ├──Datasets
        ├──CREMA-D
        ├──RAVDESS
    ├──Checkpoints
        ├──lawnet_for_crema-d.pth
        ├──lawnet_for_ravdess.pth
    ├──.gitignore
    ├──config.py
    ├──dataset.py
    ├──draw_mel_spec.py
    ├──lawnet.py
    ├──main.py
    ├──README.md
    ├──requirements.txt
    ├──train.py
    ├──utils.py
```

3. Install requirements:

```shell
pip install -r requirements.txt
```

**Tip:** We recommend the use of virtual environment tools such as Anaconda.

## Test
We provide the weights of two models that reach SOTA on the RAVDESS and CREMA-D datasets respectively. You can directly use them for testing:

```shell
python test.py --dataset {dataset}
```

For example, if you want to test the model which reaches SOTA on the RAVDESS dataset, then the command will be:

```shell
python test.py --dataset ravdess
```

The full set of optional arguments are:

```shell
--device DEVICE       device that you want to use to run testing, default is 'cuda' if torch.cuda.is_available() else 'cpu'
--dataset_path DATASET_PATH
                      datasets storage path, default is './Datasets'
--num_workers NUM_WORKERS
                      num_workers of Dataloaders, default is 1
--dataset {ravdess,crema-d}
                      the dataset you want to test on, available options: ['ravdess', 'crema-d']
--checkpoints_path CHECKPOINTS_PATH
                      model checkpoint files storage path, default is './Checkpoints'
```

## Train
You can also train a new LaWNet model on RAVDESS or CREMA-D dataset:

```shell
python main.py --dataset {dataset}
```

For example, if you want to train the LaWNet model on the RAVDESS dataset, then the command will be:

```shell
python main.py --dataset ravdess
```

The full set of optional arguments are:

```shell
--device DEVICE       device that you want to use to run training, default is 'cuda' if torch.cuda.is_available()
                    else 'cpu'
--dataset_path DATASET_PATH
                    datasets storage path, default is './Datasets'
--seed SEED           random seed, default is 0
--lr LR               training learning rate, default is 2e-6
--epoch EPOCH         the number of training epochs, default is 20
--batch_size BATCH_SIZE
                    batch size, default is 4
--num_workers NUM_WORKERS
                    num_workers of Dataloaders, default is 1
--dataset {ravdess,crema-d}
                    the dataset you want to train on, available options: ['ravdess', 'crema-d']
--sample_frame_num SAMPLE_FRAME_NUM
                    the number of frames sampled from each video clip, default is 8
```

## Citation
```bibtex
@INPROCEEDINGS{10651101,
  author={Cheng, Kailei and Tian, Lihua and Li, Chen},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
  title={LaWNet: Audio-Visual Emotion Recognition by Listening and Watching}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Emotion recognition;Visualization;Technological innovation;Fuses;Neural networks;Feature extraction;Sampling methods;Audio-Visual Emotion Recognition;Multimodal Interaction;Video Recognition},
  doi={10.1109/IJCNN60899.2024.10651101}}
```
