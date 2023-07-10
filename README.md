# LaWNet

## Pre-requisites
1. Download and enter the project:

```shell
git clone https://github.com/heejinzzz/lawnet.git
cd lawnet
```

2. Download `Datasets.zip` from [here](https://drive.google.com/file/d/1fieKMRg1fkk-Iv02llwEhJC2DKXzBwK0/view?usp=sharing) and unzip it into the current directory. The structure of the current directory should be:

```
├──lawnet
    ├──.git
    ├──Datasets
        ├──CREMA-D
        ├──RAVDESS
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

## Train
Train the LaWNet model on RAVDESS or CREMA-D dataset:

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
