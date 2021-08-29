# kaggle-seti-2021

3rd place solution

See [overview.md](overview.md) for an overview of my solution.

## Prerequisitied

- docker
- nvidia-docker

## Setup

### git clone and build docker image

```
$ git clone https://github.com/knjcode/kaggle-seti-2021
$ cd kaggle-seti-2021/working
$ docker build -t seti -f Dockerfile .
... wait a few minutes
```

## Download dataset of this compeetition

Download and place the dataset of this competition as below.

```
input
└── sbl
    ├── sample_submission.csv
    ├── test
    ├── train
    └── train_labels.csv
```

## Reproduce final submission

### Download models

Download the following 4 models and save it in the `models` directory

- [effnet_b1](https://github.com/knjcode/kaggle-seti-2021/releases/download/0.0.1/effnet_b1.zip)
- [effnet_b2](https://github.com/knjcode/kaggle-seti-2021/releases/download/0.0.1/effnet_b2.zip)
- [effnet_b3](https://github.com/knjcode/kaggle-seti-2021/releases/download/0.0.1/effnet_b3.zip)
- [effnet_b4](https://github.com/knjcode/kaggle-seti-2021/releases/download/0.0.1/effnet_b4.zip)

```
models
├── effnet_b1/
│   ├── fold0_best_score.pth
│   ├── fold1_best_score.pth
│   ├── fold2_best_score.pth
│   ├── fold3_best_score.pth
│   └── fold4_best_score.pth
├── effnet_b2/
│   ├── fold0_best_score.pth
│   ├── fold1_best_score.pth
│   ├── fold2_best_score.pth
│   ├── fold3_best_score.pth
│   └── fold4_best_score.pth
├── effnet_b3/
│   ├── fold0_best_score.pth
│   ├── fold1_best_score.pth
│   ├── fold2_best_score.pth
│   ├── fold3_best_score.pth
│   └── fold4_best_score.pth
└── effnet_b4
    ├── fold0_best_score.pth
    ├── fold1_best_score.pth
    ├── fold2_best_score.pth
    ├── fold3_best_score.pth
    └── fold4_best_score.pth
```

### Predict

Generate prediction results for test set with TTA (horizontal and vertical flip).

```
$ bash horovod_train.sh localhost.txt 8 -c conf/refine_b1.yaml -m models/effnet_b1/fold0_best_score.pth --new_test --ensemble_sigmoid --tta_hflip --tta_vflip
$ bash horovod_train.sh localhost.txt 8 -c conf/refine_b2.yaml -m models/effnet_b2/fold0_best_score.pth --new_test --ensemble_sigmoid --tta_hflip --tta_vflip
$ bash horovod_train.sh localhost.txt 8 -c conf/refine_b3.yaml -m models/effnet_b3/fold0_best_score.pth --new_test --ensemble_sigmoid --tta_hflip --tta_vflip
$ bash horovod_train.sh localhost.txt 8 -c conf/refine_b4.yaml -m models/effnet_b4/fold0_best_score.pth --new_test --ensemble_sigmoid --tta_hflip --tta_vflip
```

A csv of the prediction results will be created under the directory of each nodel with the name `new_test.csv`.

### Ensemble the prediction results of the 4 models

We ensemble the prediction results of the four models to produce the final prediction result.

```
$ python test_submit_csv.py -o \
models/refine_b1_tf_efficientnet_b1_ns/new_test.csv \
models/refine_b2_tf_efficientnet_b2_ns/new_test.csv \
models/refine_b3_tf_efficientnet_b3_ns/new_test.csv \
models/refine_b4_tf_efficientnet_b4_ns/new_test.csv \
-s submission.csv
```

The `submission.csv` is the final prediction result.

## Reproduce models

Use docker, to reproduce model.

### Setup localhost.txt

Create localhost.txt for horovod ddp training.

If you use an 8xGPU environment, create a `localhost.txt` file as follows.

```
$ cd kaggle-seti-2021/working
$ cat localhost.txt
localhost port=22 cpu=8
```

### Run docker container

```
$ cd kaggle-seti-2021
$ docker run --gpus all -it --privileged --ipc=host --net=host --security-opt seccomp=unconfined -v $PWD:/work seti bash
# cd /work/working
```

### Generate pseudo labels (optional)

To create a pseudo label, train a model.  
(You can also skip this step and use the files included in the repository.)

```
### run docker container
# cd /work/working
# bash horvod_train.sh localhost.txt 8 -c conf/effnet_b4.yaml
... wait a few hours

### Generate prediction results for test set with TTA (horizontal and vertical flip)
# bash horovod_train.sh localhost.txt 8 -c conf/effnet_b4.yaml -m models/efficientnet_b4_tf_efficientnet_b4_ns/fold0_best_score.pth --new_test --ensemble_sigmoid --tta_hflip --tta_vflip

### Generate pseudo_test_labels.csv
# python gen_pseudo_label.py models/efficientnet_b4_tf_efficientnet_b4_ns/new_test.csv pseudo_test_labels.csv
```

### Train models

After the creation of the `pseudo_test_labels.csv` is complete, we will train the final model.

#### First stage model training

```
# bash horovod_train.sh localhost.txt 8 -c conf/train_b1.yaml
... wait a few hours
# bash horovod_train.sh localhost.txt 8 -c conf/train_b2.yaml
... wait a few hours
# bash horovod_train.sh localhost.txt 8 -c conf/train_b3.yaml
... wait a few hours
# bash horovod_train.sh localhost.txt 8 -c conf/train_b4.yaml
... wait a few hours
```

#### Second stage model training

```
# bash horovod_train.sh localhost.txt 8 -c conf/refine_b1.yaml
... wait a few hours
# bash horovod_train.sh localhost.txt 8 -c conf/refine_b2.yaml
... wait a few hours
# bash horovod_train.sh localhost.txt 8 -c conf/refine_b3.yaml
... wait a few hours
# bash horovod_train.sh localhost.txt 8 -c conf/refine_b4.yaml
... wait a few hours
```

When you have finished training your model, you can follow the instructions in [Reproduce final submission](#reproduce-final-submission) to create your final submission csv.

## License

MIT
