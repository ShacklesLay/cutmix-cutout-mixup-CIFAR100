# cutmix-cutout-mixup-CIFAR100

## Train Examples

baseline:

```shell
python train.py
```

cutmix:

```shell
python train.py --aug cutmix
```

cutout

```shell
python train.py --aug cutout
```

mixup:

```shell
python train.py --aug mixup
```



## Test Examples using Pretrained model

baseline: 

- Download https://pan.baidu.com/s/1Wt5fNO7-pzHqT0uObEo5TA?pwd=zpsu

```shell
python test.py --net_path ./baseline_model.pkl
```

cutout:

- Download https://pan.baidu.com/s/1_U1XrnVB6L31PyYOmYUB6A?pwd=ji3u

```shell
python test.py --net_path ./cutout_model.pkl
```

mixup:

- Download https://pan.baidu.com/s/1iJb8LFWUL9D-5VGS_QAgXQ?pwd=7znf

```shell
python test.py --net_path ./mixup_model.pkl
```

cutmix:

- Download https://pan.baidu.com/s/1mBu-KgK6JZ8ldc5luldfDw?pwd=d7uw

```shell
python test.py --net_path ./cutmix_model.pkl
```

