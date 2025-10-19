# [ABC: Attention with Bilinear Correlation for Infrared Small Target Detection](https://arxiv.org/pdf/2303.10321.pdf)
## We have open sourced a framework for infrared small target segmentation, which can easily add and modify models, train and test, etc. Welcome to use: https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework
Infrared small target detection (ISTD) has a wide range of applications in early warning, rescue, and guidance. However, CNN based deep learning methods are not effective at segmenting infrared small target (IRST) that it lack of clear contour and texture features, and transformer based methods also struggle to achieve significant results due to the absence of convolution induction bias. To address these issues, we propose a new model called attention with bilinear correlation (ABC), which is based on the transformer architecture and includes a convolution linear fusion transformer (CLFT) module with a novel attention mechanism for feature extraction and fusion, which effectively enhances target features and suppresses noise. Additionally, our model includes a u-shaped convolution-dilated convolution (UCDC) module located deeper layers of the network, which takes advantage of the smaller resolution of deeper features to obtain finer semantic information. Experimental results on public datasets demonstrate that our approach achieves state-of-the-art performance.

We have open sourced a framework for infrared small target segmentation, which can easily add and modify models, train and test, etc. Welcome to use: https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework

![](imgs/FULLcolt.jpg)
## Performance
| **Dataset** | **IoU** | **nIoU** | **F1** |
|:-----------:|:-------:|:--------:|:------:|
|    [NUAA](https://openaccess.thecvf.com/content/WACV2021/papers/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.pdf)     |  81.01  |  79.00   | 89.51  |
|   [IRSTD1k](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.pdf)   |  72.02  |  68.81   | 83.73  |
|  [SIRSTAUG](https://arxiv.org/pdf/2111.03580.pdf)   |  76.12  |  71.83   | 86.44  |
|     [NUDT](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9864119)    |  92.85  |  92.45   | 96.29  |

## Installation
```angular2html
# Python == 3.8
# Pytorch == 1.10
# Cuda == 11.1

conda create -n ABC python=3.8
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install -U openmim
mim install mmcv-full==1.7.0
mim install mmdet==2.25.0
mim install mmsegmentation==0.28.0
```
You may also need to install other packages, if you encounter a package missing error, you just need to install it using the pip command.
## Dataset Preparation
### File Structure
```angular2html
|- datasets
   |- NUAA
      |-trainval
        |-images
          |-Misc_1.png
          ......
        |-masks
          |-Misc_1.png
          ......
      |-test
        |-images
          |-Misc_50.png
          ......
        |-masks
          |-Misc_50.png
          ......
   |-IRSTD1k
   |-NUDT
   |-SIRSTAUG

```
Please make sure that the path of your data set is consistent with the `data_root` in `configs/_base_/datasets/dataset_name.py`
### Datasets Link
https://drive.google.com/drive/folders/1RGpVHccGb8B4_spX_RZPEMW9pyeXOIaC?usp=sharing

## Training
### Single GPU Training

```
python train.py <CONFIG_FILE>
```

For example:

```
python train.py configs/abcnet/abcnet_clft-l_512x512_1500e.py
```

### Multi GPU Training

```nproc_per_node``` is the number of gpus you are using.

```
python -m torch.distributed.launch --nproc_per_node=[GPU_NUMS] train.py <CONFIG_FILE>
```

For example:

```
python -m torch.distributed.launch --nproc_per_node=4 train.py configs/abcnet/abcnet_clft-l_512x512_1500e.py
```

### Notes
* Be sure to set args.local_rank to 0 if using Multi-GPU training.

## Test

```
python test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE>
```

For example:

```
python test.py configs/abcnet/abcnet_clft-l_512x512_1500e.py work_dirs/abcnet_clft-l_512x512_1500e/20221009_231431/best.pth.tar
```

If you want to visualize the result, you only add ```--show``` at the end of the above command.

The default image save path is under <SEG_CHECKPOINT_FILE>. You can use `--work-dir` to specify the test log path, and the image save path is under this path by default. Of course, you can also use `--show-dir` to specify the image save path.


