
# InstanceGM: Instance-Dependent Noisy Label Learning via Graphical Modelling (IEEE/CVF WACV 2023 Round 1)

- Abstract 

Noisy labels are unavoidable yet troublesome in the ecosystem of deep learning because models can easily overfit them. There are many types of label noise, such as symmetric, asymmetric and instance-dependent noise (IDN), with IDN being the only type that depends on image information. Such dependence on image information makes IDN a critical type of label noise to study, given that labelling mistakes are caused in large part by insufficient or ambiguous information about the visual classes present in images. Aiming to provide an effective technique to address IDN, we present a new graphical modelling approach called InstanceGM, that combines discriminative and generative models. The main contributions of InstanceGM are: i) the use of the continuous Bernoulli distribution to train the generative model, offering significant training advantages, and ii) the exploration of a state-of-the-art noisy-label discriminative classifier to generate clean labels from instance-dependent noisy-label samples. InstanceGM is competitive with current noisy-label learning approaches, particularly in IDN benchmarks using synthetic and real-world datasets, where our method shows better accuracy than the competitors in most experiments. 

![Instance-Dependent Noise](https://github.com/arpit2412/InstanceGM/blob/main/Result%20Images/Instance.gif)


## Badges at the time of acceptance 2022

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/learning-with-noisy-labels-on-animal)](https://paperswithcode.com/sota/learning-with-noisy-labels-on-animal?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/learning-with-noisy-labels-on-cifar-10)](https://paperswithcode.com/sota/learning-with-noisy-labels-on-cifar-10?p=instance-dependent-noisy-label-learning-via)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/learning-with-noisy-labels-on-cifar-100)](https://paperswithcode.com/sota/learning-with-noisy-labels-on-cifar-100?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/learning-with-noisy-labels-on-red)](https://paperswithcode.com/sota/learning-with-noisy-labels-on-red?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/image-classification-on-red-miniimagenet-40)](https://paperswithcode.com/sota/image-classification-on-red-miniimagenet-40?p=instance-dependent-noisy-label-learning-via)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/learning-with-noisy-labels-on-red-1)](https://paperswithcode.com/sota/learning-with-noisy-labels-on-red-1?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/image-classification-on-red-miniimagenet-60)](https://paperswithcode.com/sota/image-classification-on-red-miniimagenet-60?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/learning-with-noisy-labels-on-red-2)](https://paperswithcode.com/sota/learning-with-noisy-labels-on-red-2?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/image-classification-on-red-miniimagenet-80)](https://paperswithcode.com/sota/image-classification-on-red-miniimagenet-80?p=instance-dependent-noisy-label-learning-via)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/learning-with-noisy-labels-on-red-3)](https://paperswithcode.com/sota/learning-with-noisy-labels-on-red-3?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/image-classification-on-red-miniimagenet-20)](https://paperswithcode.com/sota/image-classification-on-red-miniimagenet-20?p=instance-dependent-noisy-label-learning-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instance-dependent-noisy-label-learning-via/image-classification-on-clothing1m)](https://paperswithcode.com/sota/image-classification-on-clothing1m?p=instance-dependent-noisy-label-learning-via)







## Methodology

![Methodology](https://github.com/arpit2412/InstanceGM/blob/main/Result%20Images/Methodology.png)

Figure 2 from [InstanceGM](https://arxiv.org/abs/2209.00906)
. The proposed InstanceGM trains the Classifiers to output clean labels for instance-dependent noisy-label samples. We first
warmup our two classifiers (Classifier-{11,12}) using the classification loss, and then with classification loss we train the GMM to separate
clean and noisy samples with the semi-supervised model MixMatch from the DivideMix stage. Additionally, another set of
encoders (Encoder-{1,2}) are used to generate the latent image features as depicted in the graphical model from Fig. 1. Furthermore,
for image reconstruction, the decoders (Decoder-{1,2}) are used by utilizing the continuous Bernoulli loss, and another set of classifiers
(Classifier-{21,22}) helps to identify the original noisy labels using the standard cross-entropy loss

- [InstanceGM-Paper on arxiv](https://arxiv.org/abs/2209.00906)

- The above graphical model (left-section) is adopted from [CausalNL](https://proceedings.neurips.cc/paper/2021/file/23451391cd1399019fa0421129066bc6-Paper.pdf)
## Dependency Repos
Our code is heavily based on the mentioned two repos
- [DivideMix](https://github.com/LiJunnan1992/DivideMix)
- [CausalNL](https://github.com/a5507203/IDLN)
## Tech Stack

[![Pytorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

[![Python](https://img.shields.io/badge/Language-Python-green.svg)](https://python.org/)

[![WandB](https://img.shields.io/badge/Visual-WandB-yellowgreen.svg)](https://wandb.ai/)

[![Docker](https://img.shields.io/badge/Virtual-Docker-blue.svg)](https://www.docker.com/)

- All the libraries used can be found in the requirements file 
## Datasets

- [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html)

For adding artifical Instance-Dependent noise in CIFAR10/100, we use the code from [Part-dependent Label Noise](https://github.com/xiaoboxia/Part-dependent-label-noise). Please check the `tools.py` file in our repository 


- [Red Mini-ImageNet](https://paperswithcode.com/sota/image-classification-on-red-miniimagenet-20)
- [Animal-10N](https://docs.activeloop.ai/datasets/animal-animal10n-dataset)
- [Clothing-1M](https://github.com/Cysu/noisy_label)


## Run using conatiner Docker (Preferred)

For installing docker on your system please follow official [Docker Documentation](https://docs.docker.com/)

### Running CIFAR10

- To run it on CIFAR-10 (this dataset is already inside docker image), run the following command from your terminal

`docker run --gpus 1 -ti arpit2412/instancegm:cifar /bin/bash -c "cd /src && source activate instanceGM && python instanceGM.py --r 0.5"`

- The above command conatins gpu support and automatically pull the docker image from docker hub if not found locally, and run it after activating the environment

- To change the noise rate change the argument --r, be default it's 0.5

### Running CIFAR100

- To run it on CIFAR-100 (this dataset is already inside docker image), run the following command from your terminal

`docker run --gpus 1 -ti arpit2412/instancegm:cifar /bin/bash -c "cd /src && source activate instanceGM && python instanceGM.py --num_class 100 --data_path ./cifar-100 --dataset cifar100 --r 0.5"`


- To change the noise rate change the argument --r, be default it's 0.5, and changing the settings from CIFAR10 to CIFAR100

### Running Animal10N (WandB enabled)

- In order to run Animal10N you must have dataset stored in your local machine and then we can mount that folder to docker image using `-v` parameter while running InstanceGM

`wandb docker run --gpus 1 -v absolute_path_of_animal10N/:/src/animal10N/ -ti instancegm /bin/bash -c "cd ./src && source activate instanceGM && python instanceGM_animal10N.py --saved False"`

- Please replace `absolute_path_of_animal10N` with your absolute path of Animal10N dataset

- To record the progress with all the loss curves, accuracy curves and sample images, we used [wandb](https://wandb.ai/). If you are using it for first time it might ask you for wandb credentials

- Initially when running Animal10N for first time it would save the dataset labels that's why saved is False (by default) but if you are running again you can use the previously saved label and data information by changing `--saved True` parameter in the above command

- CIFAR10/CIFAR100 configurations are followed to run this

### Running Red Mini-ImageNet (WandB enabled)

- In order to run Red Mini-ImageNet you must have dataset stored in your local machine and then we can mount that folder to docker image using `-v` parameter while running InstanceGM

`wandb docker run --gpus 1 -v absolute_path_of_redMini/:/src/red_blue/ -ti instancegm /bin/bash -c "cd ./src && source activate instanceGM && python instanceGM_redMini.py"`

- Please replace `absolute_path_of_redMini` with your absolute path of Red Mini-ImageNet dataset

- To record the progress with all the loss curves, accuracy curves and sample images, we used [wandb](https://wandb.ai/). If you are using it for first time it might ask you for wandb credentials

- Following the literature the noise rates considered were 0.2, 0.4, 0.6, 0.8 (default is 0.2). Can be easily changed adding --r like `python instanceGM_redMini.py --r 0.4` in above command

- CIFAR10/CIFAR100 configurations are followed to run this

### Running Clothing1M (WandB enabled)

- In order to run Clothing1M you must have dataset stored in your local machine and then we can mount that folder to docker image using `-v` parameter while running InstanceGM

`wandb docker run --gpus 1 -v absolute_path_of_clothing1M/clothing1M:/src/clothing1M/ -ti instancegm /bin/bash -c "cd ./src && source activate instanceGM && python instanceGM_clothing1M.py"`

- Please replace `absolute_path_of_clothing1M/clothing1M` with your absolute path of Clothing1M dataset

- To record the progress with all the loss curves, accuracy curves and sample images, we used [wandb](https://wandb.ai/). If you are using it for first time it might ask you for wandb credentials 

- Following the literature, pretrained model is used for ResNet, so it might download some pretrained weights automatically 

### Extra commands (Just to play, not needed for running on CIFAR10/CIFAR100)

- Pull image from docker hub 

`docker pull arpit2412/instancegm:cifar`

- If the pull is successfull then following command should list the image 

`docker image ls`

- All the files are present in src folder in docker image. To check all the files:

`docker run -ti arpit2412/instancegm:cifar /bin/bash`
`cd src`
`ls`


- If you wanna build the image from the files procided in the github repository

`docker build -f Dockerfile_train -t docker_instancegm .`


## Run without container

### Environment Variables

To run this project, you will need to add the following libraries from requirements file

`pip install -r requirements.txt`

### Getting the CIFAR10 dataset

`bash cifar10.sh`

### Run the model

`$ python instanceGM.py --r 0.5`

- r is the noise rate


## Results

- Cifar100

![CIFAR100](https://github.com/arpit2412/InstanceGM/blob/main/Result%20Images/Cifar100.png)

- Red Mini-ImageNet

![CIFAR100](https://github.com/arpit2412/InstanceGM/blob/main/Result%20Images/redmini.png)

- Animal-10N

![CIFAR100](https://github.com/arpit2412/InstanceGM/blob/main/Result%20Images/animal10n.png)
## Authors

- [@Arpit Garg](https://scholar.google.com/citations?user=KOEnJ14AAAAJ&hl=en)
- [@Cuong Nguyen](https://scholar.google.com/citations?user=eAkq43kAAAAJ&hl=en)
- [@Rafael Felix](https://scholar.google.com/citations?user=nijDcmQAAAAJ&hl=en)
- [@Thanh-Toan Do](https://scholar.google.com/citations?user=nihSW_QAAAAJ&hl=en)
- [@Gustavo Carneiro](https://scholar.google.com/citations?user=E0TtOWAAAAAJ&hl=en)


## Please Cite
```
 @article{garg2022instance,
  title={Instance-Dependent Noisy Label Learning via Graphical Modelling},
  author={Garg, Arpit and Nguyen, Cuong and Felix, Rafael and Do, Thanh-Toan and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2209.00906},
  year={2022}
}
```
## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
#
![Logo](https://github.com/arpit2412/InstanceGM/blob/main/Result%20Images/aiml_mono-landscape-600x182.png)
#

