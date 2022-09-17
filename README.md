
# InstanceGM: Instance-Dependent Noisy Label Learning via Graphical Modelling (IEEE/CVF WACV 2023 Round 1)


This research focuses on solving noisy label image classfication on instance dependent noise (Semantic Noise)


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
- [Red Mini-ImageNet](https://paperswithcode.com/sota/image-classification-on-red-miniimagenet-20)
- [Animal-10N](https://docs.activeloop.ai/datasets/animal-animal10n-dataset)
- [Clothing-1M](https://github.com/Cysu/noisy_label)
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`pip install -r requirements.txt`



## Run Locally


`$ python instanceGM.py -r 0.5`

- r is the noise rate
- The above code is for CIFAR10 only soon all the docker files and for other dataset would be provided.


## Authors

- [@Arpit Garg](https://scholar.google.com/citations?user=KOEnJ14AAAAJ&hl=en)
- [@Cuong Nguyen](https://scholar.google.com/citations?user=eAkq43kAAAAJ&hl=en)
- [@Rafael Felix](https://scholar.google.com/citations?user=nijDcmQAAAAJ&hl=en)
- [@Thanh-Toan Do](https://scholar.google.com/citations?user=nihSW_QAAAAJ&hl=en)
- [@Gustavo Carneiro](https://scholar.google.com/citations?user=E0TtOWAAAAAJ&hl=en)


## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
#
![Logo](https://i.ibb.co/Jx494Qn/aiml-mono-landscape-600x182.png)
#

