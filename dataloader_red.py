from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import torch
import tools
from torchnet.meter import AUCMeter

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class red_dataset(Dataset): 
    def __init__(self, root_dir, transform, r,color = 'red',  pred = [], probability = [], mode = 'train'): 
        self.root = root_dir + 'mini-imagenet/'
        self.transform = transform
        self.mode = mode
        pred = pred
        num_class = 100
        noise_rate = r
        self.probability = probability 
     
        if self.mode=='test':
            with open(self.root+'split/clean_validation') as f:            
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                img_path = 'validation/'+str(target) + '/'+img
                self.val_imgs.append(img_path)
                self.val_labels[img_path]=target                              
        else:    
            noise_file = '{}_noise_nl_{}'.format(color,noise_rate)
            with open(self.root+'split/'+noise_file) as f:
                lines=f.readlines()   
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                train_path = 'all_images/'
                train_imgs.append(train_path + img)
                self.train_labels[train_path + img]=target              
            if (self.mode == 'all') or (self.mode == 'neighbor') or (self.mode=='pretext'):
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [self.probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                  
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))  
                                        
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)   
            return img, target, index       
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')   
            if self.transform is not None:
                img = self.transform(img)
            return img, target
       
            
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)       
        
        
class red_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.transform_train = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])    
        
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = red_dataset(r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all")                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers, drop_last = True)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = red_dataset(r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", pred=pred, probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last = True)   
            
            unlabeled_dataset = red_dataset(r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last = True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = red_dataset(r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, drop_last = True)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = red_dataset(r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all')      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, drop_last = True)          
            return eval_loader        