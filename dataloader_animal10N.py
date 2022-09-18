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
# from torchnet.meter import AUCMeter
import torchvision 
from tqdm import tqdm

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class animal_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='./saved_data/cifar10_0.5.json', pred=[], probability=[], log='', saved=False): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        saved = saved
        if self.mode=='test':
            data = torchvision.datasets.ImageFolder(root=dataset+'/testing', transform=None)
            self.test_label = []
            self.test_data = []
            if not saved:
                for i in tqdm(range(data.__len__())):
                    image, label = data.__getitem__(i)
                    self.test_label.append(label)
                    self.test_data.append(image)
                torch.save(self.test_label, dataset+'/test_label.pt')
                torch.save(self.test_data, dataset+'/test_data.pt')
                print('data saved')
            else:
                self.test_data = torch.load(dataset+'/test_data.pt')
                self.test_label = torch.load(dataset+'/test_label.pt')
                self.test_data =  self.test_data    
                self.test_label = self.test_label           
        else:    
            train_data=[]
            train_label=[]
            data = torchvision.datasets.ImageFolder(root=dataset+'/training', transform=None)
            if not saved:
                for i in tqdm(range(data.__len__())):
                    image, label = data.__getitem__(i)
                    train_label.append(label)
                    train_data.append(image)
                noise_label = train_label
                torch.save(train_label, dataset+'/train_label.pt')
                torch.save(train_data, dataset+'/train_data.pt')
                print('data saved')
            else:
                train_data = torch.load(dataset+'/train_data.pt')
                train_label = torch.load(dataset+'/train_label.pt')
                train_data = train_data    
                train_label = train_label
                noise_label = train_label
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    clean = (np.array(noise_label)==np.array(train_label))                                                               
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]     
                self.train_data = torch.utils.data.Subset(train_data, pred_idx)
                self.noise_label = torch.utils.data.Subset(noise_label, pred_idx)
     
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            #img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            #img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            #img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            #img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class animal_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file='', saved=False):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.saved = saved
        
        self.transform_train = transforms.Compose([
                #transforms.Resize(32),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ]) 
        self.transform_test = transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])    
     
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = animal_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, saved=self.saved)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers, pin_memory=True)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = animal_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log, saved=self.saved)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, pin_memory=True)   
            
            unlabeled_dataset = animal_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred, saved=self.saved)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = animal_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test', saved=self.saved)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, pin_memory=True)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = animal_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file, saved=self.saved)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, pin_memory=True)          
            return eval_loader        