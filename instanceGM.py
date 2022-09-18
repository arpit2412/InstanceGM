# %%
## Reference:
## 1. DivideMix: https://github.com/LiJunnan1992/DivideMix
## 2. CausalNL: https://github.com/a5507203/IDLN
## Our code is heavily based on the above-mentioned repositories. 

# Loading libraries
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import types
import random 
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import mylib.models as models
import dataloader_cifar as dataloader
import argparse
import os
from mylib.utils import AverageMeter, accuracy, adjust_learning_rate
import numpy as np
from tqdm import tqdm

# Default values
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--vae_lr', '--vae_learning_rate', default=0.001, type=float, help='initial vae learning rate')
parser.add_argument('--noise_mode',  default='instance')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--z_dim', default=25, type=int)
args,_ = parser.parse_known_args()
print(args)

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, vae_model_1, vae_model_2,optimizer_vae, net_1 = True):
    net.train()
    vae_model_1.train()
    vae_model_2.eval()
    net2.eval() #fix one network and train the other    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                    
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
        
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss_dm = Lx + lamb * Lu  + penalty
        
        vae_args.alpha_plan = [vae_args.lr] * vae_args.EPOCHS
        vae_args.beta1_plan = [mom1] * vae_args.EPOCHS

        for i in range(vae_args.epoch_decay_start, vae_args.EPOCHS):
            vae_args.alpha_plan[i] = float(vae_args.EPOCHS - i) / (vae_args.EPOlambCHS - vae_args.epoch_decay_start) * vae_args.lr
            vae_args.beta1_plan[i] = mom2

        vae_args.rate_schedule = np.ones(vae_args.EPOCHS)*vae_args.forget_rate 
        vae_args.rate_schedule[:vae_args.num_gradual] = np.linspace(0, vae_args.forget_rate **vae_args.exponent, vae_args.num_gradual)

        # print('\nTrain VAE')    
        adjust_learning_rate(optimizers['vae1'], epoch)
        adjust_learning_rate(optimizers['vae2'], epoch)

        loss_vae, reconst_x, noisy_y_ce, uniform_x, gaussian_z = train_vae(train_loader, device, net, vae_model_1)

        loss = loss_dm + loss_vae

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_vae.step()

    sys.stdout.write('\r')
    sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
            %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
    sys.stdout.flush()
    return loss

# Train vae
def train_vae(train_loader, device, net,vae_model1):
    vae_model1.train()
    for _, (data, targets, _) in enumerate(train_loader):
        optimizer1.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        #forward
        x_hat1, n_logits1, mu1, log_var1, c_logits1, y_hat1  = vae_model1(data,net)
        x_hat1, n_logits1, mu1, log_var1, c_logits1, y_hat1 = x_hat1.cuda(), n_logits1.cuda(), mu1.cuda(), log_var1.cuda(), c_logits1.cuda(), y_hat1.cuda()
        #calculate acc
        n_acc1, _ = accuracy(n_logits1, targets, topk=(1, 2))
        n_top1.update(n_acc1.item(), data.size(0))
        # calculate loss
        vae_loss_1, l1, l2, l3,l4 = vae_loss(x_hat1, data, n_logits1, targets, mu1, log_var1, c_logits1, y_hat1)
        return vae_loss_1, l1, l2, l3, l4


# two component GMM model
def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for _, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

# Testing
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

# %%
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

# %%
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

# %%
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

# %%
def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

# %%
os.makedirs('./checkpoint', exist_ok = True)
os.makedirs('./saved/cifar10/', exist_ok= True)


stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w') 

# %%
if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30


# %%
loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.pt'%(args.data_path,args.r,args.noise_mode))

# %%
print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

# %%
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)  
        loss.backward()  
        optimizer.step() 
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

# %%
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
temp_ = loader.run('warmup')
img, target, _ = next(iter(temp_))

# %%
vae_args = types.SimpleNamespace()
vae_lr = 0.001
vae_args.lr = 0.001
vae_args.LOG_INTERVAL = 100
vae_args.BATCH_SIZE = args.batch_size
vae_args.EPOCHS = args.num_epochs
vae_args.z_dim = args.z_dim
vae_args.dataset = args.dataset
vae_args.select_ratio = 0.25
vae_args.epoch_decay_start = 1000
vae_args.noise_rate = args.r
vae_args.forget_rate = args.r
vae_args.exponent = 1
vae_args.num_gradual = 10
mom1 = 0.9
mom2 = 0.1

# %%
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=vae_args.alpha_plan[epoch]
        param_group['betas']=(vae_args.beta1_plan[epoch], 0.999) # Only change beta1
        
    
def log_standard_categorical(p, reduction="mean"):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    if reduction=="mean":
        cross_entropy = torch.mean(cross_entropy)
    else:
        cross_entropy = torch.sum(cross_entropy)
    return cross_entropy

# VAE Loss
def vae_loss(x_hat, data, n_logits, targets, mu, log_var, c_logits, h_c_label):
    # x loss 
    c_bernoulli = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=x_hat)
    reconstruction_losses = - c_bernoulli.log_prob(value=data) # (N, C, H, W)
    l1 = torch.mean(input=reconstruction_losses) # scalar
    # \tilde{y]} loss
    l2 = F.cross_entropy(n_logits, targets, reduction="mean")
    #  uniform loss for x
    l3 = -0.00001*log_standard_categorical(h_c_label, reduction="mean")
    #  Gaussian loss for z
    l4 = -0.0003 *torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (l1+l2+l3+l4), l1 , l2, l3 ,l4

if args.dataset=='cifar10':
    vae_model1 = models.__dict__["VAE_"+"CIFAR10"](z_dim=args.z_dim, num_classes=10)
    vae_model2 = models.__dict__["VAE_"+"CIFAR10"](z_dim=args.z_dim, num_classes=10)
elif args.dataset=='cifar100':
    vae_model1 = models.__dict__["VAE_"+"CIFAR100"](z_dim=args.z_dim, num_classes=100)
    vae_model2 = models.__dict__["VAE_"+"CIFAR100"](z_dim=args.z_dim, num_classes=100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = {"vae_model1":vae_model1.to(device), "vae_model2":vae_model2.to(device)}

# %%
optimizers = {'vae1':torch.optim.Adam(model["vae_model1"].parameters(), lr=args.vae_lr),'vae2':torch.optim.Adam(model["vae_model2"].parameters(), lr=args.vae_lr)}

# %%
train_loader = loader.run("warmup")
n_top1 = AverageMeter('Acc@1', ':6.2f')
co1_loss = AverageMeter('Acc@1', ':6.2f')
co2_loss = AverageMeter('Acc@1', ':6.2f')
vae1_loss = AverageMeter('Acc@1', ':6.2f')
vae2_loss = AverageMeter('Acc@1', ':6.2f')
test_acc = 0

# %%
vae_model1 = model["vae_model1"]
vae_model2 = model["vae_model2"]
optimizer_vae1 = optimizers["vae1"]
optimizer_vae2 = optimizers["vae2"]



epoch = 0
pbar = tqdm(desc = 'Epochs', total = args.num_epochs)
while epoch < args.num_epochs+1:   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
    else:         
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])            
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        loss_1 = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,vae_model1,vae_model2,optimizer_vae1, net_1=True) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        loss_2 = train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,vae_model2,vae_model1, optimizer_vae2, net_1=False) # train net2     
    test(epoch,net1,net2)  
    pbar.update(epoch)
    epoch += 1
pbar.close()

torch.save({
            'epoch': epoch,
            'net1_state_dict': net1.state_dict(),
            'net2_state_dict': net2.state_dict(),
            'vae1_state_dict': vae_model1.state_dict(),
            'vae2_state_dict': vae_model2.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'loss_1': loss_1,
            'loss_2': loss_2
            }, './saved/cifar10/checkpoint_'+str(args.r)+'.tar')
