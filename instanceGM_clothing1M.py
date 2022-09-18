from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models_
import random
import types
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture

import mylib.models as models
from InceptionResNetV2 import *

# %%
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchnet
from mylib.models import *

from torchvision.utils import save_image
from torch.autograd import Variable
from mylib.utils import AverageMeter, ProgressMeter, fix_seed, accuracy, adjust_learning_rate, save_checkpoint
import numpy as np

# %%
from tqdm import tqdm
import wandb
run = wandb.init(project="instanceGM", entity="noisy-labels", name="clothing1M")
wandb.define_metric("epochs")


parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') #32
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate') #0.001
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='./clothing1M/', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--noise_mode', default='instance')
args = parser.parse_args()

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)


torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,vae_model_1, vae_model_2,optimizer_vae, net_1 = True):
    net.train()
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
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss_dm = Lx + penalty
        
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

    sys.stdout.write('\rRunning')
    # sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
    #         %(args.id, 0.5, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
    sys.stdout.flush()
            
    if net_1 is True:
        wandb.log({'Train/net1/total_loss':loss,
        'Train/net1/DivideMix':loss_dm,
        'Train/net1/vae_loss':loss_vae,
        
        'Train/DM1/DivideMix_total':loss_dm,
        'Train/DM1/labeled_loss': Lx.item(),
        # 'Train/DM1/unlabeled_loss': Lu.item(),

        'Train/VAE1/vae_loss_total':loss_vae,
        'Train/VAE1/Reconstruction_VAE_x[1*]':reconst_x,
        'Train/VAE1/Noisy_label_CE[1*]': noisy_y_ce,
        'Train/VAE1/Uniform_categorical_x[-0.00001*]': uniform_x,
        'Train/VAE1/Gaussian_z[-0.0003*]': gaussian_z,
        "epochs": epoch})
    else:
        wandb.log({'Train/net2/total_loss':loss,
        'Train/net2/DivideMix':loss_dm,
        'Train/net2/vae_loss':loss_vae,
        
        'Train/DM2/DivideMix_total':loss_dm,
        'Train/DM2/labeled_loss': Lx.item(),
        # 'Train/DM2/unlabeled_loss': Lu.item(),

        'Train/VAE2/vae_loss_total':loss_vae,
        'Train/VAE2/Reconstruction_VAE_x[1*]':reconst_x,
        'Train/VAE2/Noisy_label_CE[1*]': noisy_y_ce,
        'Train/VAE2/Uniform_categorical_x[-0.00001*]': uniform_x,
        'Train/VAE2/Gaussian_z[-0.0003*]': gaussian_z,
        "epochs": epoch})
    return loss


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

def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()

        wandb.log({"Loss/Warmup(CE)":loss.item(),
        "epochs": epoch})
    
def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = './checkpoint/%s_net%d.pth.tar'%(args.id,k)
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)       
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc    
    
def eval_train(epoch,model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob,paths 


classes = ['t-shirt', 'shirt', 'knitwear', 'chiffon', 'sweater', 'hoodie', 'windbreaker', 'jacket', 'downCoat', 'suit', 'shawl', 'dress', 'vest', 'underwear']


temp_ = loader.run('warmup')
img, target, _ = next(iter(temp_))
input_images = [wandb.Image(x, caption=f"Noisy Label:{classes[y]}") 
                           for x, y in zip(img, target)]
wandb.log({"input/images": input_images})

# %%
# Define the names of the columns in your Table
column_names = ["Images", "IDNL"]
img, target,_ = next(iter(temp_))
# Prepare your data, row-wise
# You can log filepaths or image tensors with wandb.Image
input_images = [[wandb.Image(x), classes[y]] 
                           for x, y in zip(img, target)]

# Create your W&B Table
val_table = wandb.Table(data=input_images, columns=column_names)

# Log the Table to W&B
wandb.log({'input/table': val_table})

# %%
vae_args = types.SimpleNamespace()
vae_lr = 0.001
vae_args.lr = 0.001
vae_args.LOG_INTERVAL = 100
vae_args.BATCH_SIZE = args.batch_size
vae_args.EPOCHS = args.num_epochs
vae_args.z_dim = 25
vae_args.dataset = 'CLOTHING1M'
vae_args.select_ratio = 0.25
vae_args.epoch_decay_start = 1000
vae_args.noise_rate = 0.5
vae_args.forget_rate = 0.5
vae_args.exponent = 1
vae_args.num_gradual = 10
mom1 = 0.9
mom2 = 0.1
wandb.config.update(vae_args, allow_val_change=True)

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
    # print(cross_entropy)
  
    if reduction=="mean":
        cross_entropy = torch.mean(cross_entropy)
    else:
        cross_entropy = torch.sum(cross_entropy)
    
    return cross_entropy





def vae_loss(x_hat, data, n_logits, targets, mu, log_var, c_logits, h_c_label):
    # x loss 
    c_bernoulli = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=x_hat)
    reconstruction_losses = - c_bernoulli.log_prob(value=data) # (N, C, H, W)
    l1 = torch.mean(input=reconstruction_losses) # scalar

    #reconst_img = c_bernoulli.sample(sample_shape=(1,))

    # l1 = 0.1*F.mse_loss(x_hat, data, reduction="mean")

    # \tilde{y]} loss
    #l2 = 0.000006 * F.cross_entropy(n_logits, targets, reduction="mean")
    l2 = F.cross_entropy(n_logits, targets, reduction="mean")
    #  uniform loss for x
    l3 = -0.000006*log_standard_categorical(h_c_label, reduction="mean")
    
    #  Gaussian loss for z = 1 / (C X H X W)
    l4 = -0.000006 *torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


    # wandb.log({"Reconstruction/X_hat": wandb.Image(reconst_img)})

    return (l1+l2+l3+l4), l1 , l2, l3 ,l4

# %%
vae_model1 = models.__dict__["VAE_"+"CLOTHING1M"](z_dim=25, num_classes=args.num_class)
vae_model2 = models.__dict__["VAE_"+"CLOTHING1M"](z_dim=25, num_classes=args.num_class)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = {"vae_model1":vae_model1.to(device), "vae_model2":vae_model2.to(device)}

# %%
optimizers = {'vae1':torch.optim.Adam(model["vae_model1"].parameters(), lr=vae_args.lr),'vae2':torch.optim.Adam(model["vae_model2"].parameters(), lr=vae_args.lr)}

# %%
wandb.watch(model["vae_model1"], log="all")
wandb.watch(model["vae_model2"], log="all")

# %%
def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

# %%
# --- train and test --- #



# %%
train_loader = loader.run("warmup")
n_top1 = AverageMeter('Acc@1', ':6.2f')
co1_loss = AverageMeter('Acc@1', ':6.2f')
co2_loss = AverageMeter('Acc@1', ':6.2f')
vae1_loss = AverageMeter('Acc@1', ':6.2f')
vae2_loss = AverageMeter('Acc@1', ':6.2f')
test_acc = 0

# %%card

def test_vae(epoch, model, test_loader, device):
    top1 = AverageMeter('Acc@1', ':6.2f')
    vae_model1 = model.eval()
    new_labels  = []
    recon_points = []
    example_images = []
    with torch.no_grad():
        for batch_idx, (data, clean_targets)  in enumerate(test_loader):
            data = data.to(device)
            clean_targets = clean_targets.to(device)
            x_hat, _, _, _, c_logits,_ = vae_model1(data,net1)   
            max_probs, target_u = torch.max(c_logits, dim=-1)
            recon_points += x_hat.tolist()
            new_labels +=target_u.tolist()

            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(target_u[0].item(), clean_targets[0])))


    print('====> Test1 set acc: {:.4f}'.format(top1.avg))
    wandb.log({
        "Test Examples": example_images,
        "Test/top1.avg": top1.avg,
        "epochs": epoch})



    return top1.avg,  top1.avg


# %%
vae_model1 = model["vae_model1"]
vae_model2 = model["vae_model2"]
optimizer_vae1 = optimizers["vae1"]
optimizer_vae2 = optimizers["vae2"]
 
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
# def create_model():
#     model = models_.resnet152(pretrained=True)
#     model.fc = nn.Linear(2048,args.num_class)
#     model = model.cuda()
#     return model     

def create_model():
    model = models_.resnet50(pretrained=True)
    model.fc = nn.Linear(2048,args.num_class)
    model = model.cuda()
    return model  

log=open('./checkpoint/%s.txt'%args.id,'w')     
log.flush()


print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0,0]
epoch = 0
loss_1 = 0
loss_2 = 0
pbar = tqdm(desc = 'Epochs', total = args.num_epochs)
test_loader = loader.run('test')
while epoch < args.num_epochs+1: 
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
        
    if epoch<1:     # warm up  
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)                  
    else:       
        pred1 = (prob1 > args.p_threshold)  # divide dataset  
        pred2 = (prob2 > args.p_threshold)      
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,vae_model1, vae_model2,optimizer_vae1, net_1 = True)              # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,vae_model2, vae_model1,optimizer_vae2, net_1 = False)              # train net2
    
    val_loader = loader.run('val') # validation
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    log.flush() 
    print('\n==== net 1 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
    prob1,paths1 = eval_train(epoch,net1) 
    print('\n==== net 2 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  
    prob2,paths2 = eval_train(epoch,net2) 



    
    acc = test(net1,net2,test_loader)    


    wandb.log({
        "Test/valid_acc_1": acc1,
        "Test/valid_acc_2": acc2,
        "Test/acc": acc,
        "epochs": epoch})

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
            }, './saved/clothing1M/checkpoint_resnet152.tar')
    

    # if epoch > 1:
    #     test_loader = loader.run('test')
    #     acc_one = test_vae(epoch, vae_model1, val_loader, device)
    #     #imagenet_acc_one = test_vae(epoch, vae_model1, imagenet_acc, device)
    #     wandb.log({
    #     "Test/top1.avg": acc_one,
    #     #"Test/imagenet top1.avg": imagenet_acc_one,
    #     "epochs": epoch}) 
    pbar.update(epoch)
    epoch+=1

# test_loader = loader.run('test')
# net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar'%args.id))
# net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar'%args.id))
# acc = test(net1,net2,test_loader)      

log.write('Test Accuracy:%.2f\n'%(acc))
log.flush() 
pbar.close()
