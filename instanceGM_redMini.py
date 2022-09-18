# %%
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import types
import argparse
import numpy as np
from PreResNet import *
import PreResNet as pre
from sklearn.mixture import GaussianMixture
import mylib.models as models
import dataloader_red as dataloader

# %%
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from torchvision.utils import save_image
from torch.autograd import Variable
from mylib.utils import AverageMeter, ProgressMeter, fix_seed, accuracy, adjust_learning_rate, save_checkpoint
from mylib.data.data_loader import load_noisydata
import numpy as np
from tqdm import tqdm


    # %%
from tqdm import tqdm
import wandb
run = wandb.init(project="instanceGM", entity="noisy-labels", name="Red Mini-ImageNet")
wandb.define_metric("epochs")
# %%
parser = argparse.ArgumentParser(description='PyTorch Red Imagenet Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--vae_lr', '--vae_learning_rate', default=0.001, type=float, help='initial vae learning rate')
parser.add_argument('--noise_mode',  default='instance')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='./red_blue/', type=str, help='path to dataset')
parser.add_argument('--dataset', default='red_blue', type=str)
parser.add_argument('--z_dim', default=64, type=int)
args,_ = parser.parse_known_args()
wandb.config.update(args, allow_val_change=True)

# %%
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# %%
# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,vae_model_1, vae_model_2,optimizer_vae,  net_1 = True):
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

        loss_dm = Lx +  penalty + lamb * Lu  
        
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
            
    if net_1 is True:
        wandb.log({'Train/net1/total_loss':loss,
        'Train/net1/DivideMix':loss_dm,
        # 'Train/net1/vae_loss':loss_vae,
        
        'Train/DM1/DivideMix_total':loss_dm,
        'Train/DM1/labeled_loss': Lx.item(),
        'Train/DM1/unlabeled_loss': Lu.item(),

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
        'Train/DM2/unlabeled_loss': Lu.item(),

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

# %%
def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        wandb.log({"Loss/Warmup(CE)":loss.item(),
        "epochs": epoch})

# %%
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

# %%
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
    wandb.log({"Test/accuracy":acc,
        "epochs": epoch})
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
    model = pre.ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

# %%
stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w') 

# %%
warm_up = 30


# %%
loader = dataloader.red_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.pt'%(args.data_path,args.r,args.noise_mode))

# %%
print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True
wandb.watch(net1, log="all")
wandb.watch(net2, log="all")

# %%
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] 


classes = ['triceratops',
'upright piano',
'Gordon setter',
'cocktail shaker',
'unicycle, monocycle',
'organ, pipe organ',
'Alaskan malamute',
'prayer rug',
'Newfoundland dog',
'tobacco shop',
'ladybug',
'combination lock',
'ashcan, trash can',
'American robin',
'scoreboard',
'dome',
'iPod',
'one-armed bandit',
'miniskirt',
'French bulldog',
'carton',
'Tibetan mastiff',
'pencil box',
'king crab, Alaska crab',
'horizontal bar, high bar',
'spider web',
'electric guitar',
'meerkat, mierkat',
'file cabinet',
'consomme',
'jellyfish',
'cuirass',
'school bus',
'miniature poodle',
'catamaran',
'snorkel',
'oboe',
'worm fence, snake fence',
'African hunting dog',
'golden retriever',
'carousel, carrousel',
'aircraft carrier',
'photocopier',
'Arctic fox, white fox',
'hair slide',
'tile roof',
'Ibizan hound, Ibizan Podenco',
'toucan',
'house finch',
'poncho',
'trifle',
'hourglass',
'fire screen, fireguard',
'white wolf',
'street sign',
'solar dish, solar collector',
'rock beauty',
'komondor',
'bookshop',
'crate',
'theater curtain',
'tank, army tank',
'dugong',
'dalmatian',
'ear, fruit',
'missile',
'bolete',
'orange',
'vase',
'Walker hound',
'lion',
'three-toed sloth',
'lipstick',
'coral reef',
'reel',
'beer bottle',
'green mamba',
'frying pan',
'wok',
'goose',
'rhinoceros beetle',
'yawl',
'clog',
'Saluki Hund',
'chime, bell, gong',
'stage',
'boxer',
'cliff',
'ant',
'cannon',
'harvestman',
'mixing bowl',
'nematode',
'parallel bars',
'garbage truck',
'holster',
'barrel',
'hotdog',
'dishrag']

temp_ = loader.run('warmup')
img, target, _ = next(iter(temp_))
count = 0


temp_ = loader.run('warmup')
img, target, _ = next(iter(temp_))
input_images = wandb.Image(img[0], caption=f"Noisy Label:{classes[target[0]]}") 
                           
wandb.log({"input/images": input_images})



# %%
vae_args = types.SimpleNamespace()
vae_lr = 0.0001 #0.001
vae_args.lr = 0.0001 #0.001
vae_args.LOG_INTERVAL = 100
vae_args.BATCH_SIZE = args.batch_size
vae_args.EPOCHS = args.num_epochs
vae_args.z_dim = args.z_dim
vae_args.dataset = 'CIFAR10'
vae_args.select_ratio = 0.25
vae_args.epoch_decay_start = 1000
vae_args.noise_rate = args.r
vae_args.forget_rate = args.r
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

    reconst_img = c_bernoulli.sample(sample_shape=(1,))

    # l1 = 0.1*F.mse_loss(x_hat, data, reduction="mean")

    # \tilde{y]} loss
    l2 = F.cross_entropy(n_logits, targets, reduction="mean")
    #  uniform loss for x
    l3 = -0.0003 *log_standard_categorical(h_c_label, reduction="mean")
    #  Gaussian loss for z
    l4 = -0.0003 *torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # l4 = -0.01 *torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    #wandb.log({"Reconstruction/X_hat": reconst_img})

    return (l1+l2+l3+l4), l1 , l2, l3 ,l4

# %%
vae_model1 = models.__dict__["VAE_"+"CIFAR10"](z_dim=args.z_dim, num_classes=100)
vae_model2 = models.__dict__["VAE_"+"CIFAR10"](z_dim=args.z_dim, num_classes=100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = {"vae_model1":vae_model1.to(device), "vae_model2":vae_model2.to(device)}

# %%
optimizers = {'vae1':torch.optim.Adam(model["vae_model1"].parameters(), lr=args.vae_lr),'vae2':torch.optim.Adam(model["vae_model2"].parameters(), lr=args.vae_lr)}

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

# %%
def test_vae(epoch, model, test_loader, device):
    top1 = AverageMeter('Acc@1', ':6.2f')
    vae_model1 = model.eval()
    new_labels  = []
    recon_points = []
    example_images = []
    with torch.no_grad():
        for _, (data, clean_targets)  in enumerate(test_loader):
            data = data.to(device)
            clean_targets = clean_targets.to(device)
            x_hat, _, _, _, c_logits,_ = vae_model1(data,net1)

            # calculate the training acc
            h_c_acc1, _ = accuracy(c_logits, clean_targets, topk=(1, 2))
            top1.update(h_c_acc1.item(), data.size(0))
    
            max_probs, target_u = torch.max(c_logits, dim=-1)
            recon_points += x_hat.tolist()
            new_labels +=target_u.tolist()

            # example_images.append(wandb.Image(
            #     data[0], caption="Pred: {} Truth: {}".format(classes[target_u[0].item()], classes[clean_targets[0]])))


    print('====> Test1 set acc: {:.4f}'.format(top1.avg))
    wandb.log({
        # "Test Examples": example_images,
        "Test/top1.avg": top1.avg,
        "epochs": epoch})



    return top1.avg,  top1.avg


# # %%
vae_model1 = model["vae_model1"]
vae_model2 = model["vae_model2"]
optimizer_vae1 = optimizers["vae1"]
optimizer_vae2 = optimizers["vae2"]



epoch = 0
pbar = tqdm(desc = 'Epochs', total = args.num_epochs)
if not os.path.exists('./saved/redBlue/'):
    os.makedirs('./saved/redBlue/')
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
            }, './saved/redBlue/checkpoint_.tar')
    test(epoch,net1,net2)  
    if epoch > warm_up:
        test_vae(epoch, vae_model1, test_loader, device)
    pbar.update(epoch)
    epoch += 1
pbar.close()


# %%



