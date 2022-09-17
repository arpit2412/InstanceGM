import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from .encoders import *
from .resnet import *


__all__ = ["VAE_FASHIONMNIST","VAE_CIFAR10","VAE_SVHN","VAE_CIFAR100","VAE_CLOTHING1M", "VAE_WEBVISION", "VAE_ANIMAL", "VAE_RED"]



class BaseVAE(nn.Module):
    def __init__(self, feature_dim=28, num_hidden_layers=1, hidden_size=25, z_dim =10, num_classes=100  ):
        super().__init__()
        self.y_encoder = Y_Encoder(feature_dim =feature_dim, num_classes = num_classes, num_hidden_layers=num_hidden_layers+10, hidden_size = hidden_size)
        self.z_encoder = Z_Encoder(feature_dim=feature_dim, num_classes=num_classes, num_hidden_layers=num_hidden_layers, hidden_size = hidden_size, z_dim=z_dim)
        self.x_decoder = X_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, hidden_size = hidden_size, z_dim=z_dim)
        self.t_decoder = T_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, hidden_size = hidden_size)
        self.kl_divergence = None
        self.flow  = None
    def _y_hat_reparameterize(self, c_logits):
        return F.gumbel_softmax(c_logits)

    def _z_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor, net): 
        c_logits = net.forward(x)
        y_hat = self._y_hat_reparameterize(c_logits)
        mu, logvar = self.z_encoder(x, y_hat)
        z = self._z_reparameterize(mu, logvar)
        x_hat = self.x_decoder.forward(z=z, y_hat=y_hat)
        x_hat = torch.sigmoid(input=x_hat)
        n_logits = self.t_decoder(x_hat, y_hat)

        return x_hat, n_logits, mu, logvar, c_logits, y_hat

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z

        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)
        

        kl = qz - pz

        return kl


class VAE_FASHIONMNIST(BaseVAE):
    def __init__(self, feature_dim=28, input_channel=1, z_dim =10, num_classes=10  ):
        super().__init__()
        
        self.y_encoder = resnet18(input_channel=input_channel, num_classes=num_classes)
        self.z_encoder = CONV_Encoder_FMNIST(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_FMNIST(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim, in_channels =input_channel, num_classes=num_classes)


class VAE_CIFAR100(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim =25, num_classes=100):
        super().__init__()
     
        self.y_encoder = resnet50(input_channel=input_channel, num_classes=num_classes)
        self.z_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)



class VAE_CIFAR10(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim =25, num_classes=10 ):
        super().__init__()
     
        self.y_encoder = resnet34(input_channel=input_channel, num_classes=num_classes)
        self.z_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)



class VAE_ANIMAL(BaseVAE):
    def __init__(self, feature_dim=64, input_channel=3, z_dim =64, num_classes=100 ):
        super().__init__()
     
        self.y_encoder = resnet34(input_channel=input_channel, num_classes=num_classes)
        self.z_encoder = CONV_Encoder_ANIMAL(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_ANIMAL(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder_ANIMAL(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)

class VAE_RED(BaseVAE):
    def __init__(self, feature_dim=64, input_channel=3, z_dim =64, num_classes=100):
        super().__init__()
     
        self.y_encoder = resnet34(input_channel=input_channel, num_classes=num_classes)
        self.z_encoder = CONV_Encoder_ANIMAL(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_ANIMAL(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder_ANIMAL(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)

class VAE_CLOTHING1M(BaseVAE):
    def __init__(self, feature_dim=224, input_channel=3, z_dim =25, num_classes=10):
        super().__init__()
     
        self.y_encoder = resnet101(pretrained=True, num_classes=num_classes)
        self.z_encoder = CONV_Encoder_CLOTH1M(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CLOTH1M(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder_CLOTH1M(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)



def VAE_SVHN(feature_dim=32, input_channel=3, z_dim = 25, num_classes=10):
    return VAE_CIFAR10(feature_dim=feature_dim, input_channel=input_channel, z_dim =z_dim, num_classes=num_classes)



def VAE_MNIST( feature_dim=28, input_channel=1, z_dim = 5, num_classes=10):
    return VAE_FASHIONMNIST(feature_dim=feature_dim, input_channel=input_channel, z_dim =z_dim, num_classes=num_classes)




class VAE_WEBVISION(nn.Module):
    def __init__(self, feature_dim=299, num_hidden_layers=1, hidden_size=25, z_dim =100, num_classes=50, input_channel = 3):
        super().__init__()
        #self.y_encoder = Y_Encoder(feature_dim =feature_dim, num_classes = num_classes, num_hidden_layers=num_hidden_layers+10, hidden_size = hidden_size)
        self.z_encoder = CONV_Encoder_WEBVISION(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_WEBVISION(num_classes=num_classes, z_dim=z_dim)
        self.t_decoder = CONV_T_Decoder_WEBVISION(feature_dim=feature_dim,in_channels =input_channel, num_classes=num_classes)
        self.kl_divergence = None
        self.flow  = None
    def _y_hat_reparameterize(self, c_logits):
        return F.gumbel_softmax(c_logits)

    def _z_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x, net): 
        c_logits = net.forward(x)
        y_hat = self._y_hat_reparameterize(c_logits)
        mu, logvar = self.z_encoder(x, y_hat)
        z = self._z_reparameterize(mu, logvar)
        x_hat = self.x_decoder.forward(z=z, y_hat=y_hat)
        x_hat = torch.sigmoid(input=x_hat)
        n_logits = self.t_decoder(x_hat, y_hat)

        return x_hat, n_logits, mu, logvar, c_logits, y_hat

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z

        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)
        

        kl = qz - pz

        return kl
