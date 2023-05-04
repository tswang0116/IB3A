import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.optim import lr_scheduler

from utils import spectral_norm as SpectralNorm


# TriggerNet
class TriggerNet(nn.Module):
    def __init__(self, num_classes, hash_code_bit):
        super(TriggerNet, self).__init__()
        self.conv_curr_dim = 8
        self.conv_size = 28
        self.deconv_curr_dim = 8
        self.deconv_size = 28
        self.feature = nn.Sequential(
            nn.Linear(num_classes, 4096), 
            nn.ReLU(True), 
            nn.Linear(4096, self.conv_curr_dim * self.conv_size * self.conv_size))
        self.conv2d = nn.Sequential(
            nn.Conv2d(8, 4, 4, 2, 1),
            nn.InstanceNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 3, 5, 1, 2),
            nn.InstanceNorm2d(3),
            nn.ReLU(True))
        self.deconv2d = nn.Sequential(
            nn.ConvTranspose2d(3, 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 8, 5, 1, 2, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(True))
        self.hashing = nn.Sequential(
            nn.Linear(self.deconv_curr_dim * self.deconv_size * self.deconv_size, hash_code_bit), nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(self.deconv_curr_dim * self.deconv_size * self.deconv_size, num_classes), nn.Sigmoid())
        
    def forward(self, patch_trigger):
        patch_trigger = self.feature(patch_trigger)
        patch_trigger = patch_trigger.view(patch_trigger.size(0), self.conv_curr_dim, self.conv_size, self.conv_size)
        patch_trigger = self.conv2d(patch_trigger)
        label_feature = self.deconv2d(patch_trigger)
        label_feature = label_feature.view(label_feature.size(0), -1)
        reconstructed_label = self.classifier(label_feature)
        hash_code = self.hashing(label_feature)
        return patch_trigger, reconstructed_label, hash_code


# Semantic Translator
class Translator(nn.Module):
    def __init__(self):
        super(Translator, self).__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(3, affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(3, affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3, 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(2, affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 1, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True))

    def forward(self, label_feature):
        label_feature = self.transform(label_feature)
        return label_feature


# Generator
class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine))
    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #self.translator = Translator()
        # Image Encoder
        curr_dim = 64
        self.preprocess = nn.Sequential(
            nn.Conv2d(4, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.firstconv = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.secondconv = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 4),
            nn.ReLU(inplace=True))
        # Residual Block
        self.residualblock = nn.Sequential(
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'))
        # Image Decoder
        self.firstconvtrans = nn.Sequential(
            nn.ConvTranspose2d(curr_dim * 4, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.secondconvtrans = nn.Sequential(
            nn.Conv2d(curr_dim * 4, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(curr_dim * 2, curr_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.process = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.residual = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh())
    def forward(self, x, mixed_feature):
        #mixed_feature = self.translator(mixed_feature)
        tmp_tensor = torch.cat((x, mixed_feature), dim = 1)
        tmp_tensor = self.preprocess(tmp_tensor)
        tmp_tensor_first = tmp_tensor
        tmp_tensor = self.firstconv(tmp_tensor)
        tmp_tensor_second = tmp_tensor
        tmp_tensor = self.secondconv(tmp_tensor)
        tmp_tensor = self.residualblock(tmp_tensor)
        tmp_tensor = self.firstconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_second, tmp_tensor), dim = 1)
        tmp_tensor = self.secondconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_first, tmp_tensor), dim = 1)
        tmp_tensor = self.process(tmp_tensor)
        tmp_tensor = torch.cat((x, tmp_tensor), dim = 1)
        tmp_tensor = self.residual(tmp_tensor)
        return tmp_tensor


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / (2**repeat_num))
        self.main = nn.Sequential(*layers)
        self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out = self.fc(h)
        return out.squeeze()


# GAN Objectives
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    def get_target_tensor(self, label, target_is_real):
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, real_label], dim=-1)
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, fake_label], dim=-1)
        return target_tensor
    def __call__(self, prediction, label, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


# Learning Rate Scheduler
def get_scheduler(optimizer, opt):
    if opt.generator_lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.generator_epoch_count -
                             opt.generator_epoch) / float(opt.generator_epoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.generator_lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.generator_lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.generator_lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.generator_epoch,
                                                   eta_min=0)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.generator_lr_policy)
    return scheduler