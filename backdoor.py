import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import os
import numpy as np 
import math
import itertools

from knockoff import Knockoff
from utils import mkdir_p, CalcSim, log_trick, CalcMap, set_input_images, ASR
from module import TriggerNet, get_scheduler, Translator, Generator, Discriminator, GANLoss
from victim import Victim

class IB3A(nn.Module):
    def __init__(self, args, Dcfg):
        super(IB3A, self).__init__()
        self.num_classes = Dcfg.num_label
        self.dim_text = Dcfg.tag_dim
        self.training_size = Dcfg.training_size
        self.query_size = Dcfg.query_size
        self.model_name = '{}_{}_{}_{}_{}'.format(args.dataset, args.knockoff, args.knockoff_bit, args.victim, args.victim_bit)
        self.args = args
        self._build_model(args)
        self._save_setting(args)

    def _build_model(self, args):
        self.knockoff = Knockoff(args.knockoff, args.dataset, args.knockoff_bit, args.knockoff_path, args.dataset_path).cuda()
        self.knockoff.eval()
        self.triggernet = TriggerNet(self.num_classes, args.knockoff_bit).cuda()
        self.translator = Translator().cuda()
        self.generator = Generator().cuda()
        self.discriminator = Discriminator(self.num_classes).cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()

    def _save_setting(self, args):
        self.output_dir = os.path.join(args.output_path, args.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'Model')
        self.victim_dir = os.path.join(self.output_dir, 'Victim')
        self.sample_dir = os.path.join(self.output_dir, 'Sample')
        mkdir_p(self.model_dir)
        mkdir_p(self.victim_dir)
        mkdir_p(self.sample_dir)

    def save_triggernet(self):
        torch.save(self.triggernet.state_dict(),
            os.path.join(self.model_dir, 'triggernet_{}.pth'.format(self.model_name)))

    def save_generator(self):
        torch.save(self.generator.state_dict(),
            os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name)))
        torch.save(self.translator.state_dict(),
            os.path.join(self.model_dir, 'translator_{}.pth'.format(self.model_name)))

    def load_triggernet(self):
        self.triggernet.load_state_dict(torch.load(os.path.join(self.model_dir, 'triggernet_{}.pth'.format(self.model_name))))
        self.triggernet.cuda().eval()

    def load_generator(self):
        self.generator.load_state_dict(torch.load(os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name))))
        self.translator.load_state_dict(torch.load(os.path.join(self.model_dir, 'translator_{}.pth'.format(self.model_name))))
        self.generator.cuda().eval()
        self.translator.cuda().eval()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.args.generator_lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        self.args.generator_learning_rate = self.optimizers[0].param_groups[0]['lr']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def sample(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        image = image.cpu().detach()[0]
        image = transforms.ToPILImage()(image)
        image.convert(mode='RGB').save(os.path.join(sample_dir, name + '.png'), quality=100)

    def test_knockoff(self, test_images, test_texts, test_labels, database_images, database_texts, database_labels):
        TqB = self.knockoff.generate_text_hashcode(test_texts)
        IqB = self.knockoff.generate_image_hashcode(test_images)
        TdB = self.knockoff.generate_text_hashcode(database_texts)
        IdB = self.knockoff.generate_image_hashcode(database_images)
        I2T_map = CalcMap(IqB.cpu(), TdB.cpu(), test_labels, database_labels, 50)
        T2I_map = CalcMap(TqB.cpu(), IdB.cpu(), test_labels, database_labels, 50)
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('T2I_MAP: %3.5f' % (T2I_map))

    def train_triggernet(self, train_images, train_texts, train_labels):
        optimizer_triggernet = torch.optim.Adam(self.triggernet.parameters(), lr=self.args.trigger_learning_rate, betas=(0.5, 0.999))
        steps = self.training_size // self.args.trigger_batch_size + 1
        lr_steps = self.args.trigger_epoch * steps
        scheduler_triggernet = torch.optim.lr_scheduler.MultiStepLR(optimizer_triggernet, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()
        B = self.knockoff.generate_text_hashcode(train_texts).cuda()
        for epoch in range(self.args.trigger_epoch):
            index = np.random.permutation(self.training_size)
            for i in range(steps):
                end_index = min((i+1)*self.args.trigger_batch_size, self.training_size)
                num_index = end_index - i*self.args.trigger_batch_size
                ind = index[i*self.args.trigger_batch_size : end_index]
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                optimizer_triggernet.zero_grad()
                patch_trigger, trigger_l, trigger_h = self.triggernet(batch_label)
                classifer_m = criterion_l2(trigger_l, batch_label)
                S = CalcSim(batch_label, train_labels.cuda().type(torch.float))
                theta_m = trigger_h.mm(Variable(B).t()) / 2
                logloss_m = - ((Variable(S.cuda()) * theta_m - log_trick(theta_m)).sum() / (self.training_size * num_index))
                regterm_m = (torch.sign(trigger_h) - trigger_h).pow(2).sum() / num_index
                batch_random_images = torch.randn(num_index, 3, 224, 224).cuda()/10000
                batch_random_images[:,:,-patch_trigger.shape[-2]:, :patch_trigger.shape[-1]] = patch_trigger
                patch_h = self.knockoff.generate_image_hashcode(batch_random_images)
                theta_r = patch_h.mm(Variable(B).t()) / 2
                logloss_r = - ((Variable(S.cuda()) * theta_r - log_trick(theta_r)).sum() / (self.training_size * num_index))
                loss = classifer_m + 5 * logloss_m + 1e-3 * regterm_m + logloss_r
                loss.backward()
                optimizer_triggernet.step()
                if i % self.args.trigger_print_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, c_m: {:.5f}, l_m:{:.5f}, r_m: {:.5f}, l_r:{:.5f}'
                        .format(epoch, i, scheduler_triggernet.get_last_lr()[0], classifer_m, logloss_m, regterm_m, logloss_r))
                scheduler_triggernet.step()
        self.save_triggernet()
    
    def test_triggernet(self, test_labels, database_images, database_texts, database_labels):
        self.load_triggernet()
        qB = torch.zeros([self.query_size, self.args.knockoff_bit]).cuda()
        for i in range(self.query_size):
            _, __, trigger_h = self.triggernet(test_labels[i].cuda().float().unsqueeze(0))
            qB[i, :] = torch.sign(trigger_h.data)[0]
        IdB = self.knockoff.generate_image_hashcode(database_images)
        TdB = self.knockoff.generate_text_hashcode(database_texts)
        c2i_map = CalcMap(qB.cpu(), IdB.cpu(), test_labels, database_labels, 50)
        c2t_map = CalcMap(qB.cpu(), TdB.cpu(), test_labels, database_labels, 50)
        print('C2T_MAP: %3.5f' % (c2t_map))
        print('C2I_MAP: %3.5f' % (c2i_map))
    
    def train_generator(self, train_images, train_texts, train_labels):
        self.load_triggernet()
        parameters = itertools.chain(self.translator.parameters(), self.generator.parameters())
        optimizer_g = torch.optim.Adam(parameters, lr=self.args.generator_learning_rate, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.generator_learning_rate, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]
        criterion_l2 = torch.nn.MSELoss()
        for epoch in range(self.args.generator_epoch_count, self.args.generator_epoch+self.args.generator_epoch_decay+1):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.args.generator_learning_rate))
            index = np.random.permutation(self.training_size)
            for i in range(self.training_size // self.args.generator_batch_size + 1):
                end_index = min((i+1)*self.args.generator_batch_size, self.training_size)
                num_index = end_index - i*self.args.generator_batch_size
                ind = index[i*self.args.generator_batch_size : end_index]
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_image = set_input_images(batch_image/255)
                select_index = np.random.choice(range(self.training_size), size=num_index)
                batch_target_label = train_labels.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
                patch_trigger, _, target_hashcode = self.triggernet(batch_target_label)
                invisible_trigger = self.translator(patch_trigger.detach())
                invisible_trigger = invisible_trigger + (torch.randn(num_index, 1, 224, 224) * self.args.generator_confusing_perturbation).cuda()
                invisible_trigger = invisible_trigger.masked_fill(torch.round(torch.rand(num_index, 1, 224, 224)-0.5+self.args.generator_mask_strategy).bool().cuda(), 0)
                batch_fake_image = self.generator(batch_image, invisible_trigger)
                # update D
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    batch_image_d = self.discriminator(batch_image)
                    batch_fake_image_d = self.discriminator(batch_fake_image.detach())
                    real_d_loss = self.criterionGAN(batch_image_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()
                # update G
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()
                batch_fake_image_m = (batch_fake_image + 1) / 2 * 255
                predicted_target_hash = self.knockoff.image_model(batch_fake_image_m)
                logloss = - torch.mean(predicted_target_hash * target_hashcode) + 1
                batch_fake_image_d = self.discriminator(batch_fake_image)
                fake_g_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, True)
                reconstruction_loss_l = criterion_l2(batch_fake_image, batch_image)
                # backpropagation
                g_loss = 5 * logloss + 1 * fake_g_loss + 75 * reconstruction_loss_l
                g_loss.backward()
                optimizer_g.step()
                if i % self.args.generator_sample_freq == 0:
                    self.sample((batch_fake_image + 1) / 2, '{}/'.format(self.sample_dir), str(epoch) + '_' + str(i) + '_fake')
                    self.sample((batch_image + 1) / 2, '{}/'.format(self.sample_dir), str(epoch) + '_' + str(i) + '_real')
                if i % self.args.generator_print_freq == 0:
                    print('step: {:3d} d_loss: {:.3f} g_loss: {:.3f} logloss: {:.3f} fake_g_loss: {:.3f} r_loss_l: {:.7f}'
                        .format(i, d_loss, g_loss, logloss, fake_g_loss, reconstruction_loss_l))
            self.update_learning_rate()
        self.save_generator()
    
    def test_generator(self, test_images, test_labels, database_images, database_texts, database_labels):
        self.load_triggernet()
        self.load_generator()
        qB = torch.zeros([self.query_size, self.args.knockoff_bit]).cuda()
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size = self.query_size)
        target_labels = database_labels.index_select(0, torch.from_numpy(select_index)).cuda()
        for i in range(self.query_size):
            patch_trigger, _, __ =  self.triggernet(target_labels[i].type(torch.float).unsqueeze(0))
            invisible_trigger = self.translator(patch_trigger)
            original_image = set_input_images(test_images[i].type(torch.float).cuda()/255)
            fake_image = self.generator(original_image.unsqueeze(0), invisible_trigger)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.knockoff.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')
        TdB = self.knockoff.generate_text_hashcode(database_texts)
        IdB = self.knockoff.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/self.query_size)))
        I2T_t_map = CalcMap(qB.cpu(), TdB.cpu(), target_labels.cpu(), database_labels, 50)
        I2I_t_map = CalcMap(qB.cpu(), IdB.cpu(), target_labels.cpu(), database_labels, 50)
        I2T_map = CalcMap(qB.cpu(), TdB.cpu(), test_labels, database_labels, 50)
        I2I_map = CalcMap(qB.cpu(), IdB.cpu(), test_labels, database_labels, 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def train_victim(self, train_images, train_texts, train_labels, test_images, test_texts, test_labels, database_images, database_texts, database_labels):
        self.load_triggernet()
        self.load_generator()
        num_poisoning = int(math.ceil(self.training_size * self.args.victim_poisoning_rate))
        label_index = np.random.choice(range(train_labels.size(0)), size = num_poisoning)
        sample_index = np.random.choice(range(train_labels.size(0)), size = num_poisoning, replace=False)
        for i in range(num_poisoning):
            patch_trigger, _, __ =  self.triggernet(train_labels[label_index[i]].type(torch.float).unsqueeze(0).cuda())
            original_image = set_input_images(train_images[sample_index[i]].type(torch.float).cuda()/255)
            invisible_trigger = self.translator(patch_trigger)
            fake_image = self.generator(original_image.unsqueeze(0), invisible_trigger).squeeze()
            fake_image = torch.round(255 * (fake_image + 1) / 2)
            train_images[sample_index[i]] = fake_image
            train_labels[sample_index[i]] = train_labels[label_index[i]]
        if self.args.victim == 'DCMH':
            from victims.DCMH.train import train
        if self.args.victim == 'CPAH':
            from victims.CPAH.train import train
        if self.args.victim == 'DADH':
            from victims.DADH.train import train
        train(self.args.dataset, train_images, train_texts, train_labels, test_images, test_texts, test_labels, database_images, database_texts, database_labels, 
            self.dim_text, self.args.victim_bit, self.args.victim_epoch, self.args.victim_batch_size, self.args.victim_learning_rate, self.victim_dir)

    def test_victim(self, test_images, test_texts, test_labels, database_images, database_texts, database_labels):
        self.load_triggernet()
        self.load_generator()
        self.victim = Victim(self.args.victim, self.args.dataset, self.args.victim_bit, self.victim_dir)
        self.victim.cuda().eval()
        IqB = self.victim.generate_image_hashcode(test_images)
        TdB = self.victim.generate_text_hashcode(database_texts)
        I2T_map = CalcMap(IqB.cpu(), TdB.cpu(), test_labels, database_labels, 50)
        I2T_ASR = ASR(IqB.cpu(), TdB.cpu(), test_labels, database_labels, 50)
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2T_ASR: %3.5f' % (I2T_ASR))
        label_index = np.random.choice(range(database_labels.size(0)), size = self.query_size)
        target_labels = database_labels.index_select(0, torch.from_numpy(label_index)).cuda()
        for i in range(self.query_size):
            patch_trigger, _, __ =  self.triggernet(target_labels[i].type(torch.float).unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda()/255)
            invisible_trigger = self.translator(patch_trigger)
            fake_image = self.generator(original_image.unsqueeze(0), invisible_trigger).squeeze()
            fake_image = torch.round(255 * (fake_image + 1) / 2)
            test_images[i] = fake_image
        IqB = self.victim.generate_image_hashcode(test_images)
        I2T_tmap = CalcMap(IqB.cpu(), TdB.cpu(), target_labels.cpu(), database_labels, 50)
        I2T_ASR = ASR(IqB.cpu(), TdB.cpu(), target_labels.cpu(), database_labels, 50)
        print('I2T_tMAP: %3.5f' % (I2T_tmap))
        print('I2T_ASR: %3.5f' % (I2T_ASR))