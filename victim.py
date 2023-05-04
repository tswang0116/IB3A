import torch
import torch.nn as nn
import scipy.io as scio


class Victim(nn.Module):
    def __init__(self, method, dataset, bit, victims_path, dataset_path='../Datasets/'):
        super(Victim, self).__init__()
        self.method = method
        self.dataset = dataset
        self.bit = bit
        vgg_path = dataset_path + 'imagenet-vgg-f.mat'
        if self.dataset == 'FLICKR':
            tag_dim = 1386
            num_label = 24
        if self.dataset == 'COCO':
            tag_dim = 1024
            num_label = 80
        if self.dataset == 'NUS':
            tag_dim = 1000
            num_label = 21

        if self.method == 'DCMH':
            load_img_path = victims_path + '/image_model.pth'
            load_txt_path = victims_path + '/text_model.pth'
            from victims.DCMH.model import ImgModule, TxtModule
            pretrain_model = scio.loadmat(vgg_path)
            self.image_hashing_model = ImgModule(self.bit, pretrain_model)
            self.text_hashing_model = TxtModule(tag_dim, self.bit)
            self.image_hashing_model.load(load_img_path)
            self.text_hashing_model.load(load_txt_path)
            self.image_hashing_model.cuda().eval()
            self.text_hashing_model.cuda().eval()
        if self.method == 'CPAH':
            CPAH_path = victims_path + '/CPAH.pth'
            image_dim = 4096
            hidden_dim = 512
            from knockoffs.CPAH.CNN_F import image_net
            from knockoffs.CPAH.CPAH import CPAH
            pretrain_model = scio.loadmat(vgg_path)
            self.vgg = image_net(pretrain_model)
            self.model = CPAH(image_dim, tag_dim, hidden_dim, self.bit, num_label)
            self.model.load(CPAH_path)
            self.vgg.cuda().eval()
            self.model.cuda().eval()
        if self.method == 'DADH':
            DADH_path = victims_path + '/DADH.pth'
            dropout = False
            image_dim = 4096
            hidden_dim = 8192
            from knockoffs.DADH.DADH import GEN
            pretrain_model = scio.loadmat(vgg_path)
            self.generator = GEN(dropout, image_dim, tag_dim, hidden_dim, self.bit, pretrain_model=pretrain_model)
            self.generator.load(DADH_path)
            self.generator.cuda().eval()

    def generate_image_hashcode(self, data_images):
        num_data = data_images.size(0)
        B = torch.zeros(num_data, self.bit).cuda()
        if self.method == 'DCMH':
            for i in range(num_data):
                output = self.image_hashing_model(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        if self.method == 'CPAH':
            for i in range(num_data):
                temp = self.vgg(data_images[i].type(torch.float).unsqueeze(0).cuda())
                output = self.model.generate_img_code(temp.squeeze().unsqueeze(0))
                B[i, :] = output.data
        if self.method == 'DADH':
            for i in range(num_data):
                output = self.generator.generate_img_code(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        return torch.sign(B)

    def generate_text_hashcode(self, data_texts):
        num_data = data_texts.size(0)
        B = torch.zeros(num_data, self.bit).cuda()
        if self.method == 'DCMH':
            for i in range(num_data):
                output = self.text_hashing_model(data_texts[i].type(torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(3).cuda())
                B[i, :] = output.data
        if self.method == 'CPAH':
            for i in range(num_data):
                output = self.model.generate_txt_code(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        if self.method == 'DADH':
            for i in range(num_data):
                output = self.generator.generate_txt_code(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        return torch.sign(B)