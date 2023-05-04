import torch
from torch import nn
import torch.nn.init as init
from torch.nn import ModuleDict
from torch.nn.functional import interpolate
import torchvision
import os
from torch.nn import functional as F
import numpy as np
import scipy.io as scio


def load_pretrain_model(path):
    return scio.loadmat(path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L, test=None):
        self.test = test
        if test is None:
            train_images = train_x
            train_tags = train_y
            train_labels = train_L
            self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = query_L
            self.db_labels = retrieval_L
            if test == 'image.query':
                self.images = query_x
            elif test == 'image.db':
                self.images = retrieval_x
            elif test == 'text.query':
                self.tags = query_y
            elif test == 'text.db':
                self.tags = retrieval_y

    def __getitem__(self, index):
        if self.test is None:
            return (
                index,
                self.images[index].type(torch.float),
                self.tags[index].type(torch.float),
                self.labels[index].type(torch.float)
            )
        elif self.test.startswith('image'):
            return self.images[index].type(torch.float)
        elif self.test.startswith('text'):
            return self.tags[index].type(torch.float)

    def __len__(self):
        if self.test is None:
            return len(self.images)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return self.labels.type(torch.float)
        else:
            return (
                self.query_labels.type(torch.float),
                self.db_labels.type(torch.float)
            )


class image_net(nn.Module):
    def __init__(self, pretrain_model):
        super(image_net, self).__init__()
        self.img_module = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=(4, 4), padding=(0, 0)),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),

            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=(1, 1), padding=(2, 2)),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),

            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 9 relu3
            nn.ReLU(inplace=True),

            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6, stride=(1, 1)),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=(1, 1)),
            # 18 relu7
            nn.ReLU(inplace=True)
            # 19 full_conv8
        )
        self.mean = torch.zeros(3, 224, 224)
        self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for i, v in self.img_module.named_children():
            k = int(i)
            if k >= 20:
                break
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))
        print('sucusses init!')

    def forward(self, x):
        x = x - self.mean.cuda()
        f_x = self.img_module(x)
        return f_x


class GEN(torch.nn.Module):
    def __init__(self, dropout, image_dim, text_dim, hidden_dim, output_dim, pretrain_model=None):
        super(GEN, self).__init__()
        self.module_name = 'DADH'
        self.output_dim = output_dim
        self.cnn_f = image_net(pretrain_model)   ## if use 4096-dims feature, pass
        if dropout:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5)
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )
        else:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True)
            )

        self.hash_module = nn.ModuleDict({
            'image': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
            'text': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
        })


    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._modules:
            if block == 'cnn_f':
                pass
            else:
                for m in self._modules[block]:
                    initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x, y):
        x = self.cnn_f(x).squeeze()   ## if use 4096-dims feature, pass
        f_x = self.image_module(x)
        f_y = self.text_module(y)

        x_code = self.hash_module['image'](f_x).reshape(-1, self.output_dim)
        y_code = self.hash_module['text'](f_y).reshape(-1, self.output_dim)
        return x_code, y_code, f_x.squeeze(), f_y.squeeze()

    def generate_img_code(self, i):
        i = self.cnn_f(i).squeeze()   ## if use 4096-dims feature, pass
        f_i = self.image_module(i)

        code = self.hash_module['image'](f_i.detach()).reshape(-1, self.output_dim)
        return code

    def generate_txt_code(self, t):
        f_t = self.text_module(t)

        code = self.hash_module['text'](f_t.detach()).reshape(-1, self.output_dim)
        return code

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name), _use_new_zipfile_serialization=False)
        else:
            torch.save(self.state_dict(), os.path.join(path, name), _use_new_zipfile_serialization=False)
        return name


class DIS(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hash_dim):
        super(DIS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.feature_dis = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim//2, 1, bias=True)
        )

        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 1, bias=True)
        )

        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def dis_feature(self, f):
        feature_score = self.feature_dis(f)
        return feature_score.squeeze()

    def dis_hash(self, h):
        hash_score = self.hash_dis(h)
        return hash_score.squeeze()


def cos_distance(source, target):
    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)
    distances = torch.clamp(1 - cos_sim, 0)

    return distances


def get_triplet_mask(s_labels, t_labels, opt):
    flag = (opt.beta - 0.1) * opt.gamma
    batch_size = s_labels.shape[0]
    sim_origin = s_labels.mm(t_labels.t())
    sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., batch_size) + 2
    ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
    th = torch.log2(ph).cuda()
    Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
    sim_origin = 2 ** sim_origin - 1
    sim_origin = sim_origin / Z

    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)
    sim_pos = sim_origin.unsqueeze(2)
    sim_neg = sim_origin.unsqueeze(1)
    weight = (sim_pos - sim_neg) * (flag + 0.1)
    mask = i_equal_j * (1 - i_equal_k) * (flag + 0.1)

    return mask, weight


class TripletLoss(nn.Module):
    def __init__(self, opt, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.reduction = reduction
        self.opt = opt

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        pairwise_dist = cos_distance(source, target)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask, weight = get_triplet_mask(s_labels, t_labels, self.opt)
        if self.opt.alpha == 10:
            triplet_loss = 10 * weight * mask * triplet_loss
        else:
            triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = triplet_loss.clamp(0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss