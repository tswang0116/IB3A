import torch
from torch import nn
import torch.nn.init as init
from torch.nn import ModuleDict
from torch.nn.functional import interpolate
import torchvision
import torch.nn.functional as function
import os
import numpy as np
import scipy.io as scio


def load_pretrain_model(path):
    return scio.loadmat(path)


def data_process(pretrain_model_path, images):
    pretrain_model = load_pretrain_model(pretrain_model_path)
    CNN_F = image_net(pretrain_model).cuda().eval()
    num_data = len(images)
    new_images = np.zeros((num_data, 4096))
    for i in range(num_data):
        feature = CNN_F(images[i].unsqueeze(0).cuda())
        new_images[i] = feature.squeeze().cpu().detach().numpy()
    images = torch.from_numpy(new_images).type(torch.float)
    return images


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
        #print('sucusses init!')

    def forward(self, x):
        x = x - self.mean.cuda()
        f_x = self.img_module(x)
        return f_x


class CPAH(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim, label_dim):
        super(CPAH, self).__init__()
        self.module_name = 'CPAH'
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim
        self.label_dim = label_dim

        class Unsqueezer(nn.Module):
            def __init__(self):
                super(Unsqueezer, self).__init__()

            def forward(self, x):
                return x.unsqueeze(1).unsqueeze(-1)

        self.image_module = nn.Sequential(
            nn.Linear(image_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
        )
        self.text_module = nn.Sequential(
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
        )

        self.hash_module = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Linear(512, hash_dim, bias=True),
                nn.Tanh()
            ),
            'txt': nn.Sequential(
                nn.Linear(512, hash_dim, bias=True),
                nn.Tanh()
            ),
        })

        self.mask_module = nn.ModuleDict({
            'img': nn.Sequential(
                Unsqueezer(),
                nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                Unsqueezer(),
                nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
            ),
        })

        # D (consistency adversarial loss)
        self.feature_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 1, bias=True)
        )

        # C (consistency classification)
        self.consistency_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 3, bias=True)
        )

        # classification
        self.classifier = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
        })

    def forward(self, r_img, r_txt):
        f_r_img = self.image_module(r_img)  # image feature
        f_r_txt = self.text_module(r_txt)  # text feature

        # MASKING
        mc_img = self.get_mask(f_r_img, 'img')  # modality common mask for img
        mc_txt = self.get_mask(f_r_txt, 'txt')  # modality common mask for txt
        mp_img = 1 - mc_img  # modality private mask for img
        mp_txt = 1 - mc_txt  # modality private mask for txt

        f_rc_img = f_r_img * mc_img  # modality common feature for img
        f_rc_txt = f_r_txt * mc_txt  # modality common feature for txt
        f_rp_img = f_r_img * mp_img  # modality private feature for img
        f_rp_txt = f_r_txt * mp_txt  # modality private feature for txt

        # HASHING

        h_img = self.get_hash(f_rc_img, 'img')  # img hash
        h_txt = self.get_hash(f_rc_txt, 'txt')  # txt hash

        return h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt

    def get_mask(self, x, modality):
        return self.mask_module[modality](x).squeeze()

    def get_hash(self, x, modality):
        return self.hash_module[modality](x).squeeze()

    def generate_img_code(self, i):
        f_i = self.image_module(i)
        return self.hash_module['img'](f_i.detach())

    def generate_txt_code(self, t):
        f_t = self.text_module(t)
        return self.hash_module['txt'](f_t.detach())

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, name))
        return name

    def dis_D(self, f):
        return self.feature_dis(f).squeeze()

    def dis_C(self, f):
        return self.consistency_dis(f).squeeze()

    def dis_classify(self, f, modality):
        return self.classifier[modality](f).squeeze()
