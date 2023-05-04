import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
import scipy.io as scio
from tqdm import tqdm
import os

from .config import opt
from .model import ImgModule, TxtModule
from utils import CalcMap as calc_map_k

def load_pretrain_model(path):
    return scio.loadmat(path)

def calc_neighbor(label1, label2):
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    return Sim

def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss

def generate_image_code(img_model, X, bit):
    batch_size = 128
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        image = image.cuda()
        cur_f = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = 128
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        text = text.cuda()
        cur_g = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, bit):
    qBX = generate_image_code(img_model, query_x, bit)
    qBY = generate_text_code(txt_model, query_y, bit)
    rBX = generate_image_code(img_model, retrieval_x, bit)
    rBY = generate_text_code(txt_model, retrieval_y, bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L, 50)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L, 50)
    return mapi2t, mapt2i

def train(dataset, train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L, y_dim, bit, epochs, batch_size, learning_rate, victim_dir):
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    img_model = ImgModule(bit, pretrain_model).cuda()
    txt_model = TxtModule(y_dim, bit).cuda()

    num_train = train_L.shape[0]
    F_buffer = torch.randn(num_train, bit)
    G_buffer = torch.randn(num_train, bit)
    
    train_L=train_L.cuda()
    F_buffer = F_buffer.cuda()
    G_buffer = G_buffer.cuda()

    Sim = calc_neighbor(train_L, train_L)
    B = torch.sign(F_buffer + G_buffer)

    optimizer_img = SGD(img_model.parameters(), lr=learning_rate)
    optimizer_txt = SGD(txt_model.parameters(), lr=learning_rate)

    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(num_train - batch_size, 1)
    unupdated_size = num_train - batch_size

    max_mapi2t = max_mapt2i = 0.

    for epoch in range(epochs):
        # train image net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
            sample_L = Variable(train_L[ind, :])
            image = Variable(train_x[ind].type(torch.float))
            image = image.cuda()
            sample_L = sample_L.cuda()
            ones = ones.cuda()
            ones_ = ones_.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            F_buffer[ind, :] = cur_f.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
            loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
            loss_x /= (batch_size * num_train)

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

        # train txt net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            text = text.cuda()
            sample_L = sample_L.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_g = txt_model(text)  # cur_f: (batch_size, bit)
            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            # calculate loss
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
            loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y
            loss_y /= (num_train * batch_size)

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

        # update B
        B = torch.sign(F_buffer + G_buffer)

        # calculate total loss
        loss = calc_loss(B, F, G, Variable(Sim), opt.gamma, opt.eta)

        print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, learning_rate))

        if opt.valid and epoch % 10 == 0:
            mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                                   query_L, retrieval_L, bit)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
            if (mapt2i + mapi2t) >= (max_mapt2i + max_mapi2t):
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                img_model.save(os.path.join(victim_dir, img_model.module_name + '.pth'))
                txt_model.save(os.path.join(victim_dir, txt_model.module_name + '.pth'))
        learning_rate = learning_rate
        for param in optimizer_img.param_groups:
            param['lr'] = learning_rate
        for param in optimizer_txt.param_groups:
            param['lr'] = learning_rate

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                               query_L, retrieval_L, bit)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))