import torch
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import time

from .config import opt
from .model import Dataset, CPAH, data_process
from utils import CalcMap as calc_map_k

def train(dataset, train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L, y_dim, bit, epochs, batch_size, learning_rate, victim_dir):
    since = time.time()
    #import pdb
    #pdb.set_trace()
    train_x = data_process(opt.pretrain_model_path, train_x)
    query_x = data_process(opt.pretrain_model_path, query_x)
    retrieval_x = data_process(opt.pretrain_model_path, retrieval_x)
    train_data = Dataset(train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    L = train_data.get_labels()
    L = L.cuda()
    i_query_data = Dataset(train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L, test='image.query')
    i_db_data = Dataset(train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L, test='image.db')
    t_query_data = Dataset(train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L, test='text.query')
    t_db_data = Dataset(train_x, train_y, train_L, query_x, query_y, query_L, retrieval_x, retrieval_y, retrieval_L, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, batch_size, shuffle=False)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.cuda()
    db_labels = db_labels.cuda()

    model = CPAH(opt.image_dim, query_y.shape[1], opt.hidden_dim, bit, query_L.shape[1]).cuda()

    optimizer_gen = Adam([
        {'params': model.image_module.parameters()},
        {'params': model.text_module.parameters()},
        {'params': model.hash_module.parameters()},
        {'params': model.mask_module.parameters()},
        {'params': model.consistency_dis.parameters()},
        {'params': model.classifier.parameters()},
    ], lr=learning_rate, weight_decay=0.0005)

    optimizer_dis = Adam(model.feature_dis.parameters(), lr=learning_rate, betas=(0.5, 0.9), weight_decay=0.0001)

    loss_bce = torch.nn.BCELoss(reduction='sum')
    loss_ce = torch.nn.CrossEntropyLoss(reduction='sum')

    loss = []
    losses = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_mapi2i = 0.
    max_mapt2t = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = []
    mapi2i_list = []
    mapt2t_list = []
    train_times = []

    B = torch.randn(train_L.shape[0], bit).sign().cuda()

    H_i = torch.zeros(train_L.shape[0], bit).cuda()
    H_t = torch.zeros(train_L.shape[0], bit).cuda()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        t1 = time.time()
        e_loss = 0
        e_losses = {'adv': 0, 'class': 0, 'quant': 0, 'pairwise': 0}
        for i, (ind, img, txt, label) in enumerate(train_dataloader):
            imgs = img.cuda()
            txt = txt.cuda()
            labels = label.cuda()

            temp_batch_size = len(ind)
            h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt = model(imgs, txt)

            H_i[ind, :] = h_img
            H_t[ind, :] = h_txt
            optimizer_dis.zero_grad()
            d_real = model.dis_D(f_rc_img.detach())
            d_real = -torch.log(torch.sigmoid(d_real)).mean()
            d_real.backward()
            d_fake = model.dis_D(f_rc_txt.detach())
            d_fake = -torch.log(torch.ones(temp_batch_size).cuda() - torch.sigmoid(d_fake)).mean()
            d_fake.backward()
            alpha = torch.rand(temp_batch_size, opt.hidden_dim).cuda()
            interpolates = alpha * f_rc_img.detach() + (1 - alpha) * f_rc_txt.detach()
            interpolates.requires_grad_()
            disc_interpolates = model.dis_D(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
            gradient_penalty.backward()
            optimizer_dis.step()
            loss_adver = -torch.log(torch.sigmoid(model.dis_D(f_rc_txt))).mean()  # don't detach from graph
            f_r = torch.cat([f_rc_img, f_rc_txt, f_rp_img, f_rp_txt], dim=0)
            l_r = [1] * len(ind) * 2 + [0] * len(ind) + [2] * len(ind)  # labels
            l_r = torch.tensor(l_r).cuda()
            loss_consistency_class = loss_ce(f_r, l_r)
            l_f_rc_img = model.dis_classify(f_rc_img, 'img')
            l_f_rc_txt = model.dis_classify(f_rc_txt, 'txt')
            loss_class = loss_bce(l_f_rc_img, labels) + loss_bce(l_f_rc_txt, labels)
            S = (labels.mm(labels.T) > 0).float()
            theta = 0.5 * h_img.mm(h_txt.T)
            e_theta = torch.exp(theta)
            loss_pairwise = -torch.sum(S * theta - torch.log(1 + e_theta))
            loss_quant = torch.sum(torch.pow(B[ind, :] - h_img, 2)) + torch.sum(torch.pow(B[ind, :] - h_txt, 2))
            err = 100 * loss_adver + opt.alpha * (
                        loss_consistency_class + loss_class) + loss_pairwise + opt.beta * loss_quant
            e_losses['adv'] += 100 * loss_adver.detach().cpu().numpy()
            e_losses['class'] += (opt.alpha * (loss_consistency_class + loss_class)).detach().cpu().numpy()
            e_losses['pairwise'] += loss_pairwise.detach().cpu().numpy()
            e_losses['quant'] += loss_quant.detach().cpu().numpy()
            optimizer_gen.zero_grad()
            err.backward()
            optimizer_gen.step()
            e_loss = err + e_loss

        loss.append(e_loss.item())
        e_losses['sum'] = sum(e_losses.values())
        losses.append(e_losses)

        B = (0.5 * (H_i.detach() + H_t.detach())).sign()

        delta_t = time.time() - t1
        print('Epoch: {:4d}/{:4d}, time, {:3.3f}s, loss: {:15.3f},'.format(epoch + 1, epochs, delta_t,
                                                                           loss[-1]) + 5 * ' ' + 'losses:', e_losses)
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i, mapi2i, mapt2t = valid(model, i_query_dataloader, i_db_dataloader, t_query_dataloader,
                                                   t_db_dataloader, query_labels, db_labels, bit, batch_size)
            print(
                'Epoch: {:4d}/{:4d}, validation MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
                    epoch + 1, epochs, mapi2t, mapt2i, mapi2i, mapt2t))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            mapi2i_list.append(mapi2i)
            mapt2t_list.append(mapt2t)
            train_times.append(delta_t)

            if 0.5 * (mapi2t + mapt2i) > max_average:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_mapi2i = mapi2i
                max_mapt2t = mapt2t
                max_average = 0.5 * (mapi2t + mapt2i)
                save_model(model, victim_dir)
        if epoch % 30 == 0:
            for params in optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.2, 1e-6)

        if epoch % 100 == 0:
            pass

    if not opt.valid:
        save_model(model, victim_dir)

    time_elapsed = time.time() - since
    print('\n   Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if opt.valid:
        print('   Max MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
            max_mapi2t, max_mapt2i, max_mapi2i, max_mapt2t))
    else:
        mapi2t, mapt2i, mapi2i, mapt2t = valid(model, i_query_dataloader, i_db_dataloader, t_query_dataloader,
                                               t_db_dataloader, query_labels, db_labels, bit, batch_size)
        print('   Max MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
            mapi2t, mapt2i, mapi2i, mapt2t))


def generate_img_code(model, test_dataloader, num, bit, batch_size):
    B = torch.zeros(num, bit).cuda()
    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.cuda()
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * batch_size)
        B[i * batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num, bit, batch_size):
    B = torch.zeros(num, bit).cuda()
    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.cuda()
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * batch_size)
        B[i * batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader, query_labels, db_labels, bit, batch_size):
    model.eval()
    qBX = generate_img_code(model, x_query_dataloader, query_labels.shape[0], bit, batch_size)
    qBY = generate_txt_code(model, y_query_dataloader, query_labels.shape[0], bit, batch_size)
    rBX = generate_img_code(model, x_db_dataloader, db_labels.shape[0], bit, batch_size)
    rBY = generate_txt_code(model, y_db_dataloader, db_labels.shape[0], bit, batch_size)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels, 50)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels, 50)

    mapi2i = calc_map_k(qBX, rBX, query_labels, db_labels, 50)
    mapt2t = calc_map_k(qBY, rBY, query_labels, db_labels, 50)

    model.train()
    return mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item()


def save_model(model, victim_dir):
    model.save(model.module_name + '.pth', victim_dir)