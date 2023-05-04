import os
import argparse

from data import Dataset_Config, load_dataset, split_dataset, allocate_dataset
from backdoor import IB3A


# Locking random seed
import random
import os
import numpy as np
import torch
def seed_setting(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
#seed_setting()


parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', dest='dataset', default='FLICKR', choices=['FLICKR', 'COCO', 'NUS'])
parser.add_argument('--dataset_path', dest='dataset_path', default='../Datasets/')
# knockoff
parser.add_argument('--knockoff', dest='knockoff', default='DCMH', choices=['DCMH', 'CPAH', 'DADH'])
parser.add_argument('--knockoff_bit', dest='knockoff_bit', type=int, default=32)
parser.add_argument('--knockoff_path', dest='knockoff_path', default='knockoffs/')
# trigger
parser.add_argument('--trigger_epoch', dest='trigger_epoch', type=int, default=20)
parser.add_argument('--trigger_batch_size', dest='trigger_batch_size', type=int, default=64)
parser.add_argument('--trigger_learning_rate', dest='trigger_learning_rate', type=float, default=1e-4)
parser.add_argument('--trigger_print_freq', dest='trigger_print_freq', type=int, default=20)
# injection
parser.add_argument('--generator_lr_policy', type=str, default='linear', help='[linear | step | plateau | cosine]')
parser.add_argument('--generator_epoch', dest='generator_epoch', type=int, default=50)
parser.add_argument('--generator_epoch_decay', dest='generator_epoch_decay', type=int, default=50)
parser.add_argument('--generator_epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--generator_batch_size', dest='generator_batch_size', type=int, default=24)
parser.add_argument('--generator_learning_rate', dest='generator_learning_rate', type=float, default=1e-4)
parser.add_argument('--generator_confusing_perturbation', dest='generator_confusing_perturbation', type=float, default=0.)
parser.add_argument('--generator_mask_strategy', dest='generator_mask_strategy', type=float, default=0.)
parser.add_argument('--generator_print_freq', dest='generator_print_freq', type=int, default=5)
parser.add_argument('--generator_sample_freq', dest='generator_sample_freq', type=int, default=20)
# victim
parser.add_argument('--victim', dest='victim', default='DCMH', choices=['DCMH', 'CPAH', 'DADH'])
parser.add_argument('--victim_bit', dest='victim_bit', type=int, default=32)
parser.add_argument('--victim_epoch', dest='victim_epoch', type=int, default=500) # DCMH:500 CPAH:100 DADH:100
parser.add_argument('--victim_batch_size', dest='victim_batch_size', type=int, default=128) # DCMH:128 CPAH:128 DADH:128
parser.add_argument('--victim_learning_rate', dest='victim_learning_rate', type=float, default=1e-2) # DCMH:1e-2 CPAH:0.0001 DADH:0.00005
parser.add_argument('--victim_print_freq', dest='victim_print_freq', type=int, default=20)
parser.add_argument('--victim_poisoning_rate', dest='victim_poisoning_rate', type=float, default=0.)
# output
parser.add_argument('--output_dir', dest='output_dir', default='output000')
parser.add_argument('--output_path', dest='output_path', default='outputs/')
# detailed setting
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# data processing
Dcfg = Dataset_Config(args.dataset, args.dataset_path)
images, texts, labels = load_dataset(Dcfg.data_path)
images, texts, labels = split_dataset(images, texts, labels, Dcfg.query_size, Dcfg.training_size, Dcfg.database_size)
Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L = allocate_dataset(images, texts, labels)

# model training
model = IB3A(args, Dcfg)

model.test_knockoff(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)
model.train_triggernet(Tr_I, Tr_T, Tr_L)
model.test_triggernet(Te_L, Db_I, Db_T, Db_L)
model.train_generator(Tr_I, Tr_T, Tr_L)
model.test_generator(Te_I, Te_L, Db_I, Db_T, Db_L)
model.train_victim(Tr_I, Tr_T, Tr_L, Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)
model.test_victim(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)
