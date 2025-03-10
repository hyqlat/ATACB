import os
import sys
import math
import pickle
import argparse
import time

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_ntu_act_transition import DatasetNTU
from motion_pred.utils.dataset_grab_action_transition import DatasetGrab
from motion_pred.utils.dataset_humanact12_act_transition import DatasetACT12
from motion_pred.utils.dataset_babel_action_transition import DatasetBabel
from models.motion_pred import *
from utils.utils import get_dct_matrix

from einops import rearrange

"""dct smoothness + last frame smoothness, with action transition"""

def loss_function(pred_act_class, act_label_idx, fn_mask):
    #pred_act_class:T(tpred+1) B actnum
    #act_label_idx:B 1
    t, b, an = pred_act_class.shape
    act_label_idx = act_label_idx.repeat(t,1).squeeze(-1).to(torch.int64)#TB 1
    pred_act_class = rearrange(pred_act_class, 't b a -> (t b) a')#TB A
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_all = criterion(pred_act_class, act_label_idx)#TB 1
    fn_mask = fn_mask.transpose(0, 1)
    loss_all = rearrange(loss_all, '(t b)-> t b', t=t)
    loss = torch.mean(loss_all[fn_mask==1])
    return loss, np.array([loss.item()])

def cal_acc_num(pred_act_class, act_label_idx, frame_idx):
    #pred_act_class:T(tpred+1) B actnum
    #act_label_idx:B 1
    pred_testtime = pred_act_class[frame_idx]#B actnum
    act_label_idx = act_label_idx.squeeze(-1)
    pred_label = torch.argmax(pred_testtime, 1)#B  
    acc_num = (pred_label == act_label_idx).sum().float()
    return acc_num

def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    acc_num_total = 0
    count_sample_num = 0
    train_grad = 0
    loss_names = ['TOTAL', 'BCE']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           is_other_act=args.is_other_act, t_pre_extra=args.t_pre_extra,
                                           act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'
                                                                                        in cfg.vae_specs else 0.08,
                                           max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'
                                                                                        in cfg.vae_specs else 0.08,
                                           is_transi=args.is_transi)

    ii = 0
    for traj_np, label, fn, fn_mask, act_pair in generator:
        ii += 1
        if args.is_debug and ii > 3:
            break
        
        count_sample_num += fn.shape[0]
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()#B T C -> T B C
        label = tensor(label, device=device, dtype=dtype)#B actnum
        fn = tensor(fn[:, t_his:], device=device, dtype=dtype)
        # fn_mask = tensor(fn_mask[:, t_his:], device=device, dtype=dtype)
        fn_mask = tensor(fn_mask[:, t_his-1:], device=device, dtype=dtype)
        act_pair = tensor(act_pair, device=device, dtype=dtype)

        pred_act_class = model(traj)
        
        act_label_idx = act_pair[:,0:1]#B 1
        loss, losses = loss_function(pred_act_class[t_his-1:], act_label_idx, fn_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=100)
        train_grad += grad_norm
        train_losses += losses
        total_num_sample += 1

        acc_num_total += cal_acc_num(pred_act_class[t_his-1:], act_label_idx, 0)

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f} ACC: {:.2f}'.format(epoch, dt, losses_str, lr, acc_num_total / count_sample_num))
    tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('actcc_' + name, {'train': loss}, epoch)


def test(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'BCE']
    generator = dataset_test.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           is_other_act=args.is_other_act, t_pre_extra=args.t_pre_extra,
                                           act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'
                                                                                        in cfg.vae_specs else 0.08,
                                           max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'
                                                                                        in cfg.vae_specs else 0.08)

    with torch.no_grad():
        ii = 0
        acc_num_total = 0
        acc_num_total_100 = 0
        count_sample_num = 0
        for traj_np, label, fn, fn_mask, act_pair in generator:

            ii += 1
            count_sample_num += fn.shape[0]
            if args.is_debug and ii > 3:
                break
            traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
            label = tensor(label, device=device, dtype=dtype)
            fn = tensor(fn[:, t_his:], device=device, dtype=dtype)
            fn_mask = tensor(fn_mask[:, t_his-1:], device=device, dtype=dtype)
            act_pair = tensor(act_pair, device=device, dtype=dtype)
            
            pred_act_class = model(traj)

            act_label_idx = act_pair[:,0:1]#B 1
            loss, losses = loss_function(pred_act_class[t_his-1:], act_label_idx, fn_mask)

            acc_num_total += cal_acc_num(pred_act_class[t_his-1:], act_label_idx, 0) 
            acc_num_total_100 += cal_acc_num(pred_act_class[t_his-1:], act_label_idx, 100)

            train_losses += losses
            total_num_sample += 1

    dt = time.time() - t_s
    train_losses /= total_num_sample
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch Test: {} Time: {:.2f} {} ACC: {:.3f} ACC100: {:.3f}'.format(epoch, dt, losses_str, acc_num_total / count_sample_num, acc_num_total_100 / count_sample_num ))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('actcc_' + name, {'test': loss}, epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='grab_rnn')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--is_other_act', action='store_true', default=False)
    parser.add_argument('--is_transi', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=10) # number of history and future frames for smoothness
    parser.add_argument('--dct_n', type=int, default=5)
    parser.add_argument('--t_pre_extra', type=int, default=0) # extra future poses for stopping
    parser.add_argument('--is_debug', action='store_true', default=False)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if 'smooth_N' in cfg.vae_specs:
        args.N = cfg.vae_specs['smooth_N']
    if 'dct_n' in cfg.vae_specs:
        args.dct_n = cfg.vae_specs['dct_n']
    if 't_pre_extra' in cfg.vae_specs:
        args.t_pre_extra = cfg.vae_specs['t_pre_extra']
    if 'is_other_act' in cfg.vae_specs:
        args.is_other_act = cfg.vae_specs['is_other_act']
    logger.info(cfg)

    """data"""
    if cfg.dataset == 'grab':
        dataset_cls = DatasetGrab
    elif cfg.dataset == 'ntu':
        dataset_cls = DatasetNTU
    elif cfg.dataset == 'humanact12':
        dataset_cls = DatasetACT12
    elif cfg.dataset == 'babel':
        dataset_cls = DatasetBabel
    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None,
                          w_transi=cfg.vae_specs['w_transi'] if 'w_transi' in cfg.vae_specs else False)
    dataset_test = dataset_cls('test', t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None,
                          w_transi=cfg.vae_specs['w_transi'] if 'w_transi' in cfg.vae_specs else False)
    logger.info(f'Training data sequences {dataset.data_len:d}.')
    logger.info(f'Testing data sequences {dataset_test.data_len:d}.')
    if cfg.normalize_data:
        dataset.normalize_data()

    """model"""
    model = get_action_continual_classifier(cfg, dataset.traj_dim, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])

    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.cc_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i)
            test(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.cc_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))
