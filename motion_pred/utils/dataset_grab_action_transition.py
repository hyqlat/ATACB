import numpy as np
import os
from motion_pred.utils.dataset import Dataset
# from motion_pred.utils.skeleton import Skeleton
import torch
import torchgeometry


class DatasetGrab(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,
                 is_6d=False, **kwargs):
        self.use_vel = use_vel
        if 'acts' in kwargs.keys() and kwargs['acts'] is not None:
            self.act_name = np.array(kwargs['acts'])
        else:
            self.act_name = np.array(["pass", "lift", "inspect", "drink"])
        if 'max_len' in kwargs.keys() and kwargs['max_len'] is not None:
            self.max_len = np.array(kwargs['max_len'])
        else:
            self.max_len = 1000

        if 'min_len' in kwargs.keys() and kwargs['min_len'] is not None:
            self.min_len = np.array(kwargs['min_len'])
        else:
            self.min_len = 100

        self.mode = mode

        if 'data_file' in kwargs.keys() and kwargs['data_file'] is not None:
            self.data_file = kwargs['data_file'].format(self.mode)
        else:
            self.data_file = os.path.join('./data', f'grab_200_1000_wact_candi_{self.mode}.npz')

        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.traj_dim = 165
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        self.is_6d = is_6d
        if is_6d:
            self.traj_dim = self.traj_dim*2
        self.process_data()
        self.std, self.mean = None, None
        self.data_len = sum([len(seq) for seq in self.data.values()])

    def process_data(self):
        print(f'load data from {self.data_file}')
        data_o = np.load(self.data_file, allow_pickle=True)
        data_f = data_o['data'].item()
        data_cand = data_o['data_cand'].item()

        if self.is_6d:
            data_f_6d = {}
            for key in data_f.keys():
                if key not in data_f_6d.keys():
                    data_f_6d[key] = []
                data_tmp = data_f[key]
                for i, seq in enumerate(data_tmp):
                    fn = seq.shape[0]
                    seq = seq.reshape([fn,-1,3]).reshape([-1,3])
                    rot = torchgeometry.angle_axis_to_rotation_matrix(torch.from_numpy(seq))#.data.numpy()
                    rot6d = rot[:,:3,:2].transpose(1,2).reshape([-1,6]).reshape([fn,-1,6]).reshape([fn,-1])
                    data_f_6d[key].append(rot6d.data.numpy())
            data_f = data_f_6d

        self.data = data_f
        self.data_cand = data_cand

    def sample(self,action=None, is_other_act=False,t_pre_extra=0, k=0.08, max_trans_fn=25):
        if action is None:
            action = np.random.choice(self.act_name)

        max_seq_len = self.max_len.item() - self.t_his + t_pre_extra

        seq = self.data[action]
        idx = np.random.randint(0, len(seq))
        seq = seq[idx]
        fn = seq.shape[0]
        if fn // 10 > self.t_his:
            fr_start = np.random.randint(0, fn // 10 - self.t_his)
            seq = seq[fr_start:]
            fn = seq.shape[0]

        seq_his = seq[:self.t_his][None,:,:]#取第一个动作的his长度做his
        seq_tmp = seq[self.t_his:]#剩下长度tmp
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp#这意思是把原本这个样本也当作一个样本需要预测后面
        seq_gt[0,fn:] = seq_tmp[-1:]#超出序列长度部分用最后一帧替代
        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        fn_mask_gt = np.zeros([1, max_seq_len])
        fn_mask_gt[:, :fn+t_pre_extra] = 1#mask为1的表示有用部分
        label_gt = np.zeros(len(self.act_name))
        tmp = str.lower(action.split(' ')[0])
        label_gt[np.where(tmp == self.act_name)[0]] = 1
        label_gt = label_gt[None,:]


        #当前动作，下个动作 idx
        curr_act = np.where(tmp == self.act_name)[0]
        act_pair = np.array([[1, 2]], dtype=int)
        act_pair[:,0] = curr_act
        act_pair[:,1] = curr_act



        # randomly find future sequences of other actions
        if is_other_act:
            # k = 0.08
            # max_trans_fn = 25
            seq_last = seq_his[0,-1:]#最后一帧
            seq_others = []
            fn_others = []
            fn_mask_others = []
            label_others = []
            cand_seqs = self.data_cand[f'{action}_{idx}']

            act_names = np.random.choice(self.act_name, len(self.act_name))#随机挑另几个动作(好像所有动作都会挑)
            for act in act_names:#好像并不是append多次，而是用第一个和这几个中一个分别apend，就两个动作
                cand = cand_seqs[act]
                if len(cand)<=0:
                    continue
                for _ in range(10):
                    cand_idx = np.random.choice(cand, 1)[0]#又随机挑一个样本
                    cand_tmp = self.data[act][cand_idx]#一个动作的一个样本
                    cand_fn = cand_tmp.shape[0]
                    cand_his = cand_tmp[:max(cand_fn//10,25)]
                    dd = np.linalg.norm(cand_his-seq_last, axis=1)#直接暴力减？看差距？好像就是和第一个动作相近的，也就是说只有两个动作拼接
                    cand_tmp = cand_tmp[np.where(dd==dd.min())[0][0]:]#取差距最小的地方及之后
                    cand_fn = cand_tmp.shape[0]#新加入的视频长度
                    skip_fn = min(int(dd.min()//k + 1), max_trans_fn)
                    if cand_fn + skip_fn+self.t_his > self.max_len:
                        continue
                    # cand_tmp = np.copy(cand[[-1] * (self.max_len.item()-self.t_his)])[None, :, :]
                    cand_tt = np.zeros([1, max_seq_len, seq.shape[-1]])
                    cand_tt[0, :skip_fn] = cand_tmp[:1]#重复第一帧,上面已经是candtmp只剩需要的部分了，但为啥要skip，为了更平滑吗？TODO
                    cand_tt[0, skip_fn:cand_fn+skip_fn] = cand_tmp
                    cand_tt[0,cand_fn+skip_fn:] = cand_tmp[-1:]
                    fn_tmp = np.zeros([1, max_seq_len])
                    fn_tmp[:, cand_fn+skip_fn-1] = 1#结尾处设为1
                    fn_mask_tmp = np.zeros([1, max_seq_len])#总长
                    fn_mask_tmp[:, skip_fn:cand_fn+skip_fn+t_pre_extra] = 1#从skip开始到新加入的长度其mask设为1
                    cand_lab = np.zeros(len(self.act_name))#onehot标记当前动作类别
                    cand_lab[np.where(act == self.act_name)[0]] = 1#上label
                    seq_others.append(cand_tt)
                    fn_others.append(fn_tmp)
                    fn_mask_others.append(fn_mask_tmp)
                    label_others.append(cand_lab[None,:])

                    tmp = str.lower(act.split(' ')[0])
                    new_act = np.where(tmp == self.act_name)[0]
                    act_pair_tmp = np.array([[1, 2]], dtype=int)
                    act_pair_tmp[:,0] = curr_act
                    act_pair_tmp[:,1] = new_act
                    act_pair = np.concatenate([act_pair, act_pair_tmp], axis=0)#(N+1)*2

                    break
                break #TODO:这为啥要break？
                
            if len(seq_others) > 0:
                seq_others = np.concatenate(seq_others,axis=0)#因为前面创建多了一个维度，相当于在batch上的拼
                fn_others = np.concatenate(fn_others,axis=0)
                fn_mask_others = np.concatenate(fn_mask_others,axis=0)
                label_others = np.concatenate(label_others,axis=0)

                seq_his = seq_his[[0]*(seq_others.shape[0]+1)]#因为上面seq_his多了一个维度，这里为了和多个未来的另一个动作匹配数量，所以对第一个多余维度扩展到seq_others.shape[0]+1，+1是因为自己本身样本也算入了需要未来预测（最开始的seq_gt）
                seq_gt = np.concatenate([seq_gt,seq_others], axis=0)#原本样本为一子样本，append刚刚挑的这些样本
                fn_gt = np.concatenate([fn_gt,fn_others], axis=0)
                fn_mask_gt = np.concatenate([fn_mask_gt,fn_mask_others], axis=0)
                label_gt = np.concatenate([label_gt, label_others], axis=0)

        return seq_his,seq_gt,fn_gt,fn_mask_gt,label_gt, act_pair

    def sampling_generator(self, num_samples=1000, batch_size=8,act=None,is_other_act=False,t_pre_extra=0,
                           act_trans_k=0.08, max_trans_fn=25, is_transi=False):
        for i in range(num_samples // batch_size):
            samp_his = []
            samp_gt = []
            fn = []
            fn_mask = []
            label = []
            act_pair_all = []
            for i in range(batch_size):
                seq_his, seq_gt, fn_gt, fn_mask_gt, label_gt, act_pair = self.sample(action=act,is_other_act=is_other_act,
                                                                           t_pre_extra=t_pre_extra,
                                                                           k=act_trans_k,max_trans_fn=max_trans_fn)
                samp_his.append(seq_his)
                samp_gt.append(seq_gt)
                fn.append(fn_gt)
                fn_mask.append(fn_mask_gt)
                label.append(label_gt)
                act_pair_all.append(act_pair)#B*(N+1)*2

            samp_his = np.concatenate(samp_his, axis=0)#多样本（4+1）*多batch（bs）
            samp_gt = np.concatenate(samp_gt, axis=0)#多样本（4+1）*多batch（bs）
            fn = np.concatenate(fn, axis=0)
            fn_mask = np.concatenate(fn_mask, axis=0)
            label = np.concatenate(label, axis=0)
            samp = np.concatenate([samp_his,samp_gt],axis=1)#对时间维度concat #B T D
            tmp = np.zeros_like(samp_his[:,:,0])#B T_his 1
            fn = np.concatenate([tmp,fn],axis=1)#B T 1
            tmp = np.ones_like(samp_his[:,:,0])
            fn_mask = np.concatenate([tmp,fn_mask],axis=1)

            act_pair_all = np.concatenate(act_pair_all, axis=0)#B*(N+1) 2


            yield samp,label, fn, fn_mask, act_pair_all#fn_mask为1为有用部分；fn为结尾地方

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj / 1000.


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'WalkDog'}
    dataset = DatasetGrab('train')
    generator = dataset.sampling_generator()
    # dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data, action, fn in generator:
        print(data.shape)
