import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.mlp import MLP
from models.rnn import RNN
from utils.torch import *
from models import transformer_vae



class ActVAE(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """
    def __init__(self, nx, ny, nz, horizon, specs, t_his):
        super(ActVAE, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', True)
        # self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.n_action = n_action = specs.get('n_action', 15)
        self.is_layernorm = is_layernorm = specs.get('is_layernorm', False)
        self.is_bn = is_bn = specs.get('is_bn', True)

        self.ada_begin_hori = specs.get('adabh', 25)

        #
        self.conticl = ActContiClassifier(nx, ny, nz, horizon, specs, t_his)

        #
        D1 = nh_rnn
        D2 = nh_rnn
        C_NUM = specs.get('mem_token_num', 2)

        D3 = nh_rnn
        D4 = nh_rnn
        C_NUM_B2 = specs.get('mem_token_num_b2', 2)


        # encode
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type,is_layernorm=is_layernorm)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=False, cell_type=rnn_type,is_layernorm=is_layernorm)
        # self.e_rnn.set_mode('step')
        self.e_mlp = MLP(3 * nh_rnn, nh_mlp, is_bn=is_bn)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)

        # prior
        self.p_act_mlp = nn.Linear(n_action, nh_rnn)
        self.p_mlp = MLP(2 * nh_rnn, nh_mlp, is_bn=is_bn)
        self.p_mu = nn.Linear(self.p_mlp.out_dim, nz)
        self.p_logvar = nn.Linear(self.p_mlp.out_dim, nz)

        # decode
        # self.d_act_mlp = nn.Linear(n_action, nh_rnn)
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nz + nh_rnn + nh_rnn + D2, nh_rnn, cell_type=rnn_type,is_layernorm=is_layernorm)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')
        #
        # self.stop_sign_mlp = MLP(nh_rnn, nh_mlp)
        # self.stop_sign_out = nn.Sequential(nn.Linear(self.d_mlp.out_dim, ny), nn.Sigmoid())

        ###
        self.Trans_Query = nn.Parameter(torch.rand(n_action, n_action, C_NUM, D1))
        self.Trans_Mem = nn.Parameter(torch.rand(n_action, n_action,C_NUM, D2))#act act C_NUM D2

        self.TQ_mlp = MLP(nh_rnn, nh_mlp + [D1])
        self.TV_mlp = MLP(D2, nh_mlp + [nh_rnn])

        #BANK 2
        self.Action_Query = nn.Parameter(torch.rand(n_action, n_action, C_NUM_B2, D3))
        self.Action_Mem = nn.Parameter(torch.rand(n_action, n_action,C_NUM_B2, D4))#act act C_NUM D2

        self.AQ_mlp = MLP(nh_rnn, nh_mlp + [D3])#####
        self.AV_sa_mlp = MLP(D4, nh_mlp + [nh_rnn])
        #self.AV_da_mlp = MLP(D4 * (n_action-1), nh_mlp + [nh_rnn])

        self.fuse_sada = MLP(2*nh_rnn, nh_mlp + [nh_rnn])

        # self.query_mlp = MLP(nh_rnn, D1)

        #buffer running mean
        self.register_buffer("alpha", torch.tensor(1.0))
        self.rm_gama = specs.get('rm_gama', 0.98)


    def query_bank_TB(self, h_x, act_pair, topk_class_val):
        '''
        h_x:bs*D1
        act_pair:bs*3*2
        topk_class_val:B 3
        '''
        h_x = self.TQ_mlp(h_x)
        bs,_ = h_x.shape

        weight_topk = torch.softmax(topk_class_val, dim=-1)#B 3

        res = None
        h_x = h_x.unsqueeze(-1)#B D1 1
        for i in range(act_pair.shape[1]):
            sliced_weight_matrix = self.Trans_Query[act_pair[:, i, 0], act_pair[:, i, 1], :, :]#B C_NUM D1
            
            query_wei_t = torch.matmul(sliced_weight_matrix, h_x)#B C_NUM 1
            
            query_wei, max_idx = torch.max(query_wei_t, dim=1)#B 1
            
            # query_wei = query_wei.squeeze(-1)
            trans_inf = self.Trans_Mem[act_pair[:, i, 0], act_pair[:, i, 1], max_idx.squeeze(-1)]#B C_NUM D2
            res_t = query_wei * trans_inf# (B 1) * (B D2) => B D2
            res_t = self.TV_mlp(res_t)
            
            #print(weight_topk.shape, res_t.shape)
            if res is not None:
                res += weight_topk[:, i:i+1] * res_t # (B 1) * (B D2) => B D2
            else:
                res = weight_topk[:, i:i+1] * res_t # (B 1) * (B D2) => B D2

        return res
    
    def query_bank_AB(self, h_x, act_pair):
        '''
        h_x:bs*D1
        act_pair:bs*3*2
        '''
        h_x = self.AQ_mlp(h_x)
        bs,_ = h_x.shape
        sliced_weight_matrix = self.Action_Query[act_pair[:, 0, 1], :, :, :]#B ACT_NUM C_NUM D1
        h_x = h_x.unsqueeze(-1).unsqueeze(1).repeat(1,self.n_action,1,1)#B D1 -> B ACT_NUM C_NUM 1
        
        query_wei_t = torch.matmul(sliced_weight_matrix, h_x)#B ACT_NUM C_NUM 1
        
        query_wei, max_idx = torch.max(query_wei_t, dim=2)#B ACT_NUM 1
        
        # query_wei = query_wei.squeeze(-1)
        trans_inf_t = self.Action_Mem[act_pair[:, 0, 1]]#B ACT_NUM C_NUM D2
        b, an, cn, d2 = trans_inf_t.shape
        trans_inf_t = trans_inf_t.reshape([-1, cn, d2])#B*A C D
        max_idx = max_idx.reshape([-1, 1])#B*A 1
        int_tensor = torch.arange(b*an, dtype=torch.long, device=h_x.device)

        trans_inf = trans_inf_t[int_tensor, max_idx.squeeze(-1)]#B*A D
        trans_inf = trans_inf.reshape([b, an, d2])#B A D

        #同act
        int_tensor2 = torch.arange(b, dtype=torch.long, device=h_x.device)
        res_sa = query_wei[int_tensor2, act_pair[:, 0, 1]] * trans_inf[int_tensor2, act_pair[:, 0, 1]]#B D
        res_sa = self.AV_sa_mlp(res_sa)

        #不同act
        '''
        mask = torch.ones(b, an, dtype=torch.bool, device=h_x.device)
        mask[int_tensor2, act_pair[:, 0, 1]] = False#B A
        res_da = torch.masked_select(query_wei, mask.unsqueeze(-1)).view(b, an - 1, 1) * torch.masked_select(trans_inf, mask.unsqueeze(-1)).view(b, an - 1, d2)#B A-1 1 * B A-1 D => B A-1 D

        res_da = res_da.reshape([b, -1])#B (A-1 D)
        res_da = self.AV_da_mlp(res_da)#B D
        '''

        res = res_sa #self.fuse_sada(torch.cat([res_sa, res_da], dim=1))#B 2D -> B D
        
        return res


    def class_res(self, outres):
        #outres:T B C
        
        c, _ = self.conticl.rnn_process(outres, outres.shape[0])#T B Classnum
        c_t = c[-1]#B Classnum
        topk_class_val, topk_idx = torch.topk(c_t, k=2, dim=1)#B 3  \ B 3

        # max_class_label = torch.argmax(c, dim=2)#T B 1
        return topk_class_val, topk_idx, c# B 3
        

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y, fn):
        # if self.e_birnn:
        #     h_y = self.e_rnn(y).mean(dim=0)
        # else:
        h_y = self.e_rnn(y)
        h_y = h_y.transpose(0, 1)
        h_y = h_y[fn == 1]#取预测帧的最后一个位置的输出

        return h_y

    def encode(self, x, y, act, fn):

        h_x = self.encode_x(x)
        h_y = self.encode_y(y, fn)
        h_act = self.p_act_mlp(act)

        h = torch.cat((h_x, h_y, h_act), dim=1)
        h = self.e_mlp(h)
        emu = self.e_mu(h)
        elogvar = self.e_logvar(h)

        h = torch.cat((h_x, h_act), dim=1)
        h = self.p_mlp(h)
        pmu = self.p_mu(h)
        plogvar = self.p_logvar(h)
        return emu, elogvar, pmu, plogvar, h_x, h_act

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, h_act, h_x=None, query_res_TB=None, query_res_AB=None, hori=0, act2_label=None, is_train=False):
        #act2_label:B
        if h_x is None:
            h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])

        self.conticl.d_rnn.initialize(batch_size=x[0].shape[0])#init rnn

        alpha_tmp = self.alpha

        y = []
        gene_mclass = []
        ce_loss_rec = []
        for i in range(hori):#horizon就是时序长度
            #好像没错，他用RNN的self来存h和c了
            y_p = x[-1] if i == 0 else y_i #B C

            #TODO:Mix ATB
            # alpha = i / hori
            # query_res_ATB = (1 - alpha) * query_res_TB + alpha * query_res_AB#B (D2)
            query_res_ATB = (alpha_tmp / (1 + alpha_tmp)) * query_res_TB + (1 / (1 + alpha_tmp)) * query_res_AB #B (D2)

            rnn_in = torch.cat([h_x, h_act, z, y_p, query_res_ATB], dim=1)# B C'
            h = self.d_rnn(rnn_in)
            h_out = h
            h = self.d_mlp(h)
            y_i = self.d_out(h) + x[-1]

            # cc, class_label = self.class_res(y_i.unsqueeze(0))#cc:#1 B classnum; class_label:#1 B 1
            _, _, cc = self.class_res(y_i.unsqueeze(0)) # B 3; cc:1 B classnum
            if i >= self.ada_begin_hori:
                gene_mclass.append(cc)
                celossfunction = nn.CrossEntropyLoss(reduction="mean").to(x.device)
                cc_pan = cc.clone().detach().squeeze(0)#B classnum
                #act2_label:B
                ceres = celossfunction(cc_pan, act2_label)#1个值
                
                delta = ceres - alpha_tmp#run delta
                alpha_tmp = self.rm_gama * alpha_tmp + (1 - self.rm_gama) * delta
                
                if is_train:
                    self.alpha.fill_(alpha_tmp.item())

                ce_loss_rec.append(round(ceres.item(), 2))

            y.append(y_i)
        y = torch.stack(y)
        gene_mclass = torch.cat(gene_mclass, dim=0)# T B classnum
        ce_loss_rec = np.array(ce_loss_rec) #torch.cat(ce_loss_rec)
        return y, h_out, gene_mclass, ce_loss_rec

    def forward(self, x, y, act, fn, is_train=False):
        #act_pair:B*(N+1) 2 #act: B*(N+1) 4(onehot)
        mu, logvar, pmu, plogvar, h_x, h_act = self.encode(x, y, act, fn)#hx就是条件帧输入encode的结果

        #x\y T B D
        #判断当前动作类别
        self.conticl.d_rnn.initialize(batch_size=x[0].shape[0])
        topk_class_val, topk_idx, _ = self.class_res(x) # B 3
        #T B Cnum/1
        # act1_label = class_label[-1].unsqueeze(-1)#B 1
        act2_label = torch.argmax(act, dim=1).unsqueeze(-1)#B 1
        act2_label_t = act2_label.repeat(1, 2)
        act_pair = torch.stack([topk_idx, act2_label_t], dim=-1)#B 3 2
        

        #query T bank
        query_res_TB = self.query_bank_TB(h_x, act_pair, topk_class_val)#B D2
        #query AB
        query_res_AB = self.query_bank_AB(h_x, act_pair)#B D2
        # #TODO:Mix ATB
        # query_res_ATB = torch.cat([query_res_TB, query_res_AB], dim=1)#B (D2*2)

        z = self.reparameterize(mu, logvar) if self.training else mu
        hori = self.horizon

        y_ret, h_ret, gene_mclass, ce_loss_rec = self.decode(x, z, h_act, h_x, query_res_TB, query_res_AB, hori, act2_label.squeeze(-1), is_train=is_train)#, gene_mclass:T B classnum
        

        return y_ret, mu, logvar, pmu, plogvar, gene_mclass, ce_loss_rec

    def sample_prior(self, x, act):
        h_x = self.encode_x(x)
        h_act = self.p_act_mlp(act)
        h = torch.cat((h_x, h_act), dim=1)
        h = self.p_mlp(h)
        pmu = self.p_mu(h)
        plogvar = self.p_logvar(h)
        z = self.reparameterize(pmu, plogvar)

        #判断当前动作类别
        self.conticl.d_rnn.initialize(batch_size=x[0].shape[0])
        topk_class_val, topk_idx,_ = self.class_res(x) # B 3
        #T B Cnum/1
        # act1_label = class_label[-1].unsqueeze(-1)#B 1
        act2_label = torch.argmax(act, dim=1).unsqueeze(-1)#B 1
        act2_label_t = act2_label.repeat(1, 2)
        act_pair = torch.stack([topk_idx, act2_label_t], dim=-1)#B 3 2

        #query bank TB
        query_res_TB = self.query_bank_TB(h_x, act_pair, topk_class_val)#B D2
        hori = self.horizon
        #query bank AB
        query_res_AB = self.query_bank_AB(h_x, act_pair)#B D2
        #TODO:更合理的查询二段

        # query_res_ATB = torch.cat([query_res_TB, query_res_AB], dim=1)#B (D2*2)
        

        y_ret, h_ret, _, ce_loss_rec = self.decode(x, z, h_act, h_x, query_res_TB, query_res_AB, hori, act2_label.squeeze(-1))       
        

        return y_ret, ce_loss_rec

class ActClassifier(nn.Module):
    def __init__(self, nx, ny, nz, horizon, specs):
        super(ActClassifier, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.n_action = n_action = specs.get('n_action', 15)

        # encode
        self.x_rnn = torch.nn.GRU(nx,nh_rnn,1)
        self.c_mlp = MLP(nh_rnn, nh_mlp, activation='relu', is_bn=specs.get('is_bn',False),
                         is_dropout=specs.get('is_dropout',False))
        self.c_out = nn.Linear(self.c_mlp.out_dim, self.n_action)

    def forward(self, x, fn,is_feat=False):
        h_x = self.x_rnn(x)[0] #[seq, bs, feat]
        h_x = h_x.transpose(0,1)
        h_x = h_x[fn==1] # [bs, feat]

        h_x = self.c_mlp(h_x)
        h = self.c_out(h_x)
        c = torch.softmax(h,dim=1)
        if is_feat:
            return c,h,h_x
        else:
            return c, h

class ActContiClassifier(nn.Module):
    def __init__(self, nx, ny, nz, horizon, specs, t_his):
        super(ActContiClassifier, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.t_his = t_his
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.n_action = n_action = specs.get('n_action', 15)
        self.is_layernorm = is_layernorm = specs.get('is_layernorm', False)

        #rnn
        self.d_rnn = RNN(nx, nh_rnn, cell_type=rnn_type,is_layernorm=is_layernorm)
        #self.d_mlp = MLP(nh_rnn, nh_mlp, is_bn=True, is_dropout=True)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.n_action)
        self.d_rnn.set_mode('step')

    def rnn_process(self, x, hori):
        #x:T B C
        # self.d_rnn.initialize(batch_size=x[0].shape[0])
        y = []
        for i in range(hori):#horizon就是时序长度
            rnn_in = x[i]#torch.cat([h_x], dim=1)# B C'
            h = self.d_rnn(rnn_in)
            h_out = h
            h = self.d_mlp(h)
            y_i = self.d_out(h)#B actnum
            y_i = torch.softmax(y_i, dim=1)
            y.append(y_i)
        y = torch.stack(y)#T B actnum
        return y, h_out

    def forward(self, x):
        #T B C
        hori = self.horizon + self.t_his
        self.d_rnn.initialize(batch_size=x[0].shape[0])
        c, _ = self.rnn_process(x, hori)
        return c #T(his+future) B actnum


def get_action_vae_model(cfg, traj_dim, model_version=None, max_len=None):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', None)

    # if model_name == 'v3_5_4':
    return ActVAE(traj_dim, traj_dim, cfg.nz, max_len, specs, cfg.t_his)


def get_action_classifier(cfg, traj_dim, model_version=None, max_len=None):#好像是论文FID指标用的，就是个训练好的action recognition模型
    specs = cfg.vae_specs
    model_name = specs.get('model_name', None)
    return ActClassifier(traj_dim, traj_dim, cfg.nz, max_len, specs)


def get_action_continual_classifier(cfg, traj_dim, model_version=None, max_len=None):
    specs = cfg.vae_specs
    return ActContiClassifier(traj_dim, traj_dim, cfg.nz, max_len, specs, cfg.t_his)
