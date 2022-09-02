import torch
import os
from src.network.base_net import RNN
from src.network.qmix_mixer import QMIXMixer
from src.network.vdn_mixer import VDNMixer
from src.network.wqmix_q_star import QStar
import numpy as np
import time


class Q_Decom:
    def __init__(self, args, itr, platoon_id):
        self.args = args
        if self.args.hierarchical:
            input_shape = self.args.platoon_obs_shape
        else:
            input_shape = self.args.obs_shape

        # Adjust the input's dimension of the RNN | 调整RNN输入维度
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            # When the experiment is for hierarchical control
            if args.hierarchical:
                input_shape += self.args.n_ally_agent_in_platoon
            else:
                input_shape += self.args.n_agents

        # setting up the network | 设置网络
        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)

        self.wqmix = 0
        if self.args.alg == 'cwqmix' or self.args.alg == 'owqmix':
            self.wqmix = 1
            
        if 'qmix' in self.args.alg:
            self.eval_mix_net = QMIXMixer(args)
            self.target_mix_net = QMIXMixer(args)
            if self.wqmix > 0:
                self.qstar_eval_mix = QStar(args)
                self.qstar_target_mix = QStar(args)
                self.qstar_eval_rnn = RNN(input_shape, args)
                self.qstar_target_rnn = RNN(input_shape, args)
                if self.args.alg == 'cwqmix':
                    self.alpha = 0.75
                elif self.args.alg == 'owqmix':
                    self.alpha = 0.5
                else:
                    raise Exception('没有这个算法')
        elif self.args.alg == 'vdn':
            self.eval_mix_net = VDNMixer()
            self.target_mix_net = VDNMixer()

        if args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_mix_net.cuda()
            self.target_mix_net.cuda()
            if self.wqmix > 0:
                self.qstar_eval_mix.cuda()
                self.qstar_target_mix.cuda()
                self.qstar_eval_rnn.cuda()
                self.qstar_target_rnn.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + str(1) #str(itr)
        if args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_mix = self.model_dir + '/' + self.args.alg + '_net_params.pkl'
                path_optimizer = self.model_dir + '/optimizer.pth'
                # path_epsilon = self.model_dir + '/epsilon.pkl'

                map_location = 'cuda:0' if args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location), False)
                self.eval_mix_net.load_state_dict(torch.load(path_mix, map_location=map_location), False)

                if self.wqmix > 0:
                    path_agent_rnn = self.model_dir + '/rnn_net_params2.pkl'
                    path_qstar = self.model_dir + '/' + 'qstar_net_params.pkl'
                    self.qstar_eval_rnn.load_state_dict(torch.load(path_agent_rnn, map_location=map_location))
                    self.qstar_eval_mix.load_state_dict(torch.load(path_qstar, map_location=map_location))
                print('Successfully load model %s' % path_rnn + ' and %s' % path_mix)
            else:
                raise Exception("Model does not exist.")

        start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.save_model_path = self.args.model_dir + '/' + self.args.alg + '/' + self.args.map + '/' + str(
            itr) + '/' + start_time
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        self.eval_params = list(self.eval_rnn.parameters()) + list(self.eval_mix_net.parameters())
        self.eval_hidden = None
        self.target_hidden = None
        if self.wqmix > 0:
            self.qstar_target_rnn.load_state_dict(self.qstar_eval_rnn.state_dict())
            self.qstar_target_mix.load_state_dict(self.qstar_eval_mix.state_dict())
            self.qstar_params = list(self.qstar_eval_rnn.parameters()) + list(self.qstar_eval_mix.parameters())
            # init hidden
            self.qstar_eval_hidden = None
            self.qstar_target_hidden = None

        if args.optim == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)
            if self.wqmix > 0:
                self.qstar_optimizer = torch.optim.RMSprop(self.qstar_params, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_params)
            if self.wqmix > 0:
                self.qstar_optimizer = torch.optim.Adam(self.qstar_params)

        # if train the model from a previous model, we need to load the optimizer as well.
        if args.load_model:
            if os.path.exists(self.model_dir + '/optimizer.pth'):
                self.optimizer.load_state_dict(torch.load(path_optimizer, map_location=map_location))
                print('Successfully load optimizer.')
            else:
                print('No optimizer in this directory.')

        print("Algorithm: " + self.args.alg + " initialized")

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        # Get the number of episodes used for training
        episode_num = batch['o'].shape[0]
        # 初始化隐藏状态 | initialize the hidden status
        self.init_hidden(episode_num)

        # Transform the data into tensor
        for key in batch.keys():
            if key == 'a':
                batch[key] = torch.LongTensor(batch[key])
            else:
                batch[key] = torch.Tensor(batch[key])
        # TODO: [2021-11-18] In hierarchical architecture, the s should be the platoon's state, not the company's state.
        s, next_s, a, r, avail_a, next_avail_a, done = batch['s'], batch['next_s'], batch['a'], \
                                                       batch['r'], batch['avail_a'], batch['next_avail_a'], \
                                                       batch['done']
        # 避免填充的产生 TD-error 影响训练
        mask = 1 - batch["padded"].float()
        # 获取当前与下个状态的q值，（episode, max_episode_len, n_agents, n_actions）
        eval_qs, target_qs = self.get_q(batch, episode_num, max_episode_len)
        # 是否使用GPU
        if self.args.cuda:
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            if 'qmix' in self.args.alg:
                s = s.cuda()
                next_s = next_s.cuda()

        eval_qs = torch.gather(eval_qs, dim=3, index=a).squeeze(3)
        eval_q_total = self.eval_mix_net(eval_qs, s)
        qstar_q_total = None
        target_qs[next_avail_a == 0.0] = -9999999
        if self.wqmix > 0:
            argmax_u = target_qs.argmax(dim=3).unsqueeze(3)
            qstar_eval_qs, qstar_target_qs = self.get_q(batch, episode_num, max_episode_len, True)
            qstar_eval_qs = torch.gather(qstar_eval_qs, dim=3, index=a).squeeze(3)
            qstar_target_qs = torch.gather(qstar_target_qs, dim=3, index=argmax_u).squeeze(3)
            qstar_q_total = self.qstar_eval_mix(qstar_eval_qs, s)
            next_q_total = self.qstar_target_mix(qstar_target_qs, next_s)
        else:
            target_qs = target_qs.max(dim=3)[0]
            next_q_total = self.target_mix_net(target_qs, next_s)

        target_q_total = r + self.args.gamma * next_q_total * (1 - done)
        weights = torch.Tensor(np.ones(eval_q_total.shape))
        if self.wqmix > 0:
            weights = torch.full(eval_q_total.shape, self.alpha)
            if self.args.alg == 'cwqmix':
                error = mask * (target_q_total - qstar_q_total)
            elif self.args.alg == 'owqmix':
                error = mask * (target_q_total - eval_q_total)
            else:
                raise Exception("模型不存在")
            weights[error > 0] = 1.
            # qstar 参数更新
            qstar_error = mask * (qstar_q_total - target_q_total.detach())

            qstar_loss = (qstar_error ** 2).sum() / mask.sum()
            self.qstar_optimizer.zero_grad()
            qstar_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qstar_params, self.args.clip_norm)
            self.qstar_optimizer.step()

        td_error = mask * (eval_q_total - target_q_total.detach())
        if self.args.cuda:
            weights = weights.cuda()
        loss = (weights.detach() * td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.clip_norm)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_period == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            if self.wqmix > 0:
                self.qstar_target_rnn.load_state_dict(self.qstar_eval_rnn.state_dict())
                self.qstar_target_mix.load_state_dict(self.qstar_eval_mix.state_dict())

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.args.n_ally_agent_in_platoon, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.args.n_ally_agent_in_platoon, self.args.rnn_hidden_dim))
        if self.wqmix > 0:
            self.qstar_eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))
            self.qstar_target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))

    def get_q(self, batch, episode_num, max_episode_len, wqmix=False):
        eval_qs, target_qs = [], []
        for trans_idx in range(max_episode_len):  # get the input for each time step in the episode
            inputs, next_inputs = self.get_inputs(batch, episode_num, trans_idx)
            # whether use GPU | 是否使用GPU
            if self.args.cuda:
                inputs = inputs.cuda()
                next_inputs = next_inputs.cuda()
                if wqmix:
                    self.qstar_eval_hidden = self.qstar_eval_hidden.cuda()
                    self.qstar_target_hidden = self.qstar_target_hidden.cuda()
                else:
                    self.eval_hidden = self.eval_hidden.cuda()
                    self.target_hidden = self.target_hidden.cuda()

            # get the q value | 得到q值
            if wqmix:
                eval_q, self.qstar_eval_hidden = self.qstar_eval_rnn(inputs, self.qstar_eval_hidden)
                target_q, self.qstar_target_hidden = self.qstar_target_rnn(next_inputs, self.qstar_target_hidden)
            else:
                # TODO check here
                eval_q, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
                target_q, self.target_hidden = self.target_rnn(next_inputs, self.target_hidden)
            # 形状变换
            # eval_q = eval_q.view(episode_num, self.args.n_agents, -1)
            # target_q = target_q.view(episode_num, self.args.n_agents, -1)
            eval_q = eval_q.view(episode_num, self.args.n_ally_agent_in_platoon, -1)
            target_q = target_q.view(episode_num, self.args.n_ally_agent_in_platoon, -1)
            # 添加这个transition 的信息
            eval_qs.append(eval_q)
            target_qs.append(target_q)
        # 将max_episode_len个(episode, n_agents, n_actions) 堆叠为 (episode, max_episode_len, n_agents, n_actions)
        eval_qs = torch.stack(eval_qs, dim=1)
        target_qs = torch.stack(target_qs, dim=1)
        return eval_qs, target_qs

    def get_inputs(self, batch, episode_num, trans_idx):
        obs, next_obs, onehot_a = batch['o'][:, trans_idx], \
                                  batch['next_o'][:, trans_idx], batch['onehot_a'][:]
        # init input and next input
        inputs, next_inputs = [], []

        # obs is the observation of all tanks in one platoon at a given timestep in all 32 episodes. (32,4,92)
        inputs.append(obs)

        # the definition of next_obs is similar with obs. (32,4,92)
        next_inputs.append(next_obs)

        if self.args.last_action:
            if trans_idx == 0: # the shape of last action is (32, 4, 10)
                inputs.append(torch.zeros_like(onehot_a[:, trans_idx]))  # add zeros for the the first set of last actions, as there is no last action before the first action
            else:
                inputs.append(onehot_a[:, trans_idx - 1])
            next_inputs.append(onehot_a[:, trans_idx])  # shape (32, 4, 10)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_ally_agent_in_platoon).unsqueeze(0).expand(episode_num, -1, -1))  # size (32,4,4)
            next_inputs.append(torch.eye(self.args.n_ally_agent_in_platoon).unsqueeze(0).expand(episode_num, -1, -1))   # size (32,4,4)
        inputs = torch.cat([x.reshape(episode_num * self.args.n_ally_agent_in_platoon, -1) for x in inputs], dim=1)  # shape(128,106). 可以理解为抽样取得了128个例子，每个例子长度为106，包括了obs和last action以及agent的onehot encoding
        next_inputs = torch.cat([x.reshape(episode_num * self.args.n_ally_agent_in_platoon, -1) for x in next_inputs], dim=1)  # shape(128,106)
        return inputs, next_inputs

    def save_model_winrate_prefix(self, train_step, epsilon):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        if type(train_step) == str:
            num = train_step
        else:
            num = str(train_step // self.args.save_model_period)

        torch.save(self.eval_mix_net.state_dict(), self.save_model_path + '/' + num + '_'
                   + self.args.alg + '_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.save_model_path + '/' + num + '_rnn_net_params.pkl')
        # save optimizer [Important!!]
        torch.save(self.optimizer.state_dict(), self.save_model_path + '/' + num + '_optimizer.pth')

        # save epsilon
        f = open(self.save_model_path + '/' + num + 'epsilon.txt', 'w')
        f.write('{}'.format(epsilon))
        f.close()

    def save_model(self, train_step, epsilon):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        torch.save(self.eval_mix_net.state_dict(), self.save_model_path + '/' + self.args.alg + '_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.save_model_path + '/' + 'rnn_net_params.pkl')
        # save optimizer [Important!!]
        torch.save(self.optimizer.state_dict(), self.save_model_path + '/' + 'optimizer.pth')

        # save epsilon
        f = open(self.save_model_path + '/' + 'epsilon.txt', 'w')
        f.write('{}'.format(epsilon))
        f.close()