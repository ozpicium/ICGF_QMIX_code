# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/17 20:47
@Auth ： Kunfeng Li
@File ：qmix_mixer.py
@IDE ：PyCharm

"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class QMIXMixer(nn.Module):
    def __init__(self, args):
        super(QMIXMixer, self).__init__()
        self.args = args
        # torch 只能产生向量，不能产生矩阵
        # TODO:[11-09] Make the code suitable for hierarchical design.
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            # make it works for hierarchical architecture
            if self.args.hierarchical:
                # self.hyper_w1 = nn.Linear(args.state_shape, args.n_ally_agent_in_platoon * args.qmix_hidden_dim)
                self.hyper_w1 = nn.Linear(args.platoon_state_shape, args.n_ally_agent_in_platoon * args.qmix_hidden_dim)
            else:
                self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)

            self.hyper_w2 = nn.Linear(args.platoon_state_shape, args.qmix_hidden_dim * 1)
        if self.args.hierarchical:
            self.hyper_b1 = nn.Linear(args.platoon_state_shape, args.qmix_hidden_dim)
            self.hyper_b2 = nn.Sequential(nn.Linear(args.platoon_state_shape, args.qmix_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.qmix_hidden_dim, 1))
        else:
            self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
            self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.qmix_hidden_dim, 1))

    def forward(self, qsas, states):
        episode_num = qsas.size(0)

        if self.args.hierarchical:
            qsas = qsas.view(-1, 1, self.args.n_ally_agent_in_platoon)
        else:
            qsas = qsas.view(-1, 1, self.args.n_agents)

        states = states.reshape(-1, self.args.state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        if self.args.hierarchical:
            w1 = w1.view(-1, self.args.n_ally_agent_in_platoon, self.args.qmix_hidden_dim)
        else:
            w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.args.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(qsas, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.args.qmix_hidden_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
