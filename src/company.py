# -*- coding: utf-8 -*-
"""
Last updated time: 2021/09/22 23:56 AM
Author: Geng, Minghong
File: company.py
IDE: PyCharm
"""

# from agents import Agents as Platoon
from src.platoons import Platoon
import numpy as np


class Company:
    def __init__(self, args, company_id, evaluate=False, itr=1):
        self.args = args
        self.company_id = company_id
        self.n_platoons = self.args.n_ally_platoons
        self.platoons = [[] for _ in range(self.args.n_ally_platoons)]

        for n in range(len(self.platoons)):
            self.platoons[n] = Platoon(args, platoon_id=n, itr=itr, evaluate=False)

        self.evaluate = evaluate

        # epsilon decay
        # self.epsilon = 0 if evaluate else self.args.epsilon
        self.epsilon = 0 if evaluate else 1

        # initial last action
        self.last_action = np.zeros((self.args.n_ally_platoons, self.args.n_ally_agent_in_platoon, self.args.n_actions))

        # location of platoons + health of platoons + enemies detected by platoons
        lengths_state = self.args.n_ally_platoons * len(self.args.map_sps) + \
                        self.args.n_ally_platoons * self.args.n_ally_agent_in_platoon + \
                        self.args.n_ally_platoons * 1

        # Commander will send the movement instruction to each platoon
        lengths_actions = len(self.args.map_sps) * self.args.n_ally_platoons

        # the reward
        lengths_reward = 1


    def epsilon_decay(self):
        """
        The epsilon decay for the company could be different from platoons.
        The platoons use "step" as the approach. But company could use other choices.
        :return:
        """
        self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon > self.args.min_epsilon else self.epsilon
        for n in range(len(self.platoons)):
            self.platoons[n].epsilon_decay()

    def init_last_action(self):
        """
        The company get all of the last actions of each platoon.
        Check on 2021-10-26, this last_action is not used in training.
        :return:
        """
        self.last_action = np.zeros((self.args.n_ally_platoons, self.args.n_ally_agent_in_platoon, self.args.n_actions))
        for n in range(len(self.platoons)):
            self.platoons[n].init_last_action()
        return

    def choose_action(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False):
        return

    # TODO: Check how to record the episode buffer
    def init_episode_buffer(self):
        return

    def init_policy(self):
        for n in range(len(self.platoons)):
            self.platoons[n].policy.init_hidden(1)

    # def get_avail_actions(self):
    #     for n in range(len(self.platoons)):
    #         self.platoons[n].get_avail_actions()
    #     return
    #
    # TODO the company obs is the aggregated obs of the whole company.
    # def init_platoons(self, args, itr):
    #     for n in range(len(self.platoons)):
    #         self.platoons[n] = Platoon(args, itr=itr)

    def get_company_observation(self):
        return

    # TODO the company action is to "move to certain strategic location".
    def get_company_action(self):
        return

    def record_company_last_action(self):
        return
