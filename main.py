# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/15 17:23
@Thanks： Kunfeng Li
@Author: Geng, Minghong on 2021-09-21
"""

from src.smac_HC.env import StarCraft2Env_HC

# argument.py creates 2 functions, get_common_args() and get_q_decom_args()
# get_common_args() contains definitions of common parameters related to the environment setting, training process etc.
# get_q_decom_args()contains definitions of parameters used in algorithms like VDN, QMIX and QTRAN.
from src.common.arguments import get_common_args, get_q_decom_args
from src.common.runner import Runner
import time
from multiprocessing import Pool
import os
import torch


def main(env, arg, itr):
    # pass in the environment and arguments in Runner().
    # The "itr" is the id of process. Defined in Runner.__init__
    runner = Runner(env, arg, itr)
    # 如果训练模型
    # arguments.learn is a boolean value, and it's defined in function get_q_decom_args().
    if arguments.learn:
        runner.run()
    runner.save_results()
    runner.plot()


if __name__ == '__main__':
    start = time.time()
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('StartTime:' + start_time)

    # function get_q_decom_args() add more parameters based on the parameters generated from get_common_args(), which
    # are simple arguments.
    arguments = get_q_decom_args(get_common_args())
    print('Map Name: ', arguments.map, '\n')
    arguments.replay_dir = arguments.replay_dir + '/' + arguments.map
    print('Replay Dir: ', arguments.replay_dir, '\n')
    
    if arguments.load_model:
        model_dir = arguments.model_dir + '/' + arguments.alg + '/' + arguments.map + '/' + str(1)
        if os.path.exists(model_dir + '/epsilon.txt'):
            path_epsilon = model_dir + '/epsilon.txt'
            f = open(path_epsilon, 'r')
            arguments.epsilon = float(f.readline())
            f.close()
            print('Successfully load epsilon:' + str(arguments.epsilon))
        else:
            print('No epsilon in this directory.')

    # set up the GPU resource
    if arguments.gpu is not None:
        arguments.cuda = True
        if arguments.gpu == 'a':
            pass
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        arguments.cuda = False
        
    environment = StarCraft2Env_HC(map_name=arguments.map,
                                   step_mul=arguments.step_mul,
                                   difficulty=arguments.difficulty,
                                   replay_dir=arguments.replay_dir,
                                   hierarchical=arguments.hierarchical,
                                   n_ally_platoons=arguments.n_ally_platoons,
                                   n_ally_agent_in_platoon=arguments.n_ally_agent_in_platoon,
                                   map_sps=arguments.map_sps,
                                   # formation=arguments.formation,
                                   # platoon_indi_sps = arguments.platoon_indi_sps,
                                   train_on_intervals = arguments.train_on_intervals,
                                   FALCON_demo=arguments.FALCON_demo,
                                   obs_distance_target=arguments.obs_distance_target,
                                   allmaps = arguments.allmaps,
                                   final_SP_to_reach = arguments.final_SP_to_reach,
                                   debug=False)

    # retrieve the information of the environment
    env_info = environment.get_env_info()

    # TODO The state shape of the platoon and the company need further design

    # state shape
    arguments.state_shape = env_info['state_shape']
    arguments.company_state_shape = env_info['company_state_shape']
    arguments.platoon_state_shape = env_info['platoon_state_shape']

    # observation shape
    arguments.obs_shape = env_info['obs_shape']
    arguments.company_obs_shape = env_info['company_obs_shape']
    arguments.platoon_obs_shape = env_info['platoon_obs_shape']
    
    arguments.episode_limit = env_info['episode_limit']

    # number of action.
    # number of action equals to the number of none-attack actions (move to 4 directions, stop, and no-op) plus the
    # number of enemies. So, this number is 6+X.
    arguments.n_actions = env_info['n_actions']
    arguments.n_agents = env_info['n_agents']
    
    if arguments.num > 1:
        p = Pool(12)
        for i in range(arguments.num):
            p.apply_async(main, args=(environment, arguments, i))
        p.close()
        p.join()
    else:
        main(environment, arguments, 1)  # Start training！ Run main() function.

    duration = time.time() - start
    time_list = [0, 0, 0]
    time_list[0] = duration // 3600
    time_list[1] = (duration % 3600) // 60
    time_list[2] = round(duration % 60, 2)
    print('Time Elapsed：' + str(time_list[0]) + ' hour ' + str(time_list[1]) + 'minute' + str(time_list[2]) + 'second')
    print('StartTime：' + start_time)
    end_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('EndTime：' + end_time)
