
# -*- coding: utf-8 -*-

import os
import sys
import PyKDL
import torch
import argparse
import configparser
#import open3d
from env3d.agent.transition import Transition
import env3d.settings


from importlib import reload
import run

from env3d.env import Env
from env3d.agent.transition import Transition
from pcl_policy.pcl_rainbow.rainbow import PCL_rainbow
from pcl_policy.greedy.greedy import Greedy

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--networkPath', default='network/', help='folder to put results of experiment in')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--load_net', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    # Environment
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--train', action='store_true', help='train the agent in the terminal')
    parser.add_argument('--episode_length', type=int, default=300, help='length of mapping environment episodes')
    #Agent
    parser.add_argument('--rainbow', action='store_true', help='off policy agent rainbow')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--discount', type=float, default=0.99, metavar='y', help='Discount factor')
    parser.add_argument('--priority_weight', type=float, default=0.2, metavar='beta', help='priority weight beta')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluation-interval', type=int, default=12000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--id', type=str, default='new_base_multihead', help='Experiment_oas_emb')
    parser.add_argument('--model_path', type=str, default = "results/new_base/checkpoint.pth", help='model used during testing / visulization') #testmoreFilters.h5
    parser.add_argument('--exp_name', type=str, default = "", help='')
    parser.add_argument('--frame_width', type=int, default = 84, help='Resized frame width')
    parser.add_argument('--frame_height', type=int, default = 84, help='Resized frame height')
    parser.add_argument('--num_steps', type=int, default = 2e6, help='Number of episodes the agent plays')
    parser.add_argument('--state_length', type=int, default = 4, help='Number of most recent frames to produce the input to the network')
    parser.add_argument('--gamma', type=float, default = 0.99, help='Discount factor')
    parser.add_argument('--exploration_steps', type=int, default =50000, help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')#100000
    parser.add_argument('--initial_epsilon', type=float, default = 1.99, help='Initial value of epsilon in epsilon-greedy')
    parser.add_argument('--final_epsilon', type=float, default = 0.1, help='Final value of epsilon in epsilon-greedy')
    parser.add_argument('--initial_replay_size', type=int, default =100000, help='Number of steps to populate the replay memory before training starts')

    parser.add_argument('--num_replay_memory', type=int, default = 1500000, help='Number of replay memory the agent uses for training')

    parser.add_argument('--batch_size', type=int, default = 32, help='Mini batch size')
    parser.add_argument('--target_update_interval', type=int, default = 10000, help='The frequency with which the target network is updated')
    parser.add_argument('--train_interval', type=int, default = 4, help='The agent selects 4 actions between successive updates')
    parser.add_argument('--learning_rate', type=float, default = 0.00001, help='Learning rate used by RMSProp')
    parser.add_argument('--min_grad', type=float, default = 1e-8, help='Constant added to the squared gradient in the denominator of the RMSProp update')
    parser.add_argument('--save_interval', type=int, default = 5000, help='The frequency with which the network is saved')
    parser.add_argument('--no_op_steps', type=int, default = 10, help='Maximum number of "do nothing" actions to be performed by the agent at the start of an episode')
    parser.add_argument('--save_network_path', type=str, default = "saved_dqn_networks/", help='')
    parser.add_argument('--save_summary_path', type=str, default = "dqn_summary/", help='')


    parser.add_argument('--gpu_frac', type=float, default = 1.0, help='Set GPU use limit for tensorflow')
    parser.add_argument('--ddqn', type=bool, default = False, help='Set True to apply Double Q-learning')
    parser.add_argument('--dueling', type=bool, default = False, help='Set True to apply Duelinng Network')
    parser.add_argument('--optimizer',type=str, default='adam', help='Optimizer (Adam or Rmsp)')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    ######################     ssh -L 6006:localhost:6006  zsofia@10.20.7.59
    #Make video arguments#     http://localhost:6006/#scalars
    ######################     python3.7 -m tensorboard.main --logdir tensorboard/dddqn/1
    #python3 main.py --test_dqn --test_dqn_model_path saved_dqn_networks/new_env_125000.h5 --do_render
    parser.add_argument('-f', '--num_frames', default=100, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=0, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='./movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    parser.add_argument('--greedy', action='store_true', help="run greedy agent")
    parser.add_argument('--eval', action='store_true', help="vlidate results and print a graph")

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = True#args.enable_cudnn
    else:
        args.device = torch.device('cpu')
    return args


if __name__ == '__main__':
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    args = parse()
    results_dir = os.path.join('results', args.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    os.makedirs(args.networkPath, exist_ok=True)
    config = configparser.ConfigParser()
    config.read('./config.ini')
    metrics = {'steps': [], 'rewards': [], 'entropy': []}
    
    env = Env(args, config)
    
    #if not args.greedy:
        
    #    agent = PCL_rainbow(args, env)
    #else:
    #    agent = Greedy(args, env, action_space=6)
    if args.eval:
        agent1 = Greedy(args, env, action_space=6)
        agent2 = PCL_rainbow(args, env)
        run.eval(args, env, agent1, agent2, config)
    else:
        run.init(args, env, agent, config)
