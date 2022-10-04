from importlib import reload

from env3d.env import  Env
import rospy
import os
import numpy as np
from tqdm import trange
from pcl_policy.pcl_rainbow.memory import ReplayMemory
from pcl_policy.pcl_rainbow.rainbow import PCL_rainbow



def init(args,env, agent,config):
    
    if True:
        agent.train()
        if type(agent)== PCL_rainbow:
            timeout=False
            mem = ReplayMemory(args, args.num_replay_memory)
            priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
            results_dir = os.path.join('results', args.id)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            T, done = 0, False
            sum_reward=0
            state, _ = env.reset()
            for T in trange(1,int(args.num_steps)):#int(args.num_steps)):
                if T%50==0:
                    timeout= True
                if done or timeout:

                    print("Reward:" ,sum_reward)#,(sum_reward+(0.35*done))/(env.start_entr_map))
                    sum_reward=0
                    state, _ = env.reset()
                    done=False
                    timeout= False
                    #writer.add_scalar("Litter", litter, episode)
                action = agent.epsilon_greedy(T,500000, state)


                agent.reset_noise()  # Draw a new set of noisy weights


                next_state, reward, action, done = env.step(action)  # Step

                sum_reward=sum_reward+reward
                 # Append transition to memory

                # Train and test
                #if sim_i>0:
                #    for j in range(sim_i-1):
                #        mem.append(state, actions[j], 0, True)
                #print(state[0].shape)
                mem.append(state, action, reward, done) 
                if T >= 10000:#args.learn_start:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    agent.learn(mem)  # Train with n-step distributional double-Q learning

                    if T % args.target_update == 0:
                        agent.update_target_net()

                    # Checkpoint the network
                    if (args.save_interval != 0) and (T % args.save_interval == 0):
                        agent.save(results_dir, 'checkpoint.pth')

                state = next_state
