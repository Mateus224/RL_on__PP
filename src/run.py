from importlib import reload

from env3d.env import  Env
import rospy
import os
import numpy as np
from tqdm import trange
from pcl_policy.pcl_rainbow.memory import ReplayMemory
from pcl_policy.pcl_rainbow.rainbow import PCL_rainbow
from torch.utils.tensorboard import SummaryWriter
from pcl_policy.greedy.greedy import Greedy
import torch
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line


def init(args,env, agent,config):
    if True:
        if type(agent) == Greedy:
            writer = SummaryWriter()
            T, done, timeout = 0, False, False
            sum_reward, episode = 0, 0
            state, _ = env.reset()
            for T in trange(1,int(args.num_steps)):#int(args.num_steps)):
                if T%65==0:
                    episode+=1
                    timeout= True
                if done or timeout:
                    writer.add_scalar("Reward", sum_reward/(episode), episode)
                    print("Reward:" ,sum_reward)#,(sum_reward+(0.35*done))/(env.start_entr_map))
                    #sum_reward=0
                    state, _ = env.reset()
                    done=False
                    timeout= False
                action = agent.get_action(state)
                next_state, reward, actions, i, done = env.step(action)
                sum_reward=sum_reward+reward
                state=next_state
        if type(agent)== PCL_rainbow:
            agent.train()
            timeout=False
            episode=0
            writer = SummaryWriter()
            mem = ReplayMemory(args, args.num_replay_memory)
            priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
            results_dir = os.path.join('results', args.id)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            T, t, done = 0, 0, False
            sum_reward=0
            state, _ = env.reset()
            for T in trange(1,int(args.num_steps)):#int(args.num_steps)):
                t=t+1
                if done:
                    t=65
                if t%65==0:
                    episode+=1
                    timeout= True
                    t=0
                
                if timeout:
                    writer.add_scalar("Reward", sum_reward, episode)
                    print("Reward:" ,sum_reward)#,(sum_reward+(0.35*done))/(env.start_entr_map))
                    sum_reward=0
                    state, _ = env.reset()
                    done=False
                    timeout= False
                
                action = agent.epsilon_greedy(T,200000, state)
                #print(action)


                agent.reset_noise()  # Draw a new set of noisy weights


                next_state, reward, actions, i, done = env.step(action)  # Step
                
                if done:
                    print(400, 'asdasd', reward, False, i)
                sum_reward=sum_reward+reward
                #print("reward:",reward, sum_reward)
                 # Append transition to memory

                # Train and test
                if i>0:
                    for j in range(i-1):
                        mem.append(state, actions[j], 0, True)
                #print(state[0].shape)
                mem.append(state, actions[i], reward, done) 
                if T >= 1500:#args.learn_start:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    agent.learn(mem)  # Train with n-step distributional double-Q learning

                    if T % args.target_update == 0:
                        agent.update_target_net()

                    # Checkpoint the network
                    if (args.save_interval != 0) and (T % args.save_interval == 0):
                        agent.save(results_dir, 'checkpoint.pth')

                state = next_state

def eval(args, env, agent1, agent2, config):
    List2_columns=1
    List1_row=100
    s = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    p_g = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    p_rl= [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    metrics = {'steps': s, 'points_g': p_g, 'points_rl': p_rl}
#    if type(agent) == Greedy:
    #writer = SummaryWriter()
    for j in range(List1_row):
        timeout=False
        episode=0
        T, t, done = 0, 0, False
        steps=100
        state, _ = env.reset()
        step = 0
        done = False
        
        sum_reward=0
        while not done:
            action = agent1.get_action(state)
            
            next_state, reward, actions, i, done = env.step(action)
            sum_reward=sum_reward+reward
            if step<=steps:
                metrics['steps'][j].append(step)
                metrics['points_g'][j].append(sum_reward)
                step+=1
            else:
                print(j, sum_reward)
                done = True      
#    if type(agent)== PCL_rainbow:
    for j in range(List1_row):
        timeout=False
        episode=0
        agent2.eval()
        T, t, done = 0, 0, False
        sum_reward=0
        state, _ = env.reset()
        step = 0
        done = False
        
        while not done:
            action = agent2.make_action(state)
            next_state, reward, actions, i, done_ = env.step(action)  # Step
            sum_reward=sum_reward+reward
            state=next_state
            if step<=steps:
                #metrics['steps'][j].append(step)
                metrics['points_rl'][j].append(sum_reward)
                step+=1
            else:
                print(j, sum_reward, "rl")
                done = True
    _plot_line(metrics['steps'], metrics['points_g'], metrics['points_rl'], 'points')



def _plot_line( xs, ys_population_g,ys_population_rl, title, path='/home/matthias/'):
    max_colour_g, mean_colour_g, std_colour_g, transparent_g = 'rgb(0, 160, 0)', 'rgb(0, 210, 0)', 'rgba(0, 210, 0, 0.2)', 'rgba(0, 0, 0, 0)'
    max_colour_rl, mean_colour_rl, std_colour_rl, transparent_rl = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(0, 172, 237, 0.2)', 'rgba(0, 0, 0, 0)'
    ysG = torch.tensor(ys_population_g, dtype=torch.float32)
    ysRl = torch.tensor(ys_population_rl, dtype=torch.float32)

    ys_g = ysG[0].squeeze()
    ys_rl = ysRl[0].squeeze()

    ys_min_g, ys_max_g, ys_mean_g, ys_std_g = np.amax(ys_population_g, axis=0), np.amin(ys_population_g, axis=0), ysG.mean(0).squeeze(), ysG.std(0).squeeze()
    ys_min_rl, ys_max_rl, ys_mean_rl, ys_std_rl = np.amax(ys_population_rl, axis=0), np.amin(ys_population_rl, axis=0), ysRl.mean(0).squeeze(), ysRl.std(0).squeeze()
    
    ys_upper_g, ys_lower_g = ys_mean_g + (1.96*(ys_std_g/(ys_g**0.5))), ys_mean_g - (1.96*(ys_std_g/(ys_g**0.5)))
    ys_upper_rl, ys_lower_rl = ys_mean_rl + (1.96*(ys_std_rl/(ys_rl**0.5))), ys_mean_rl - (1.96*(ys_std_rl/(ys_rl**0.5)))

    
    trace_max_g = Scatter(x=xs[0], y=ys_max_g, fillcolor=std_colour_g,  line=Line(color=max_colour_g, dash='dash'), name='Greedy max')
    trace_upper_g = Scatter(x=xs[0], y=ys_upper_g.numpy(), fillcolor=std_colour_g, line=Line(color=transparent_g), name='+1 Std. Dev.', showlegend=False) #line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean_g = Scatter(x=xs[0], y=ys_mean_g,  fill='tonexty',  fillcolor=std_colour_g, line=Line(color=mean_colour_g), name='Greedy mean')
    trace_lower_g = Scatter(x=xs[0], y=ys_lower_g.numpy(),fill='tonexty',  fillcolor=std_colour_g, line=Line(color=transparent_g), name='-1 Std. Dev.', showlegend=False)
    trace_min_g = Scatter(x=xs[0], y=ys_min_g, line=Line(color=max_colour_g, dash='dash'), name='Greedy min')

    trace_max_rl = Scatter(x=xs[0], y=ys_max_rl, fillcolor=std_colour_rl,  line=Line(color=max_colour_rl, dash='dash'), name='RL max')
    trace_upper_rl = Scatter(x=xs[0], y=ys_upper_rl.numpy(), fillcolor=std_colour_rl, line=Line(color=transparent_rl), name='+1 Std. Dev.', showlegend=False) #line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean_rl = Scatter(x=xs[0], y=ys_mean_rl,  fill='tonexty',  fillcolor=std_colour_rl, line=Line(color=mean_colour_rl), name='RL mean')
    trace_lower_rl = Scatter(x=xs[0], y=ys_lower_rl.numpy(),fill='tonexty',  fillcolor=std_colour_rl, line=Line(color=transparent_rl), name='-1 Std. Dev.', showlegend=False)
    trace_min_rl = Scatter(x=xs[0], y=ys_min_rl, line=Line(color=max_colour_rl, dash='dash'), name='RL min')

    plotly.offline.plot({
        'data':  [trace_max_g,trace_upper_g, trace_mean_g, trace_lower_g, trace_min_g, trace_max_rl,trace_upper_rl, trace_mean_rl, trace_lower_rl, trace_min_rl],
        'layout': dict(font_size=40,title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)