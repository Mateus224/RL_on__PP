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
        state = env.reset()
        for T in trange(1,int(args.num_steps)):#int(args.num_steps)):
            t=t+1
            if done:
                t=101
            if t%101==0:
                episode+=1
                timeout= True
                t=0
            
            if timeout:
                writer.add_scalar("Reward", sum_reward, episode)
                print("Reward:" ,sum_reward , env.points)#,(sum_reward+(0.35*done))/(env.start_entr_map))
                sum_reward=0
                state = env.reset()
                done=False
                timeout= False
            
            action = agent.epsilon_greedy(T,300000, state)
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
            if T >= 20000:#args.learn_start:
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                agent.learn(mem)  # Train with n-step distributional double-Q learning

                if T % args.target_update == 0:
                    agent.update_target_net()

                # Checkpoint the network
                if (args.save_interval != 0) and (T % args.save_interval == 0):
                    agent.save(results_dir, 'checkpoint.pth')

            state = next_state

def eval(args, env, agent, config):
    List2_columns=1
    List1_row=100
    s = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    p = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    metrics = {'steps': s, 'points': p}
    if type(agent) == Greedy:
        #writer = SummaryWriter()
        for j in range(List1_row):
            timeout=False
            episode=0
            T, t, done = 0, 0, False
            sum_reward=0
            state, _ = env.reset()
            step = 0
            done = False
            print(j)
            while not done:
                action = agent.get_action(state)
                next_state, reward, actions, i, done = env.step(action)
                sum_reward=sum_reward+reward
                if step<=99:
                    step+=1
                    metrics['steps'][j].append(step)
                    metrics['points'][j].append(sum_reward)
                    
                else:
                    print(sum_reward)
                    done = True      
    if type(agent)== PCL_rainbow:
        for j in range(List1_row):
            timeout=False
            episode=0
            agent.eval()
            T, t, done = 0, 0, False
            sum_reward=0
            state = env.reset()
            step = 0
            done = False
            print(j)
            while not done:
                action = agent.make_action(state)
                next_state, reward, actions, i, done_ = env.step(action)  # Step
                sum_reward=sum_reward+reward
                state=next_state

                if step<=100:
                    metrics['steps'][j].append(step)
                    
                    metrics['points'][j].append(sum_reward)
                    step+=1
                else:
                    done = True


    _plot_line(metrics['steps'], metrics['points'], 'Points')



def _plot_line( xs, ys_population, title, path='/home/matthias/'):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'
    ys = torch.tensor(ys_population, dtype=torch.float32)

    ys_ = ys[0].squeeze()

    ys_min, ys_max, ys_mean, ys_std = np.amax(ys_population, axis=0), np.amin(ys_population, axis=0), ys.mean(0).squeeze(), ys.std(0).squeeze()
    ys_upper, ys_lower = ys_mean + (1.96*(ys_std/(ys_**0.5))), ys_mean - (1.96*(ys_std/(ys_**0.5)))


    
    trace_max = Scatter(x=xs[0], y=ys_max, fillcolor=std_colour,  line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs[0], y=ys_upper.numpy(), fillcolor=std_colour, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False) #line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs[0], y=ys_mean,  fill='tonexty',  fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs[0], y=ys_lower.numpy(),fill='tonexty',  fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs[0], y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data':  [trace_max,trace_upper, trace_mean, trace_lower, trace_min],
        'layout': dict(font_size=40,title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)