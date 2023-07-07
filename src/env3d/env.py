import numpy as np
import datetime
import math
import random
import rospy
from env3d.settings import Scene_settings
from env3d.agent.transition import Transition
from pcl_policy.pcl_rainbow.discriminator.discriminator import NaivePCTSeg
import time
import torch
import torch.nn as nn



class Env(object):
    def __init__(self,args,config):
        self.scene=Scene_settings(config)
        self.transition=Transition(config,self.scene)
        self.pcl_size=1620
        self.transition.pcl_size=self.pcl_size
        self.transition.oracle=False
        self.config=config
        device = "cuda"
        self.model = NaivePCTSeg().to(device)
        self.model = nn.DataParallel(self.model) 
        print(self.model.load_state_dict(torch.load("pcl_policy/pcl_rainbow/discriminator/model.t7")))
        self.model.eval()

        


    def reset(self):
        self.timeout=False 
        self.done=False
        self.pcl= np.zeros((3,self.pcl_size))
        
        self.scene.reset()
        self.transition.reset()
        self.old_pcl=np.zeros((1,3))
        state=[self.pcl, self.scene.quat_pose]
        self.sum_reward=0
        self.points=0
        self.sum_old_points=0
        return state 





    def step(self, unsorted_actions):
        pcl=None
        terminate=False
        pcl_state= np.zeros((3,self.pcl_size))
        oracle=True
        if oracle==True:
            pcl_state= np.zeros((4,self.pcl_size))
        actions, i = self.transition.legal_transition(unsorted_actions)
        pcl, pose, oracle = self.transition.make_action(actions[i])
        pcl=np.swapaxes(pcl,0,1)
        self.points=pcl.shape[1]
        if pcl.shape[1]>1619:
            terminate=True
            print("too many points !!!", pcl.shape)
        else:
            pcl_state[:,:pcl.shape[1]] =pcl
        state=[pcl_state, pose]
        #if state.shape[0]>512:
        if oracle:
            reward = self.transition.reward
        else:
            #print(pcl_state.shape)
            logits = self.model(torch.from_numpy(pcl_state).float().unsqueeze(0))
            preds = logits.max(dim=1)[1].cpu().numpy()
            #print(preds)
            sum_reward=np.sum(preds)
            if self.sum_reward<sum_reward:
                reward=sum_reward-self.sum_reward
                self.sum_reward=sum_reward
            else:
                reward=0
            if self.sum_old_points<pcl.shape[1]:
                reward= reward + (0.5*(pcl.shape[1]-self.sum_old_points))
                self.sum_old_points=pcl.shape[1]
        return state, reward, actions, i, terminate


    def simulate_step(self, action):
        pcl= np.zeros((3,0))
        if (self.transition.check_transition(action)):
            reward = self.transition.simulate_action(action)
            #pcl=np.swapaxes(pcl,0,1)
            #print(self.old_pcl.shape[1], pcl.shape[1])
            #if self.old_pcl.shape[1] < pcl.shape[1]:
            #    reward=pcl.shape[1]-self.old_pcl.shape[1]
            #else:
            #    reward = 0
        else:
            reward = 0
        return reward

    """
    def simulate_step(self, action):
        pcl= np.zeros((3,0))
        if (self.transition.check_transition(action)):
            pcl = self.transition.simulate_action(action)
            pcl=np.swapaxes(pcl,0,1)
            print(self.old_pcl.shape[1], pcl.shape[1])
            if self.old_pcl.shape[1] < pcl.shape[1]:
                reward=pcl.shape[1]-self.old_pcl.shape[1]
            else:
                reward = 0
        else:
            reward = 0
        return reward
    """

         
        


