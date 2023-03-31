import numpy as np




class Greedy():
    def __init__(self, args, env, action_space):
        self.pcl_old=np.zeros((3,512))
        self.postion= np.zeros((7))
        self.action_space = action_space
        self.env = env
        self.reward= np.zeros((action_space))

        

        
    def get_action(self, state, action_num=6):
        self.position=state[0]
        self.pcl_old = state[1]
        reward_old=0
        for action in range(action_num):
            self.reward[action] = self.env.simulate_step(action)
        return self.reward