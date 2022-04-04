from agent import Agent

class RangeAgent(Agent):
    def __init__(self,env):
        super().__init__(env)
        self.reset()
    def reset(self):
        self.mini = 0
        self.maxi = 0
    def forward(self,obs):
        
