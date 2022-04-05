from agent.agent import *
class RandomAgent(OnlineAgent):
    def __init__(self,env):
        super().__init__(env)
    def forward(self,obs):
        return self.env.action_space.sample()
    def learn(self,ts):
        pass