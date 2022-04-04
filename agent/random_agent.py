from agent import Agent
class RandomAgent(Agent):
    def __init__(self,env):
        super().__init__(env)
    def forward(self,obs):
        return self.env.action_space.sample()
