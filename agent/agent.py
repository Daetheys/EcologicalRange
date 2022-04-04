class Agent:
    def __init__(self,env):
        self.env = env
    def forward(self,obs):
        raise NotImplementedError
    def learn(self):
        raise NotImplementedError
