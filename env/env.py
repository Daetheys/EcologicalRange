class Env:
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.n_states = None
        self.action_dim = None
    def reset(self):
        raise NotImplementedError
    def step(self,a):
        raise NotImplementedError
