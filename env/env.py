class Env:
    def __init__(self):
        pass
    def reset(self):
        raise NotImplementedError
    def step(self,a):
        raise NotImplementedError
