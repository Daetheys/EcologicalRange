import haiku as hk
import jax.numpy as jnp
import numpy as np

class Agent:
    def __init__(self,env,seed=0,logger=None):
        self.env = env
        self.seed = seed
        self.rng = hk.PRNGSequence(self.seed)
        self.logger = logger
    def forward(self,obs):
        raise NotImplementedError
    def learn(self):
        raise NotImplementedError

import collections
Timestep = collections.namedtuple('Timestep',['o','a','lp','r','no','d','i'])

class OnlineAgent(Agent):
    def __init__(self,env,*args,**kwargs):
        assert env.batch_size == 1 #OnlineAgent cannot learn on vectorized environments
        super().__init__(env,*args,**kwargs)

    def step(self):
        pass

    def train(self,nb_steps):
        o = self.env.reset()
        for i in range(nb_steps):
            a,lp = self.forward(o)
            print(i,o,a)
            no,r,d,i = self.env.step(a)
            ts = Timestep(o,a,lp,r,no,d,i)
            if self.logger:
                self.logger.add('reward',np.array(r))
            self.learn(ts)
            if np.any(d):
                o = self.env.reset()
            else:
                o = no
    
    def test(self,nb_steps):
        o = self.env.reset()
        for i in range(nb_steps):
            a = self.forward(o)
            no,r,d,i = self.env.step(a)
            ts = Timestep(o,a,r,no,d,i)