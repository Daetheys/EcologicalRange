import haiku as hk
import jax.numpy as jnp
import numpy as np

class Agent:
    def __init__(self,env,idx,seed=0,logger_queue=None):
        self.env = env
        self.seed = seed
        self.rng = hk.PRNGSequence(self.seed)
        self.idx = idx
        self.logger_queue = logger_queue
    def forward(self,obs):
        raise NotImplementedError
    def learn(self):
        raise NotImplementedError
    def log(self,name,val):
        if self.logger_queue:
            command = ('add','worker_'+str(self.idx)+'/'+name,val)
            self.logger_queue.put(command)

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
            no,r,d,i = self.env.step(a)
            ts = Timestep(o,a,lp,r,no,d,i)

            self.log('observation',o[0])
            self.log('action',a[0])
            self.log('reward',r[0])
            self.log('done',d[0])
            self.log('new_observation',no[0])
            #self.logger.add('info',i[0])

            self.learn(ts)
            if np.any(d):
                o = self.env.reset()
            else:
                o = no
