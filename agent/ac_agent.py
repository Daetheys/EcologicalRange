from agent.agent import *
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from utils.sampling import sample_batch_index

class ACAgent(OnlineAgent):
    def __init__(self,*args,beta=0.5,alpha_v=0.1,informed=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset()

        self.alpha_v = alpha_v
        self.beta = beta
        self.informed = informed

    def reset(self):
        self.value = 0
        self.p_values = np.zeros(self.env.n_symbols)
        
    #@partial(jax.jit,static_argnums=0)
    def forward(self,obs):
        p_val = self.p_values[np.array(obs)]
        action_probs = jax.nn.softmax(p_val,axis=1)
        actions = sample_batch_index(next(self.rng),action_probs)
        logprobs = jnp.log(action_probs[np.arange(action_probs.shape[0]),np.array(actions)])
        return actions,logprobs

    def learn(self,ts):
        (o,a,lp,r,no,d,i) = ts
        chosen_symbol = o[0,a[0]]
        o = o[0]

        #Update Q
        new_value = self.value + self.alpha_v * (r - self.value)
        self.p_values[chosen_symbol] += self.beta * (r + new_value - self.value) * (1 - np.exp(lp))
        self.value = new_value

    def train(self,nb_steps):
        o = self.env.reset()
        for i in range(nb_steps):
            a,lp = self.forward(o)
            no,r,d,_ = self.env.step(a)
            ts = Timestep(o,a,lp,r,no,d,i)

            self.log('Observation',o[0])
            self.log('Action',a[0])
            self.log('Reward',r[0])
            self.log('Done',d[0])
            self.log('NewObservation',no[0])
            self.log('EnvMini',self.env.min_range[self.env.current_season])
            self.log('EnvMaxi',self.env.max_range[self.env.current_season])
            self.log('EV',self.env.contexts.prod(axis=2).sum(axis=2).mean(axis=1).item())
            self.log('PVal',self.p_values)
            #for i in range(len(self.p_values)):
            #    self.log('PVal_'+str(i),self.p_values[i])

            self.learn(ts)
            if np.any(d):
                if len(self.env.min_range) == self.env.current_season+1:
                    return
                self.env.next_season()
                self.p_values *= 0
                self.value = 0
                if self.informed:
                    self.value = self.env.get_ev()
                o = no
            else:
                o = no
