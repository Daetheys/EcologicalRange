from agent.agent import *
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from utils.sampling import sample_batch_index

class QAgent(OnlineAgent):
    def __init__(self,*args,temp=1.,alpha_q=0.1,informed=False,scaled_beta=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset()

        self.temp = temp
        self.alpha_q = alpha_q
        self.informed = informed
        self.scaled_beta = scaled_beta

    def reset(self):
        self.q_values = np.zeros(self.env.n_symbols)
        
    #@partial(jax.jit,static_argnums=0)
    def forward(self,obs):
        q_val = self.q_values[np.array(obs)]
        temp = self.temp
        if self.scaled_beta:
            temp *= self.env.get_current_range()
        q_val_temp = q_val/temp
        action_probs = jax.nn.softmax(q_val_temp,axis=1)
        self.log('ActionProbs',action_probs[0])
        print('--')
        print(q_val.min(),q_val.max(),q_val.mean(),q_val.var())
        print(q_val_temp.min(),q_val_temp.max(),q_val_temp.mean(),q_val_temp.var())
        print(temp,self.env.get_current_range(),action_probs.max())
        actions = sample_batch_index(next(self.rng),action_probs)
        logprobs = jnp.log(action_probs[np.arange(action_probs.shape[0]),np.array(actions)])
        return actions,logprobs

    def learn(self,ts):
        (o,a,lp,r,no,d,i) = ts
        chosen_symbol = o[0,a[0]]
        o = o[0]

        #Update Q
        self.q_values[chosen_symbol] += self.alpha_q * (r - self.q_values[chosen_symbol])#/np.exp(lp)

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
            self.log('EV',self.env.get_ev())
            self.log('QVal',self.q_values)

            self.learn(ts)
            if np.any(d):
                if len(self.env.min_range) == self.env.current_season+1:
                    return
                self.env.next_season()
                self.q_values *= 0
                if self.informed:
                    self.q_values += self.env.get_ev()
                o = no
            else:
                o = no
