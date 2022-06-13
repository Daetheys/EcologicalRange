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

        self.memory_reward = []
        self.memory_action = []

    def reset(self):
        self.q_values = np.zeros(self.env.n_symbols)
        
    #@partial(jax.jit,static_argnums=0)
    def forward(self,obs):
        q_val = self.q_values[np.array(obs)]
        temp = self.temp
        if self.scaled_beta == 'cheat':
            temp *= self.env.get_current_range()
        elif self.scaled_beta == 'memory':
            if max(self.q_values) != min(self.q_values):
                temp *= max(self.q_values)-min(self.q_values)
        q_val_temp = q_val/temp
        action_probs = jax.nn.softmax(q_val_temp,axis=1)
        self.log('ActionProbs',action_probs[0])
        actions = sample_batch_index(next(self.rng),action_probs)
        logprobs = jnp.log(action_probs[np.arange(action_probs.shape[0]),np.array(actions)])
        return actions,logprobs

    def learn(self,ts):
        (o,a,lp,r,no,d,i) = ts
        chosen_symbol = o[0,a[0]]
        o = o[0]

        #Update Q
        self.q_values[chosen_symbol] += self.alpha_q * (r - self.q_values[chosen_symbol])#/np.exp(lp)

    def init_new_season(self):
        self.memory_reward.append([])
        self.memory_action.append([])

    def train(self,nb_steps):
        o = self.env.reset()
        self.init_new_season()
        for i in range(nb_steps):
            a,lp = self.forward(o)
            no,r,d,_ = self.env.step(a)
            ts = Timestep(o,a,lp,r,no,d,i)

            self.memory_reward[-1].append(r)
            self.memory_action[-1].append(a)

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
                if self.informed == 'cheat':
                    self.q_values *= 0
                    self.q_values += self.env.get_ev()
                elif self.informed == 'memory':
                    unique_actions = np.unique(self.memory_action[-1])
                    print(unique_actions)
                    self.q_values[:] = self.q_values[unique_actions].mean()
                
                self.init_new_season()
                o = no
            else:
                o = no
