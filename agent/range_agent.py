from agent.agent import *
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from utils.sampling import sample_batch_index

class RangeAgent(OnlineAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset()
    def reset(self):
        self.mini = np.zeros((self.env.n_states))
        self.maxi = np.zeros((self.env.n_states))
        self.q_values = np.zeros((self.env.n_states))

        self.temp = 1
        self.alpha_ext = 0.1
        self.alpha_int = 0.01
        self.alpha_q = 0.1
    #@partial(jax.jit,static_argnums=0)
    def forward(self,obs):
        q_val = self.q_values[np.array(obs)]
        q_val_temp = q_val/self.temp
        action_probs = jax.nn.softmax(q_val_temp,axis=1)
        action_probs = action_probs*0+0.5
        actions = sample_batch_index(next(self.rng),action_probs)
        logprobs = jnp.log(action_probs[np.arange(action_probs.shape[0]),np.array(actions)])
        return actions,logprobs
        
    def learn(self,ts):
        (o,a,lp,r,no,d,i) = ts
        chosen_symbol = o[0,a[0]]

        #if chosen_symbol == 1:
        #    print('---')
        #    print(chosen_symbol,self.mini[chosen_symbol],r,self.maxi[chosen_symbol])

        #lp = 1

        #Update Mini
        alpha = self.alpha_int
        if r < self.mini[chosen_symbol]:
            alpha = self.alpha_ext
        self.mini[chosen_symbol] += alpha*(r-self.mini[chosen_symbol])#/np.exp(lp)

        #Update Maxi
        alpha = self.alpha_int
        if self.maxi[chosen_symbol] < r:
            alpha = self.alpha_ext
        self.maxi[chosen_symbol] += alpha*(r-self.maxi[chosen_symbol])#/np.exp(lp)

        #Update Q
        relative_r = (r-self.mini[chosen_symbol])/(self.maxi[chosen_symbol] - self.mini[chosen_symbol])
        self.q_values[chosen_symbol] += self.alpha_q * (relative_r - self.q_values[chosen_symbol])#/np.exp(lp)

        print('---')
        print(self.mini,jnp.exp(lp))
        print(self.maxi)
        print(chosen_symbol,relative_r)