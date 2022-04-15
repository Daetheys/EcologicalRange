from agent.agent import *
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from utils.sampling import sample_batch_index

class RangeAgent(OnlineAgent):
    def __init__(self,*args,temp=1.,alpha_ext=0.1,alpha_int=0.01,alpha_q=0.1,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset()

        self.temp = temp
        self.alpha_ext = alpha_ext
        self.alpha_int = alpha_int
        self.alpha_q = alpha_q

    def reset(self):
        self.mini = {'[0 1]':0,'[2 3]':0}
        self.maxi = {'[0 1]':0,'[2 3]':0}
        self.q_values = np.array([7.5,2.5,0.75,0.25])#np.zeros((self.env.n_states))
        
    #@partial(jax.jit,static_argnums=0)
    def forward(self,obs):
        q_val = self.q_values[np.array(obs)]
        q_val_temp = q_val/self.temp
        action_probs = jax.nn.softmax(q_val_temp,axis=1)
        actions = sample_batch_index(next(self.rng),action_probs)
        logprobs = jnp.log(action_probs[np.arange(action_probs.shape[0]),np.array(actions)])
        return actions,logprobs

    def compute_relative(self,r,context):
        if self.maxi[context] - self.mini[context] != 0:
            return (r-self.mini[context])/(self.maxi[context] - self.mini[context])
        return r

    def learn(self,ts):
        (o,a,lp,r,no,d,i) = ts
        chosen_symbol = o[0,a[0]]
        o = o[0]

        #if chosen_symbol == 1:
        #    print('---')
        #    print(chosen_symbol,self.mini[chosen_symbol],r,self.maxi[chosen_symbol])

        #lp = 1

        #Update Mini
        alpha = self.alpha_int
        if r < self.mini[str(o)]:
            alpha = self.alpha_ext
        self.mini[str(o)] += alpha*(r-self.mini[str(o)])#/np.exp(lp)

        #Update Maxi
        alpha = self.alpha_int
        if self.maxi[str(o)] < r:
            alpha = self.alpha_ext
        self.maxi[str(o)] += alpha*(r-self.maxi[str(o)])#/np.exp(lp)

        #Update Q
        relative_r = self.compute_relative(r,str(o))
        self.q_values[chosen_symbol] += self.alpha_q * (relative_r - self.q_values[chosen_symbol])#/np.exp(lp)

        for i in self.mini:
            self.log('mini'+str(i),self.mini[i])
            self.log('maxi'+str(i),self.maxi[i])
        for i in range(len(self.q_values)):
            self.log('qval'+str(i),self.q_values[i])