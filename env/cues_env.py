from env.env import Env
import jax.numpy as jnp
from functools import partial
import jax
import haiku as hk
import gym
from utils.sampling import sample_batch_position


class CuesEnv(Env):
    def __init__(self,contexts,batch_size=1,seed=0):
        self.set_context(contexts)

        self.seed = seed
        self.batch_size = batch_size

        self.rng = hk.PRNGSequence(self.seed)

        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(self.n_contexts*self.n_symbols) for i in range(self.batch_size)])
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for i in range(self.batch_size)])

    def set_context(self,contexts):
        self.contexts = contexts #jnp array [n_contexts,2,n_symbols] -> [[ [rewards,probs] ]]
        self.n_contexts,self.n_symbols,_,self.n_outcomes = self.contexts.shape
        
        self.n_states = self.n_contexts*self.n_symbols
        self.action_dim = 2

        self.__base_obs_array = jnp.tile(jnp.arange(self.n_symbols)[None],(self.batch_size,1))

        
    def reset(self):
        self.current_contexts = jax.random.choice(next(self.rng),jnp.arange(self.n_contexts),shape=(self.batch_size,))
        return self._sample_obs()

    #@partial(jax.jit,static_argnums=0)
    def _sample_obs(self):
        out = self.__base_obs_array + self.current_contexts[:,None]*self.n_symbols
        return out

    def step(self,binary_actions):
        rewards = self._step_rewards(next(self.rng),binary_actions,self.contexts,self.current_contexts,self.batch_size)
        return [None]*self.batch_size,rewards,[True]*self.batch_size,{}

    @partial(jax.jit,static_argnums=0)
    def _step_rewards(self,key,binary_actions,contexts,current_contexts,batch_size):
        rewards_th = contexts[current_contexts,binary_actions,0]
        probs = contexts[current_contexts,binary_actions,1]
        positions = sample_batch_position(key,probs)
        rewards = jnp.sum(positions*rewards_th,axis=1)
        return rewards
