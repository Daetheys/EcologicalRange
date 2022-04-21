from env.cues_env import CuesEnv
import jax
import jax.numpy as jnp
import gym
import haiku as hk

class RangeEnv(CuesEnv):
    """ Only works with batch_size = 1"""

    def __init__(self,min_range,max_range,*args,nb_arms=20,season_max_duration=10,seed=0,**kwargs):
        self.min_range = min_range
        self.max_range = max_range
        self.nb_arms = nb_arms

        self.season_max_duration = season_max_duration

        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(self.nb_arms) for i in range(1)])
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(self.nb_arms) for i in range(1)])

        self.symbols = [jnp.arange(nb_arms)]
        self.n_symbols = nb_arms
    
        self.current_season = -1
        self.season_duration = 0

        self.seed = seed

        self.rng = hk.PRNGSequence(self.seed)

        self.batch_size = 1

        #first_context = self.next_season()

        #super().__init__(first_context,*args,seed=seed,**kwargs)

    def next_season(self):
        #Sample rewards for each arm uniformly in the given range
        self.season_duration = 0
        self.current_season += 1
        rewards = jax.random.uniform(next(self.rng),shape=(self.nb_arms,),minval=self.min_range[self.current_season],maxval=self.max_range[self.current_season])
        zeros = jnp.zeros(rewards.shape)
        possible_rewards = jnp.concatenate([rewards[:,None,None],zeros[:,None,None]],axis=2)

        probsA = jnp.zeros((self.nb_arms,1,1))+1
        probsB = jnp.zeros((self.nb_arms,1,1))
        probs = jnp.concatenate([probsA,probsB],axis=2)

        actions = jnp.concatenate([possible_rewards,probs],axis=1)


        self.set_context(jnp.array([actions]))


    def _reset(self):
        self.current_season = -1
        self.season_duration = 0

        self.next_season()

    def reset(self):
        self._reset()
        return super().reset()

    def step(self,a):
        _,r,_,i = super().step(a)

        self.season_duration += 1
        
        d = jnp.array([False])
        if self.season_duration==self.season_max_duration:
            d = jnp.array([True])

        o = super().reset()

        return o,r,d,i