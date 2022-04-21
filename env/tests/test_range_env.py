import jax.numpy as jnp
from env.range_env import RangeEnv

def test_next_season():
    mini = 0
    maxi = 1
    env = RangeEnv([mini],[maxi],nb_arms=5,season_max_duration=10)
    context = env.next_season()
    assert context.shape == (1,5,2,2)
    #Test probas
    assert jnp.allclose(context[0,:,1,0],1)
    assert jnp.allclose(context[0,:,1,1],0)
    #Test rewards
    assert jnp.all(mini<=context[0,:,0,0])
    assert jnp.all(context[0,:,0,0]<=maxi)
    assert jnp.allclose(context[0,:,0,1],0)

def test_get_obs():
    mini = 0
    maxi = 1
    env = RangeEnv([mini],[maxi],nb_arms=5,season_max_duration=10)
    env.reset()
    o = env._sample_obs()
    assert jnp.allclose(o,jnp.array([0,1,2,3,4]))

def test_step():
    nb_arms = 5
    env = RangeEnv([0,1],[0,2],nb_arms=nb_arms,season_max_duration=10)
    o = env.reset()
    assert jnp.allclose(o,jnp.array([0,1,2,3,4]))
    for i in range(20):
        print(i)
        a = jnp.array([i%nb_arms])
        assert env.season_duration == i%10
        assert env.current_season == i//10
        contexts = env.contexts
        o,r,d,_ = env.step(a)
        assert not(d.item()) and i<19 or i==19 and d.item()
        assert r.item() == contexts[0,i%nb_arms,0,0]