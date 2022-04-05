from utils.sampling import *
import haiku as hk
import jax.numpy as jnp
import numpy as np


def test_sample_batch_position():
    rng = hk.PRNGSequence(42)
    prd_array = jax.random.uniform(next(rng),shape=(10,5))
    probas = prd_array/(prd_array.sum(axis=1)[:,None])
    
    nb = 300
    stack = jnp.zeros((10,5))
    for i in range(nb):
        stack += sample_batch_position(next(rng),probas)/nb
    
    eps = 0.2
    assert jnp.allclose((stack-probas)**2<eps,1)

def test_sample_batch_index():
    shape = (7,5)
    rng = hk.PRNGSequence(42)
    prd_array = jax.random.uniform(next(rng),shape=shape)
    probas = prd_array/(prd_array.sum(axis=1)[:,None])

    def to_one_hot(a,shape):
        arr = np.zeros(shape,dtype=np.int8)
        a = np.array(a,dtype=np.int8)
        arr[np.arange(shape[0]),np.array(a)] = 1
        return jnp.array(arr)
    
    nb = 300
    stack = jnp.zeros(shape)
    for i in range(nb):
        actions = sample_batch_index(next(rng),probas)
        stack += to_one_hot(actions,shape)/nb
    
    eps = 0.2
    assert jnp.allclose((stack-probas)**2<eps,1)