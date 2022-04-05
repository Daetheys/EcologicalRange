import jax.numpy as jnp
import jax

@jax.jit
def sample_batch_position(key,probs):
    batch_size = probs.shape[0]
    rd = jax.random.uniform(key,shape=(batch_size,1))
    cumprobs = jnp.cumsum(probs,axis=1)
    padded_cumprobs = jnp.concatenate([jnp.zeros((batch_size,1)),cumprobs],axis=1)
    realisations = jnp.float32(rd>padded_cumprobs)
    positions = realisations[:,:-1] - realisations[:,1:]
    return positions

@jax.jit
def sample_batch_index(key,probs):
    batch_size,nb_outcomes = probs.shape
    positions = sample_batch_position(key,probs)
    indexs =  jnp.tile(jnp.arange(nb_outcomes)[None],(batch_size,1))
    return jnp.array(jnp.sum(positions*indexs,axis=1),dtype=jnp.int8)
