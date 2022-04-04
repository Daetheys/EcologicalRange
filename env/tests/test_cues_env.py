from env.cues_env import CuesEnv
import jax.numpy as jnp
import numpy as np

def test_init():
    action00 = jnp.array([[1,0],[.75,0.25]])
    action01 = jnp.array([[10,0],[.75,0.25]])
    context1 = jnp.array([action00,action01])

    action10 = jnp.array([[1,0],[.75,0.25]])
    action11 = jnp.array([[10,0],[.75,0.25]])
    context2 = jnp.array([action00,action01])

    contexts = jnp.array([context1,context2])
    
    env = CuesEnv(contexts,batch_size=10)
    
    assert env.n_contexts == 2
    assert env.n_symbols == 2

def test_reset():
    action00 = jnp.array([[1,0],[.75,0.25]])
    action01 = jnp.array([[10,0],[.75,0.25]])
    context1 = jnp.array([action00,action01])

    action10 = jnp.array([[1,0],[.75,0.25]])
    action11 = jnp.array([[10,0],[.75,0.25]])
    context2 = jnp.array([action00,action01])

    contexts = jnp.array([context1,context2])
    
    env = CuesEnv(contexts,batch_size=10)
    obs = env.reset()

    assert obs.shape == (env.batch_size,env.n_symbols)

    #Verify contexts are either zeros or ones
    zeros = jnp.isclose(env.current_contexts,0)
    ones = jnp.isclose(env.current_contexts,1)
    assert jnp.allclose(jnp.logical_or(zeros,ones),1)

    #Verify observations are [[i,i+1]...]
    assert jnp.allclose(obs[:,1]-obs[:,0],1)
    zeros = jnp.isclose(obs[:,0],0)
    twos = jnp.isclose(obs[:,0],2)
    assert jnp.allclose(jnp.logical_or(zeros,twos),1)

def test_step():
    action00 = jnp.array([[1,0],[.75,0.25]])
    action01 = jnp.array([[10,0],[.75,0.25]])
    context1 = jnp.array([action00,action01])

    action10 = jnp.array([[1,0],[.75,0.25]])
    action11 = jnp.array([[10,0],[.75,0.25]])
    context2 = jnp.array([action00,action01])
    
    contexts_th = jnp.array([context1,context2])
    
    env = CuesEnv(contexts_th,batch_size=100)
    obs = env.reset()

    contexts = []
    rews = []
    acts = []
    for i in range(100):
        a = jnp.array(env.action_space.sample())
        nobs,rew,done,info = env.step(a)
        
        assert rew.shape == (env.batch_size,)
        
        assert nobs == [None]*env.batch_size
        assert done == [True]*env.batch_size
        assert info == {}

        contexts.append(obs)
        rews.append(rew)
        acts.append(a)

        obs = env.reset()

    contexts = jnp.concatenate(contexts)
    acts = jnp.concatenate(acts)
    rews = jnp.concatenate(rews)

    means = np.zeros((env.n_contexts,env.n_symbols))
    counts = np.zeros((env.n_contexts,env.n_symbols)) 
    for o,a,r in zip(contexts,acts,rews):
        means[o//env.n_contexts,a] += r
        counts[o//env.n_contexts,a] += 1
    means /= counts

    means_th = jnp.sum(contexts_th[:,:,0,:]*contexts_th[:,:,1,:],axis=2)

    print("If both are close it's fine")
    print(means)
    print(means_th)
    
    eps = 0.25
    assert jnp.allclose(jnp.abs(means_th-means)<eps,1)
    
