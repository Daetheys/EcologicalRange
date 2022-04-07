from agent.range_agent import *
from env.test_env import *
import gym

from agent.agent import Timestep

def test_range_agent_reset():
    n_states = 8
    act_dim = 2
    batch_size = 1
    obs_space = gym.spaces.Tuple([gym.spaces.Discrete(n_states) for i in range(batch_size)])
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for i in range(batch_size)])
    env = TestEnv(obs_space,act_space,n_states,act_dim)

    agent = RangeAgent(env)
    agent.reset()
    
    assert np.allclose(agent.mini,0)
    assert np.allclose(agent.maxi,0)
    assert np.allclose(agent.q_values,0)

def test_range_agent_forward():
    n_states = 8
    act_dim = 2
    batch_size = 1
    obs_space = gym.spaces.Tuple([gym.spaces.MultiDiscrete([n_states]*2) for i in range(batch_size)])
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for i in range(batch_size)])
    env = TestEnv(obs_space,act_space,n_states,act_dim)

    agent = RangeAgent(env)
    agent.reset()

    agent.q_values = np.arange((agent.env.n_states))
    
    o = env.reset()
    a,lp = agent.forward(o)
    
    assert a.shape == (o.shape[0],)
    assert lp.shape == (o.shape[0],)

def test_range_agent_learn():
    n_states = 8
    act_dim = 2
    batch_size = 1
    obs_space = gym.spaces.Tuple([gym.spaces.MultiDiscrete([n_states]*2) for i in range(batch_size)])
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for i in range(batch_size)])
    env = TestEnv(obs_space,act_space,n_states,act_dim)

    agent = RangeAgent(env)
    agent.reset()
    
    o = env.reset()
    a,lp= agent.forward(o)
    no,r,d,i = env.step(a)

    ts = Timestep(o,a,lp,r,no,d,i)

    agent.learn(ts)

from env.cues_env import CuesEnv
def test_agent_cuesenv():
    action00 = jnp.array([[1,0],[.75,0.25]])
    action01 = jnp.array([[10,0],[.75,0.25]])
    context1 = jnp.array([action00,action01])

    action10 = jnp.array([[-1,0],[.75,0.25]])
    action11 = jnp.array([[-10,0],[.75,0.25]])
    context2 = jnp.array([action10,action11])

    contexts = jnp.array([context1,context2])

    env = CuesEnv(contexts,batch_size=1,seed=1)

    agent = RangeAgent(env,seed=1)

    agent.train(1000)

    print(agent.mini)
    print(agent.maxi)
    print(agent.q_values)
    assert False
