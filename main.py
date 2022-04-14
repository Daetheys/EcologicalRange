from agent.range_agent import RangeAgent
from env.cues_env import CuesEnv
from logger.logger import Logger
from logger.plotter import Plotter
from trainer.trainer import Trainer

if __name__ == '__main__':

    #Cues Env contexts
    import jax.numpy as jnp
    action00 = jnp.array([[10,0],[.75,0.25]])
    action01 = jnp.array([[10,0],[.25,0.75]])
    context1 = jnp.array([action00,action01])

    action10 = jnp.array([[1,0],[.75,0.25]])
    action11 = jnp.array([[1,0],[.25,0.75]])
    context2 = jnp.array([action10,action11])

    contexts = jnp.array([context1,context2])

    config = {
        "env_class":CuesEnv,
        "env_config":{
            "contexts":contexts
            },

        "agent_class":RangeAgent,
        "agent_config":{},

        "plotter_class":Plotter,
        "plotter_config":{
            'targets':['reward'],
            'actualization_rate':0.2,
            },

        "logger_class":Logger,
        "logger_config":{
            "dtlog":0.2,
        },

        "nb_steps":1000,
        "name":None,
        "seed":1,
    }


    trainer = Trainer(config)
    trainer.start()

    print("EV",(contexts[:,:,0]*contexts[:,:,1]).sum(axis=2).reshape(-1))
    print("min",trainer.agent.mini)
    print("max",trainer.agent.maxi)
    print('delta',trainer.agent.maxi-trainer.agent.mini)
    print("qval",trainer.agent.q_values)