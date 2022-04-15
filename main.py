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
        "agent_config":{
            'temp':1.,
            'alpha_ext':0.02,
            'alpha_int':0.002,
            'alpha_q':0.02,
        },

        "plotter_class":Plotter,
        "plotter_config":{
            'targets':[['qval0','qval1','qval2','qval3'],['maxi[0 1]','mini[0 1]','maxi[2 3]','mini[2 3]']],
            'actualization_rate':0.2,
            },

        "logger_class":Logger,
        "logger_config":{
            "dtlog":0.2,
        },

        "nb_steps":350,
        "name":None,
        "seed":0,
    }


    trainer = Trainer(config)
    trainer.start()

    print("EV",(contexts[:,:,0]*contexts[:,:,1]).sum(axis=2).reshape(-1))
    print("min",trainer.agent.mini)
    print("max",trainer.agent.maxi)
    print("qval",trainer.agent.q_values)