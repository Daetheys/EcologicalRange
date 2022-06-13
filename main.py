from agent.range_agent import RangeAgent
from env.cues_env import CuesEnv
from env.range_env import RangeEnv
from env.procedural_range_env import ProceduralRangeEnv
from agent.q_agent import QAgent
from agent.ac_agent import ACAgent
from logger.logger import Logger
from plotter.plotter import Plotter
from trainer.trainer_set import TrainerSet

from plotter.plots.reward_plot import RewardPlot
from plotter.plots.range_agent_plot import RangeAgentPlot
from plotter.plots.range_env_plot import RangeEnvPlot
from plotter.plots.action_plot import ActionPlot
from plotter.plots.actionprob_plot import ActionProbPlot

import uuid
import os


def main(config):
    name = config['name']
    if name is None:
        name = str(uuid.uuid4())

    file_path = os.path.join('TRAININGS',name)

    logger = None
    logger_queue = None
    if config['logger_class']:
        #Create directory
        logger_file_path = os.path.join(file_path,'logs')
        os.makedirs(logger_file_path)
        #Start logger
        logger = config['logger_class'](logger_file_path,**config['logger_config'])
        logger.init()
        logger_queue = logger.queue
        logger.start()

    if config['plotter_class']:
        #Start plotter
        logger_file_path = os.path.join(file_path,'logs')
        plotter = config['plotter_class'](logger_file_path,**config['plotter_config'])
        plotter.start()

    trainer = TrainerSet(config,logger_queue=logger_queue,nb_trainers=config['nb_trainers'])
    trainer.init()

    print('Starting Training : ',name)
    trainer.train()
    print('Training Finished : ',name)  

    if logger:
        logger.stop()

if __name__ == '__main__':

    config = {}

    """
    #Cues Env contexts
    import jax.numpy as jnp
    action00 = jnp.array([[10,0],[.75,0.25]])
    action01 = jnp.array([[10,0],[.25,0.75]])
    context1 = jnp.array([action00,action01])

    action10 = jnp.array([[1,0],[.75,0.25]])
    action11 = jnp.array([[1,0],[.25,0.75]])
    context2 = jnp.array([action10,action11])

    contexts = jnp.array([context1,context2])

    config['env_class'] = CuesEnv
    config['env_config'] = {
            "contexts":contexts
            }
    """

    """
    config['env_class'] = RangeEnv
    config['env_config'] = {
        "min_range":[0,0,0,90,90,90,40,40,40],
        "max_range":[1,1,1,100,100,100,50,50,50],
        "nb_arms":200,
        "season_max_duration":50
        }
    """

    """
    config['env_class'] = RangeEnv
    config['env_config'] = {
        "min_range":[-10*40 for i in range(-10,100)],
        "max_range":[i*40 for i in range(-10,100)],
        "nb_arms":20,
        "season_max_duration":10
        }
    """
    
    """
    config['env_class'] = RangeEnv
    config['env_config'] = {
        "min_range":[0],
        "max_range":[1],
        "nb_arms":20,
        "season_max_duration":200
        }
    """

    
    config['env_class'] = ProceduralRangeEnv
    config['env_config'] = {
        "nb_seasons":30,
        "nb_arms":20,
        "season_max_duration":10,
        "seed":0,
    }
# 1 - Vanilla QL
# 2 - QL + memory Q
# 3 - Range
# 4 - QL + memory ?

    
    config['agent_class'] = QAgent#RangeAgent#ACAgent#QAgent
    config['agent_config'] = {
        'temp':1/8,
        'beta':0.5,
        'alpha_ext':0.5,
        'alpha_int':0.05,
        'alpha_v':0.5,
        'alpha_q':0.5,
        'informed':'memory', #cheat : knows the exact value / 'memory' : keeps the value from previous season / False : not informed
        'scaled_beta':'memory', #cheat : knows the exact range / 'memory' : remember the range from the previous season / False : not informed
    }

    config['plotter_class'] = Plotter
    config['plotter_config'] = {
        'actualization_rate':0.2,
        'plots':[RewardPlot,RangeAgentPlot,ActionProbPlot]
    }

    config['logger_class'] = Logger
    config['logger_config'] = {
            "dtlog":0.2,
    }

    config['nb_steps'] = 10000
    config['name'] = None
    config['nb_trainers'] = 1
    config['agent_seeds'] = [i+101 for i in list(range(config['nb_trainers']))]


    main(config)
