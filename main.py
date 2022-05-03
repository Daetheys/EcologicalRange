from agent.range_agent import RangeAgent
from env.cues_env import CuesEnv
from env.range_env import RangeEnv
from env.procedural_range_env import ProceduralRangeEnv
from agent.q_agent import QAgent
from logger.logger import Logger
from plotter.plotter import Plotter
from trainer.trainer_set import TrainerSet

from plotter.plots.reward_plot import RewardPlot
from plotter.plots.range_agent_plot import RangeAgentPlot
from plotter.plots.range_env_plot import RangeEnvPlot
from plotter.plots.action_plot import ActionPlot

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
        "min_range":[0,0,0,0,0,0,5,5,0,0,10]*3,
        "max_range":[1,10,10,15,15,10,10,10,15,15,15]*3,
        "nb_arms":20,
        "season_max_duration":10
        }
    """

    config['env_class'] = ProceduralRangeEnv
    config['env_config'] = {
        "nb_seasons":30,
        "nb_arms":20,
        "season_max_duration":10,
        "seed":0,
        }

    config['agent_class'] = RangeAgent
    config['agent_config'] = {
            'temp':1.,
            'alpha_ext':0.5,
            'alpha_int':0.05,
            'alpha_q':0.5,
        }

    config['plotter_class'] = Plotter
    config['plotter_config'] = {
        'actualization_rate':0.2,
        'plots':[RewardPlot,RangeAgentPlot]
            }

    config['logger_class'] = Logger
    config['logger_config'] = {
            "dtlog":0.2,
        }

    config['nb_steps'] = 1000
    config['name'] = None
    config['nb_trainers'] = 2
    config['agent_seeds'] = [i for i in list(range(config['nb_trainers']))]


    main(config)
