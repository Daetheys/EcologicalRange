from agent.range_agent import RangeAgent
from env.cues_env import CuesEnv
from env.range_env import RangeEnv
from agent.q_agent import QAgent
from logger.logger import Logger
from plotter.plotter import Plotter
from trainer.trainer import Trainer

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

    config['env_class'] = RangeEnv
    config['env_config'] = {
        "min_range":[0,0,0,0,0,0,5,5,0,0,10]*3,
        "max_range":[1,10,10,15,15,10,10,10,15,15,15]*3,
        "nb_arms":20,
        "season_max_duration":10
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
        'targets':[],
        'actualization_rate':0.2,
        'plots':[]
            }

    config['logger_class'] = Logger
    config['logger_config'] = {
            "dtlog":0.2,
        }

    config['nb_steps'] = 1000
    config['name'] = None
    config['seed'] = 0


    trainer = Trainer(config)

    trainer.init()

    from plotter.plots.reward_plot import RewardPlot
    from plotter.plots.range_agent_plot import RangeAgentPlot
    from plotter.plots.range_env_plot import RangeEnvPlot
    from plotter.plots.action_plot import ActionPlot

    config['plotter_config']['plots'].append(RewardPlot)
    config['plotter_config']['plots'].append(RangeAgentPlot)
    config['plotter_config']['plots'].append(RangeEnvPlot)
    config['plotter_config']['plots'].append(ActionPlot)

    config['plotter_config']['targets'].append(['qval_'+str(i) for i in range(len(trainer.agent.q_values))])
    config['plotter_config']['targets'].append(['mini_'+str(k) for k in trainer.agent.mini]+['maxi_'+str(k) for k in trainer.agent.mini])
    config['plotter_config']['targets'].append(['env_mini','env_maxi'])
    config['plotter_config']['targets'].append(['reward','EV'])
    config['plotter_config']['targets'].append(['action'])

    trainer.start_logger()
    trainer.start_plotter()

    trainer.start()

    #print("EV",(contexts[:,:,0]*contexts[:,:,1]).sum(axis=2).reshape(-1))
    #print("min",trainer.agent.mini)
    #print("max",trainer.agent.maxi)
    #print("qval",trainer.agent.q_values)
