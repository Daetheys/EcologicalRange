from trainer.trainer import Trainer
import multiprocessing as mp

def trainer_process(config,pipe,logger_queue,idx,agent_seed):
    trainer = Trainer(config,idx=idx,logger_queue=logger_queue,agent_seed=agent_seed)
    trainer.init()
    trainer.train()

class TrainerSet:
    def __init__(self,config,logger_queue=None,nb_trainers=2):
        self.config = config
        self.logger_queue = logger_queue
        self.nb_trainers = nb_trainers

    def init(self):
        self.pipes = []
        self.processes = []
        for i in range(self.nb_trainers):
            pipe,pipe2 = mp.Pipe(duplex=True)
            process = mp.Process(target=trainer_process,args=(self.config,pipe2,self.logger_queue,i,self.config['agent_seeds'][i]))
            self.pipes.append(pipe)
            self.processes.append(process)

    def train(self):
        for p in self.processes:
            p.start()
        for p in self.processes:
            p.join()
