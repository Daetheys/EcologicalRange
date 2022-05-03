from env.range_env import RangeEnv
import numpy as np

class ProceduralRangeEnv(RangeEnv):
    def __init__(self,nb_seasons,*args,**kwargs):
        min_range = [0]
        max_range = [1]
        for i in range(nb_seasons):
            if np.random.random()<0.2:
                a = min_range[-1]+np.random.uniform(-100,100)
                b = max_range[-1]+np.random.uniform(-100,100)
                a,b = sorted([a,b])
            else:
                a = min_range[-1]+np.random.uniform(-1,1)
                b = max_range[-1]+np.random.uniform(-1,1)
                a,b = sorted([a,b])
            min_range.append(a)
            max_range.append(b)
        super().__init__(min_range,max_range,*args,**kwargs)
