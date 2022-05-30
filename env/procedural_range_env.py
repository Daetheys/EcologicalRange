from env.range_env import RangeEnv
import numpy as np

class ProceduralRangeEnv(RangeEnv):
    def __init__(self,nb_seasons,*args,**kwargs):
        def next_season(mini,maxi,modes=[0,1,2,3,5,6,7,8],p=0.5):
            mode = None
            a=mini
            b=maxi
            if np.random.random()<p:
                if len(modes)==0:
                    modes.append(4)
                mode = np.random.choice(modes)
                mini_mode,maxi_mode = mode%3-1,mode//3-1
                print(mode,mini_mode,maxi_mode)
                a2 = min(a,b)+mini_mode*np.random.uniform(70,120)
                b2 = max(b,a)+maxi_mode*np.random.uniform(70,120)
                (a,b) = (a2,b2)

                if mini_mode + maxi_mode == 2:
                    d = b-a
                    a+=d
                    b+=d

                if mini_mode + maxi_mode == -2:
                    d = b-a
                    a-=d
                    b-=d
                
            a = a+np.random.uniform(-1,1)
            b = b+np.random.uniform(-1,1)
            a,b = sorted([a,b])
            return a,b,mode
                
        min_range = [0]
        max_range = [1]
        modes = [0,1,2,3,5,6,7,8]
        for i in range(nb_seasons):
            p = 0.3
            if nb_seasons-i-1 <= len(modes):
                p = 1
            a,b,mode = next_season(min_range[-1],max_range[-1],modes=modes,p=p)
            if mode !=4 and not(mode is None):
                print(mode,modes)
                modes.remove(mode)
            min_range.append(a)
            max_range.append(b)
        super().__init__(min_range,max_range,*args,**kwargs)
