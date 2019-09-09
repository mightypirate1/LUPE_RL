import random
from collections import deque
import numpy as np
import scipy.stats

class prioritized_experience_replay:
    def __init__(self, max_size=None, state_size=None):
        self.log        = log
        self.max_size   = max_size
        self.state_size = state_size
        self.states      =  np.zeros((max_size,*state_size[1:]))
        self.s_primes    =  np.zeros((max_size,*state_size[1:]))
        self.rewards     =  np.zeros((max_size,1))
        self.actions     =  np.zeros((max_size,1), dtype=np.uint8)
        self.dones       =  np.zeros((max_size,1), dtype=np.bool)
        self.prios       = -np.ones((max_size,1))
        self.current_size  = 0
        self.current_idx   = 0
        self.total_samples = 0

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove=False):
        #Create the sampling distribution (see paper for details)
        n = self.current_size
        # all_indices = np.arange(n)
        all_indices = np.arange(n)
        sort_idxs = self.prios[:n].argsort(axis=0).ravel()

        #make ranking
        rank = 1+n-scipy.stats.rankdata(self.prios[:n].ravel(), method='ordinal')
        #make a ranking based probability disribution (pareto-ish)
        one_over_rank = 1/rank #Rank-based sampling
        p_unnormalized = one_over_rank**alpha
        p = p_unnormalized / p_unnormalized.sum() #sampling distribution done
        is_weights_unnormalized = ((n*p)**(-beta))[:,np.newaxis] #Compute the importance sampling weights
        is_weights_all = is_weights_unnormalized/is_weights_unnormalized.reshape(-1).max()

        #Sample indices to make a batch out of!
        idx_dict = {}
        indices = np.random.choice(all_indices, replace=True, size=n_samples, p=p).tolist()

        ##Data collection, and index-tracking
        for i, idx in enumerate(indices):
            idx_dict[idx]         = i #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample

        #Return values!
        filter = idx_dict
        is_weights = is_weights_all[indices]
        data = (
                self.states[indices,:],
                self.actions[indices,:],
                self.rewards[indices,:],
                self.s_primes[indices,:],
                self.dones[indices,:],
                )
        return data, is_weights, filter

    def add_samples(self, data, prio):
        s, sp,a,r,d = data
        n = prio.size
        idxs = [x%self.max_size for x in range(self.current_idx, self.current_idx+n)]
        self.states[idxs,:] = s
        self.actions[idxs,:]  = a
        self.rewards[idxs,:]  = r
        self.s_primes[idxs,:] = sp
        self.dones[idxs,:]    = d
        self.prios[idxs,:]    = prio
        self.current_idx += n
        self.current_size = min(self.current_size+n, self.max_size)
        self.total_samples += n

    def update_prios(self, new_prios, filter):
        self.prios[list(filter.keys()),:] = new_prios[list(filter.values()),:]

    def __len__(self):
        return self.current_size

class experience_replay_buffer:
    def add(self, experience):
        self.data.append(experience)
    def get_sample(self, n):
        return random.sample(self.data, n)
    def __init__(self, max_size=0):
        self.data = deque(maxlen=max_size)
    def __len__(self):
        return self.data.__len__()
