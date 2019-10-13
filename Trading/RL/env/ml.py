import numpy as np
from base import *

#Creates a dataset from a file. Targets is a vector of the close priced averaged over different time ranges.
class data:
    def __init__(self, file=None, predict_horizon=10, state_length=128, training=False):
        self.state_len = state_length
        self.predict_horizon = predict_horizon if training else 0
        self._d = databank(file=file, state_length=state_length, episode_length=predict_horizon)
        #create data:
        components = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        n = len(self._d) - state_length - predict_horizon
        shape = self._d.get_state()[components].values.shape
        self.data = np.zeros((n, *shape))
        self.targets = np.zeros((len(self), predict_horizon))
        print("Generating data from file:", file)
        for i in range(n):
            chunk = self._d.data.iloc[i:i+self.state_len][components].copy(deep=True)
            norm_price  = chunk["Close"].mean()
            norm_vol    = chunk["Volume"].mean()
            for x in chunk:
                if x == "Volume":
                    chunk[x] = chunk[x] / norm_vol
                else:
                    chunk[x] = chunk[x] / norm_price
            if training:
                self.targets[i] = self.create_target(i) / norm_price
            self.data[i] = chunk.values
            if i%30 == 0:
                print(i,"/",n)
    def create_target(self, idx):
        close = self._d.data.iloc[idx+self.state_len:idx+self.state_len+self.predict_horizon]["Close"].values
        ret = np.zeros((self.predict_horizon))
        for i in range(self.predict_horizon):
            ret[i] = close[:i+1].mean()
        return ret
    def __len__(self):
        return len(self._d) - self.predict_horizon - self.state_len
    @property
    def x(self):
        return self.data
    @property
    def y(self):
        return self.targets

if __name__ == "__main__":
    d = data(file="LUPE.ST.csv", training=True)
    print(d.x[0], d.y[0])
