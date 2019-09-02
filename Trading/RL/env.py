import numpy as np
import pandas as pd

settings = {
            "episode_length" : 7,
            "state_length"   : 32
            }

class databank:
    def __init__(self,file="LUPE.ST.csv", episode_length=settings["episode_length"], state_length=settings["episode_length"]):
        #####
        ### Data from yahoo-file, NaNs removed:
        #  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.state_len = settings["state_length"]
        self.ep_len = episode_length
        self.data = pd.read_csv(file, delimiter=',').dropna()
        self.mean_vol = self.data["Volume"].mean()
        self.reset()

    def reset(self):
        self.t = np.random.randint(self.state_len, high=self.data.shape[0] - self.ep_len)
    def step(self):
        if self.t < self.end:
            self.t += 1
            return True
        return False
    def statemaker(self, s):
        components = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        ret = s.copy(deep=True)[components]
        mean_close = ret["Close"].mean()
        for c in components:
            if c != "Volume":
                ret[c] = ret[c] / mean_close
        ret["Volume"] = ret["Volume"] / self.mean_vol
        return ret
    def get_state(self):
        return self.statemaker( self.data.iloc[self.t-self.state_len:self.t] )
    @property
    def end(self):
        return self.data.shape[0]
    @property
    def price(self):
        return self.data.iloc[self.t]["Close"]

class account:
    def __init__(self):
        self.price = 10 # price of stock
        self.stock_count = 100 # n stock owned
        self.cash = 10 * self.price # initial cash

    def reset(self):
        pass

    def buy(self, n):
        if self.cash < n * price:
            return
        self.cash -= self.price * n
        self.stock_count += n
        return
    def sell(self, n):
        m = min(n,self.stock_count)
        self.cash += m*price
        self.stock_count -= m

    def get_state(self):
        return np.array([self.cash, self.stock_count])
    @property
    def value(self):
        return self.cash + self.price * self.stock_count

class env:
    def __init__(self):
        self.market = databank()
        self.customer = account()
        self.reset()
    def reset(self):
        self.customer.reset()
        self.market.reset()
        self.t = 0
        return self.get_state()
    def get_state(self):
        return {"customer" : self.customer.get_state(), "market" : self.market.get_state()}
    def perform_action(self, a):
        val = self.customer.value
        if a == 0: #wait
            pass
        elif a == 1: #buy
            self.customer.price = self.market.price
            self.customer.buy(1)
        elif a == 2: #sell
            self.customer.price = self.market.price
            self.customer.sell(1)
        self.t += 1
        self.market.step()
        r = (self.customer.value - val) / self.market.price
        d = False if t < settings["episode_length"] else True
        return r, self.get_state, d
