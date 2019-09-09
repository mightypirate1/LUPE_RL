import numpy as np
import pandas as pd

settings = {
            "episode_length" : 128,
            "state_length"   : 128
            }

class databank:
    def __init__(self,file="LUPE.ST.csv", episode_length=settings["episode_length"], state_length=settings["episode_length"]):
        #####
        ### Data from yahoo-file, NaNs removed:
        #  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.state_len = settings["state_length"]
        self.ep_len = episode_length
        self.data = pd.read_csv(file, delimiter=',').dropna()
        self.reset()

    def reset(self):
        self.t = np.random.randint(self.state_len, high=self.data.shape[0] - self.ep_len)
        self.mean_vol = self.data["Volume"].mean()
        self.init_price = self.data.iloc[self.t]["Open"]

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
        return self.statemaker( self.data.iloc[self.t-self.state_len:self.t]) / self.mean_price

    @property
    def end(self):
        return self.data.shape[0]

    @property
    def price(self):
        return self.data.iloc[self.t]["Close"]

    @property
    def mean_price(self):
        return self.data.iloc[self.t-settings["state_length"]:self.t]["Close"].mean()

class account:
    def __init__(self, market):
        self.market = market
        self.reset()

    def reset(self, max_stock=10, max_cash=10):
        #reset account at a given price. you get unif(1,max_stock) stocks, and cash enough to buy another unif(1,max_cash)
        price = self.init_price = self.market.price # price of stock
        self.stock_count = np.random.randint(1, high=max_stock+1)
        self.cash = np.random.randint(1, high=max_cash) * self.market.price

    def buy(self, n):
        if self.cash < n * self.market.price:
            return
        self.cash -= self.market.price * n
        self.stock_count += n
        return
    def sell(self, n):
        m = min(n,self.stock_count)
        self.cash += self.market.price * m
        self.stock_count -= m

    def get_state(self):
        return np.array([self.cash / self.market.price, self.stock_count ])

    @property
    def value(self):
        return self.cash + self.market.price * self.stock_count

class env:
    def __init__(self):
        self.market = databank()
        self.customer = account(self.market)
        self.reset()
    def reset(self):
        self.market.reset()
        self.customer.reset()
        self.t = 0
        return self.get_state()
    def get_state(self):
        return {"customer" : self.customer.get_state(), "market" : self.market.get_state()}
    def perform_action(self, a):
        val = self.customer.value
        self.customer.price = self.market.price

        if a == 0: #wait
            pass
        elif a == 1: #buy
            self.customer.buy(1)
        elif a == 2: #sell
            self.customer.sell(1)

        self.t += 1
        self.market.step()
        r = (self.customer.value - val) / self.market.price
        d = False if self.t < settings["episode_length"] else True
        return r, self.get_state(), d
