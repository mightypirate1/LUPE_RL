import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

settings = {
            "episode_length" : 128,
            "state_length"   : 128
            }

class databank:
    def __init__(self,file=None, interactive=False, episode_length=settings["episode_length"], state_length=settings["episode_length"]):
        self.state_len = settings["state_length"]
        self.ep_len = episode_length
        #####
        ### Data from yahoo-file, NaNs removed:
        #  ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if file is not None:
            self.data = pd.read_csv(file, delimiter=',').dropna()
            self.reset()
        else:
            start_date = '2001-09-06'
            stop_date = dt.datetime.now().strftime("%Y-%m-%d")
            print("Downloading LUPE: ", start_date, " -> ", stop_date)
            self.data = yf.download('LUPE.ST', start_date, stop_date).dropna()
            if interactive:
                print("Last data:")
                print(self.data.iloc[-1])
                open, high, low, close, adj_close, volume = input("open high low close, adj_close, volume = ").split(" ")
                # open, high, low, close, adj_close, volume = 1,2,3,4,5,6
                today_data = [{'name':stop_date+" 00:00:00",
                                    'Open' : float(open),
                                    'High' : float(high),
                                    'Low' : float(low),
                                    'Close' : float(close),
                                    'Adj Close' : float(adj_close),
                                    'Volume' : float(volume),
                                    },]
                print("Adding entry:")
                source_df = pd.DataFrame(today_data).set_index('name')
                self.data = self.data.append(source_df)
                self.reset_to_current()
            else:
                self.reset()

    def reset_to_current(self):
        self._reset(self.data.shape[0]-1)

    def _reset(self,t):
        self.end = t + self.ep_len
        self.t = t
        self.mean_vol = self._mean_vol(t - self.state_len, t)
    def reset(self):
        t = np.random.randint(self.state_len, high=self.data.shape[0] - self.ep_len)
        self._reset(t)

    def step(self):
        if self.t < self.end:
            self.t += 1
            return True
        return False

    def _mean_vol(self, start, stop):
        return self.data.iloc[start : stop]["Volume"].mean()

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
    def price(self):
        return self.market.price
    @property
    def value(self):
        return self.cash + self.market.price * self.stock_count

class env:
    def __init__(self, file=None, interactive=False):
        self.action_dict = {0 : "wait", 1 : "buy", 2 : "sell"}
        self.market = databank(file=file, interactive=interactive)
        self.customer = account(self.market)
        self.reset()
    def _reset(self, t):
        self.market._reset(t)
        self.customer.reset()
        return self.get_state()
    def reset(self):
        self.market.reset()
        self.customer.reset()
        return self.get_state()
    def get_state(self):
        return {"customer" : self.customer.get_state(), "market" : self.market.get_state()}
    def perform_action(self, a):
        stock_count, cash, price = self.customer.stock_count, self.customer.cash, self.market.price
        value_0 = self.customer.value
        if self.action_dict[a[0]] == "wait":
            pass
        elif self.action_dict[a[0]] == "buy":
            self.customer.buy(1)
        elif self.action_dict[a[0]] == "sell":
            self.customer.sell(1)
        self.market.step()
        _r = self.customer.value - value_0
        r = np.array( [ _r ] )
        d = np.array([False if self.t < self.market.end else True])
        return r, self.get_state(), d

    @property
    def state_len(self):
        return self.market.state_len

    @property
    def ep_len(self):
        return self.market.ep_len

    @property
    def t(self):
        return self.market.t

    @property
    def actions(self):
        return self.action_dict.values()
