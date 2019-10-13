import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

class databank:
    def __init__(self,file=None, interactive=False, episode_length=128, state_length=128, eval=False):
        self.state_len = state_length
        self.ep_len = episode_length
        self.eval = eval
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
        self.init_val = self.price
        self.last_price = self.price

    def reset(self):
        t = np.random.randint(self.state_len, high=len(self) - self.ep_len)
        self._reset(t)

    def step(self):
        self.last_price = self.price
        if self.t < self.end:
            self.t += 1
            return False
        return True

    def _mean_vol(self, start, stop):
        return float(self.data.iloc[start : stop]["Volume"].mean())

    def statemaker(self, s):
        components = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        ret = s[components].copy(deep=True)
        ret["w_price"] = self.mean_price
        ret["w_vol"]   = self.mean_vol
        return ret

    def get_state(self):
        return self.statemaker( self.data.iloc[self.t-self.state_len:self.t])

    def __len__(self):
        return self.data.shape[0]
    @property
    def delta(self):
        return self.price - self.last_price
    @property
    def price(self):
        return self.data.iloc[self.t]["Close"]
    @property
    def mean_price(self):
        return np.array(self.data.iloc[self.t-self.state_len:self.t]["Close"].mean())
    @property
    def normalized_price(self):
        return self.price / self.init_val
    @property
    def normalized_delta(self):
        return self.delta / self.init_val

class account:
    class stock:
        def __init__(self, price, type='LUPE.ST'):
            self.price = float(price)
            self.type = type

    def __init__(self, market, max_init_stock=5, max_init_cash=10):
        self.max_init_cash, self.max_init_stock = max_init_cash, max_init_stock
        self.market = market
        self.reset()

    def reset(self, init_stock=None, init_cash=None):
        #reset account at a given price. you get unif(1,init_stock) stocks, and cash enough to buy another unif(1,init_cash)
        self.stocks = []
        if init_cash is None:
            self.cash = 0
            if self.max_init_cash > 0:
                self.cash = np.random.randint(0, high=self.max_init_cash+1) * self.market.price
        else:
            self.cash = init_cash * self.market.price
        if init_stock is None:
            if self.max_init_stock > 0:
                self.buy(np.random.randint(0, high=self.max_init_stock+1), free=True)
        else:
            self.buy(init_stock, free=True)
        self.init_val = float(self.value)

    def step(self):
        return False

    def wait(self):
        return 0.0

    def buy(self, n, free=False):
        if not free and self.cash < n * self.market.price:
            return 0.0
        for _ in range(n):
            self.stocks.append( self.stock( self.market.price ) )
        if not free:
            self.cash -= self.market.price * n
        return 0.0

    def sell(self, n):
        sell_value = 0
        avg_buy_price = 0
        m = min(n,self.stock_count)
        self.cash += self.market.price * m
        for i, stock in enumerate(reversed(self.stocks)):
            avg_buy_price += stock.price / self.stock_count
            if i < m:
                sell_value += self.market.price
        self.stocks = self.stocks[:-m]
        profit = sell_value - m * avg_buy_price
        return  float(profit)

    def get_state(self):
        return np.array([self.cash / self.market.price, self.stock_count ])

    @property
    def price(self):
        return self.market.price
    @property
    def value(self):
        return float(self.cash + self.market.price * self.stock_count)
    @property
    def normalized_value(self):
        return self.value / self.market.init_val
    @property
    def value_inc(self):
        return self.value - self.init_val
    @property
    def stock_count(self):
        return len(self.stocks)

class single_stock_account(account):
    def __init__(self, *args, **kwargs):
        self.max_n_losses = None
        if "max_n_losses" in kwargs:
            self.max_n_losses = kwargs["max_n_losses"]
            del kwargs["max_n_losses"]
        self.round_earnings = 0
        self.on_market = False
        super().__init__(*args, **kwargs)
    def reset(self):
        super().reset()
        self.n_loss_trades = 0
        self.round_earnings = 0.0
        self.on_market = False
    def step(self):
        if self.market.delta < 0:
            self.n_loss_trades += 1
        return (False) if (self.max_n_losses is None) else (self.n_loss_trades >= self.max_n_losses)
    def wait(self):
        return 0.0
    def buy(self, x):
        #Buys one, and keeps it for one time step
        self.on_market = True
        self.round_earnings += self.market.normalized_delta
        return +self.market.normalized_delta
    def sell(self, x):
        self.on_market = False
        return -self.market.normalized_delta
    def get_state(self):
        f = lambda x : 2.0 * float(x) - 1.0
        return np.array([f(self.on_market)])
    @property
    def stock_count(self):
        return int(self.on_market)
    @property
    def value(self):
        return self.round_earnings
