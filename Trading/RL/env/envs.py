import numpy as np
from env.base import *

class env:
    def __init__(self, file=None, interactive=False, single_stock=False, max_n_losses=None, episode_length=128, state_length=128, max_init_stock=10, max_init_cash=10):
        self.market = databank(file=file, interactive=interactive, episode_length=episode_length, state_length=state_length)

        self.single_stock = single_stock
        self.max_n_losses = max_n_losses

        if single_stock:
            self.action_dict = {0 : "wait", 1 : "buy"}
            self.customer = single_stock_account(self.market, max_n_losses=max_n_losses, max_init_cash=0, max_init_stock=0)
        else:
            self.action_dict = {0 : "wait", 1 : "buy", 2 : "sell"}
            self.customer = account(self.market, max_init_cash=max_init_cash, max_init_stock=max_init_stock)
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
    def perform_action(self, a, eval=False):
        stock_count, cash, normalized_price, info = self.customer.stock_count, self.customer.cash, self.market.normalized_price, {}
        action = self.action_dict[a[0]]

        ####
        #### Action
        ####
        if action == "wait":
            _r = self.customer.wait()
        #
        elif action == "buy":
            _r = self.customer.buy(1)
        #
        elif action == "sell":
            _r = self.customer.sell(1)
        #Internals
        _d = self.market.step() or self.customer.step()

        ####
        #### Reward
        ####
        # _r = (self.market.normalized_price - normalized_price) * self.customer.stock_count
        info["time_limit_reached"] = self.market.t > self.market.end
        r = np.array( [ _r ] )
        d = np.array( [ _d ] )
        info["action"] = action
        return r, self.get_state(), d, info

    @property
    def n_actions(self):
        return len(self.action_dict)

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

    @property
    def value(self):
        return self.customer.value
