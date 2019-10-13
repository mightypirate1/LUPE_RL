from env.envs import *

def build(env_str):
    if env_str == "SingleStock-v0":
        return env(
                    #env-constr
                    file=None,
                    interactive=False,
                    single_stock=True,
                    max_n_losses=None,
                    #market-constr
                    episode_length=128,
                    state_length=128,
                    )
    if env_str == "MultiStock-v0":
        return env(
                    #env-constr
                    file=None,
                    interactive=False,
                    single_stock=False,
                    max_n_losses=None,
                    #market-constr
                    episode_length=128,
                    state_length=128,
                    #customer-constr
                    max_init_cash=10,
                    max_init_stock=5,
                    )

    assert False, "Unrecognized environment!"
