import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sim_utils import (AuctionSimulator, PIcompensator, ZPcompensator, ZeroOrderHold,
                                  FirstOrderLowPassFilter, TrafficFetcher, Metrics)



line_id = "a783f80d-8926-4a41-b318-1beddca022de"
integral_init = 0.05
daily_budget = 387.5

df_pacing = pd.read_pickle(f"pacing_line_{line_id}.pkl")
df_auction = pd.read_pickle(f"auction_line_{line_id}.pkl")
ts_list_pacing = df_pacing["ts"].tolist()


df_auction = df_auction[(df_auction["ts"] > ts_list_pacing[0]) & (df_auction["ts"] < ts_list_pacing[-1])]

ts_list = df_auction["ts"].tolist()

Kp = 0.05
Ki = 0.00005


simulator = AuctionSimulator(
    compensator=PIcompensator(Kp=Kp, Ki=Ki, integral_init=integral_init / Ki, integral_limits=(0, 0.5)), 
    # compensator=ZPcompensator(Kc=1.0, zeros=(1e-1,), poles=(1e-3,), Tas=1.0),
    zoh=ZeroOrderHold(hold_interval=10), 
    lpf=FirstOrderLowPassFilter(Tcut=10, sampling_time=1),
    traffic_fetcher=TrafficFetcher(
        ts_list=df_pacing["ts"].map(lambda x: x.timestamp()).tolist(), 
        traffic_curve=df_pacing["forecasted_impressions_per_min"].tolist()
    ), 
    daily_budget=daily_budget
)

(desired_velo_list, observed_velo_list, error_list, raw_lambda_list, hold_lambda_list, actual_velo_list, 
 traffic_ratio_list, auction_status_list, spend_list, integral_list) = simulator.run_sim(
        df_auction, 
        spending_velo_buffer_size=len(df_auction),
        spending_velo_buffer_time=3600
)

print(
    "daily budget:", simulator.daily_budget,
    "\nspend:", simulator.spend,
    "\nauction wins:", sum(np.array(auction_status_list) == "WON"), "total:", len(auction_status_list),
    "\nPE:", Metrics.signal_line_pacing_error(desired_velo_list, actual_velo_list),
)


data_len = len(desired_velo_list)

plt.close("all")

fig, axs = plt.subplots(7, 1, sharex=True, figsize=(9, 12))
ts_list_auction = df_auction["ts"].map(lambda x: x.timestamp())[:len(desired_velo_list)]

axs[0].plot(ts_list_auction, desired_velo_list, "k", label="desired velocity", linewidth=2)
axs[0].plot(ts_list_auction, actual_velo_list, "r--", label="actual velocity")
# axs[0].set_ylim(-0.1, 1)
axs[0].set_ylabel("$/min")
axs[0].legend()

axs[1].plot(ts_list_auction, raw_lambda_list, label="lambda")
# axs[1].plot(indices, hold_lambda_list, label="hold_lambda")
# axs[1].set_ylim(0, 0.5)
axs[1].legend()

axs[2].plot(ts_list_auction, error_list, label="input error")
axs[2].set_ylabel("$/min")
# axs[2].set_ylim(-1, 1)
axs[2].legend()

axs[3].plot(ts_list_auction, integral_list, "k-", label="integral")
axs[3].set_ylabel("$/min")
axs[3].legend()

axs[4].plot(ts_list_auction, [t[0] for t in traffic_ratio_list], "k-", label="traffic curve")
axs[4].legend()

axs[5].plot(ts_list_auction, auction_status_list, ".", markersize=5, alpha=0.2, label="auction status")
axs[5].legend()

axs[6].plot(ts_list_auction, spend_list, label="spend")
axs[6].set_ylabel("$")
axs[6].legend()
axs[6].set_xlabel("unix timestamp (sec)")



