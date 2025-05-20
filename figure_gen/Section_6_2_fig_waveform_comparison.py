import pandas as pd
import matplotlib.pyplot as plt
from sim_utils import (PlantSimulator, PIcompensator, ZPcompensator, Metrics, ZeroOrderHold,
                       ModeledAdsPlant, FirstOrderLowPassFilter, TrafficFetcher, AuctionSimulator)

line_id = "a783f80d-8926-4a41-b318-1beddca022de"
integral_init = 0.05
daily_budget = 387.5

Wn = 13.52

df_pacing = pd.read_pickle(f"pacing_line_{line_id}.pkl")
df_auction = pd.read_pickle(f"auction_line_{line_id}.pkl")


ts_list = []

for ts in range(int(df_pacing.loc[0, "ts"].timestamp()), int(df_pacing.loc[len(df_pacing) - 1, "ts"].timestamp()), 1):
    ts_list.append(ts)


Kp = 0.05
Ki = 0.00005

simulator_sim = PlantSimulator(
    compensator=PIcompensator(Kp=Kp, Ki=Ki, integral_init=0.0 / Ki, integral_limits=(0, 0.1)), 
    zoh=ZeroOrderHold(hold_interval=10), 
    plant=ModeledAdsPlant(Wn=Wn, noise_var_pect=0.05), 
    lpf=FirstOrderLowPassFilter(Tcut=10, sampling_time=1),
    traffic_fetcher=TrafficFetcher(
        ts_list=df_pacing["ts"].map(lambda x: x.timestamp()).tolist(), 
        traffic_curve=df_pacing["forecasted_impressions_per_min"].tolist()
    ), 
    daily_budget=daily_budget
)

(desired_velo_list_sim, observed_velo_list_sim, error_list_sim, raw_lambda_list_sim, 
 hold_lambda_list_sim, actual_velo_list_sim, traffic_ratio_list_sim, spend_list_sim, 
 integral_list_sim) = simulator_sim.run_sim(ts_list)


ts_list_pacing = df_pacing["ts"].tolist()


df_auction = df_auction[(df_auction["ts"] > ts_list_pacing[0]) & (df_auction["ts"] < ts_list_pacing[-1])]

simulator_auction = AuctionSimulator(
    compensator=PIcompensator(Kp=Kp, Ki=Ki, integral_init=integral_init / Ki, integral_limits=(0, 0.1)), 
    zoh=ZeroOrderHold(hold_interval=10), 
    lpf=FirstOrderLowPassFilter(Tcut=10, sampling_time=1),
    traffic_fetcher=TrafficFetcher(
        ts_list=df_pacing["ts"].map(lambda x: x.timestamp()).tolist(), 
        traffic_curve=df_pacing["forecasted_impressions_per_min"].tolist()
    ), 
    daily_budget=daily_budget
)

(desired_velo_list_auction, observed_velo_list_auction, error_list_auction, raw_lambda_list_auction, 
 hold_lambda_list_auction, actual_velo_list_auction, traffic_ratio_list_auction, 
 auction_status_list_auction, spend_list_auction, integral_list_auction) = simulator_auction.run_sim(
        df_auction, 
        spending_velo_buffer_size=len(df_auction),
        spending_velo_buffer_time=3600
)

print(
    "daily budget:", daily_budget,
    "\nspend sim:", simulator_sim.spend,
    "\nspend sim:", simulator_auction.spend,
    "\nPE simulation:", Metrics.signal_line_pacing_error(desired_velo_list_sim, actual_velo_list_sim),
    "\nPE auction:", Metrics.signal_line_pacing_error(desired_velo_list_auction, actual_velo_list_auction)
)

plt.close("all")

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 12))

ts_list_sim = ts_list[:len(actual_velo_list_sim)]
axs[0].plot(ts_list_sim, actual_velo_list_sim, label="actual velocity")
axs[0].plot(ts_list_sim, desired_velo_list_sim, label="desired velocity", linewidth=3)
axs[0].set_ylim(-0.1, 1.0)
# axs[0].set_xlabel("unix timestamp")
axs[0].set_ylabel("$/min")
axs[0].set_title("Proposed Compensator Pacing on Time-Domain Simulation")
axs[0].legend()


ts_list_auction = df_auction["ts"].map(lambda x: x.timestamp())[:len(actual_velo_list_auction)]
axs[1].plot(ts_list_auction, actual_velo_list_auction, label="actual velocity")
axs[1].plot(ts_list_auction, desired_velo_list_auction, label="desired velocity")
axs[1].set_ylim(-0.1, 1.0)
# axs[1].set_xlabel("unix timestamp")
axs[1].set_ylabel("$/min")
axs[1].set_title("Proposed Compensator Pacing on Real-World Auction")
axs[1].legend()


axs[2].plot(df_pacing["ts"].map(lambda x: x.timestamp()), df_pacing["observed_rate_usd"], label="actual velocity")
axs[2].plot(df_pacing["ts"].map(lambda x: x.timestamp()), df_pacing["desired_rate_usd"], label="desired velocity")
axs[2].set_ylim(-0.1, 1.0)
axs[2].set_xlabel("unix timestamp (sec)")
axs[2].set_ylabel("$/min")
axs[2].set_title("Legacy Pacing")
axs[2].legend()
