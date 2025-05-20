import pandas as pd
import matplotlib.pyplot as plt
from figure_gen.sim_utils import (PlantSimulator, PIcompensator, ZPcompensator, Metrics, ZeroOrderHold,
                                  ModeledAdsPlant, FirstOrderLowPassFilter, TrafficFetcher)



plt.close("all")
df_pacing = pd.read_pickle("pacing_line_a783f80d-8926-4a41-b318-1beddca022de.pkl")
ts_list = []

for ts in range(int(df_pacing.loc[0, "ts"].timestamp()), int(df_pacing.loc[len(df_pacing) - 1, "ts"].timestamp()), 1):
    ts_list.append(ts)

Wn = 13.52

Kp = 0.005
Ki = 0.00005
zeros = [1e-1]
poles = [1e-4, 1e-2]

simulator = PlantSimulator(
    # compensator=PIcompensator(Kp=Kp, Ki=Ki, integral_init=0.005 / Ki, integral_limits=(0, 0.1)), 
    compensator=ZPcompensator(Kc=1.0, zeros=zeros, poles=poles, Tas=1.0),
    zoh=ZeroOrderHold(hold_interval=10), 
    plant=ModeledAdsPlant(Wn=Wn, noise_var_pect=0.05), 
    lpf=FirstOrderLowPassFilter(Tcut=10, sampling_time=1),
    traffic_fetcher=TrafficFetcher(
        ts_list=df_pacing["ts"].map(lambda x: x.timestamp()).tolist(), 
        traffic_curve=df_pacing["forecasted_impressions_per_min"].tolist()
    ), 
    daily_budget = 387.5
)

desired_velo_list, observed_velo_list, error_list, raw_lambda_list, hold_lambda_list,  \
    actual_velo_list, traffic_ratio_list, spend_list, integral_list = simulator.run_sim(ts_list)

print(
    "daily budget:", simulator.daily_budget,
    "\nspend:", simulator.spend,
    "\nPE:", Metrics.signal_line_pacing_error(desired_velo_list, actual_velo_list)
)

data_len = len(desired_velo_list)

plt.close("all")

fig, axs = plt.subplots(6, 1, sharex=True, figsize=(9, 12))

# indices = ts_list[:data_len]
indices = [index for index in range(len(desired_velo_list))]


axs[0].plot(indices, actual_velo_list, "r", label="actual_velo")
axs[0].plot(indices, desired_velo_list, "k-", label="desired_velo", lw=2)
# axs[0].set_ylim(-0.01, 0.1)
axs[0].legend()

axs[1].plot(indices, error_list, label="error")
axs[1].legend()

axs[2].plot(indices, raw_lambda_list, label="raw_lambda")
axs[2].plot(indices, hold_lambda_list, label="hold_lambda")
axs[2].legend()

axs[3].plot(indices, [t[0] for t in traffic_ratio_list], "k-", label="current_traffic")
axs[3].legend()

axs[4].plot(indices, integral_list, "k-", label="integral")
axs[4].legend()

axs[5].plot(indices, spend_list, label="spend")
axs[5].legend()
