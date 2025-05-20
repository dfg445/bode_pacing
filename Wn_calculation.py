import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def bucketize_b_by_a_values(list_a, list_b, num_buckets=100):
    quantile_boundaries = np.quantile(np.unique(list_a), np.linspace(0, 1, num_buckets + 1))
    quantile_indices = pd.cut(list_a, bins=quantile_boundaries, labels=range(num_buckets), include_lowest=True, duplicates='drop')
    
    buckets = {i: [] for i in range(num_buckets)}
    for i, q in enumerate(quantile_indices):
        buckets[q].append(list_b[i])
        
    return quantile_boundaries, buckets

df = pd.read_pickle("pacing_line_a783f80d-8926-4a41-b318-1beddca022de.pkl")

ts_list = df["ts"].map(lambda x: x.timestamp()).tolist()
lambda_list = df["lambda"].tolist()
observed_rate_list = df["observed_rate_usd"].tolist()
observed_rate_list = observed_rate_list[1:] + [observed_rate_list[-1]]

interpolator_lambda = interp1d(ts_list, lambda_list, kind='linear', bounds_error=False, fill_value="extrapolate")
interpolator_rate = interp1d(ts_list, observed_rate_list, kind='linear', bounds_error=False, fill_value="extrapolate")

start = int(ts_list[0])
end = int(ts_list[-1])
minute_intervals = (end - start) // 60 + 2
new_ts_list = [start + index * 60 for index in range(minute_intervals)]

new_lambda_list = interpolator_lambda(new_ts_list)
new_observed_rate_list = interpolator_rate(new_ts_list)


delta_lambda_to_rates = np.abs(new_observed_rate_list / new_lambda_list)

print(
      "max $ / minute * lambda", np.quantile(delta_lambda_to_rates, 0.99), "\n"
      "min $ / minute * lambda", np.quantile(delta_lambda_to_rates, 0.01), "\n"
)
