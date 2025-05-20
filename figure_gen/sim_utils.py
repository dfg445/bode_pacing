import numpy as np
from abc import ABC, abstractmethod
import bisect
from collections import deque
from scipy.interpolate import interp1d
import control

EPS = 1e-6

def float_equal(a, b):
    return np.abs(a - b) < EPS

class Compensator(ABC):
    @abstractmethod
    def update(self, error, ts):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    

class LowPassFilter(ABC):
    @abstractmethod
    def update(self, input_signal):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    
class Plant(ABC):
    @abstractmethod
    def update(self, input_signal):
        pass
    

class Pcompensator(Compensator):
    def __init__(self, Kp):
        self._Kp = Kp
        
    def update(self, error, ts):
        proportional = self._Kp * error

        return max(0, proportional)

    def reset(self):
        pass


class PIcompensator(Compensator):
    def __init__(self, Kp, Ki, integral_init=0.0, integral_limits=(-np.inf, np.inf)):
        self._Kp = Kp
        self._Ki = Ki
        self._integral = integral_init
        self._last_ts = None
        self._integral_limits = integral_limits
        self._last_output = None

    # def update(self, error, ts):
    #     proportional = self._Kp * error
    #
    #     delta_time = 0 if self._last_ts is None else ts - self._last_ts
    #
    #     if self._last_output is None or 0 < self._last_output < 1.0:
    #         self._integral += error * delta_time
    #
    #     self._integral = min(max(self._integral_limits[0] / self._Ki, self._integral), self._integral_limits[1] / self._Ki)
    #     integral = self._Ki * self._integral
    #
    #     control_output = proportional + integral
    #     self._last_ts = ts
    #     self._last_output = control_output
    #
    #     return max(0, control_output), self._integral

    def update(self, error, ts):
        proportional = self._Kp * error

        delta_time = 0 if self._last_ts is None else ts - self._last_ts

        self._integral += error * delta_time

        self._integral = min(max(self._integral_limits[0] / self._Ki, self._integral),
                             self._integral_limits[1] / self._Ki)
        # print("int:", self._integral)
        integral = self._Ki * self._integral
        control_output = proportional + integral

        self._last_ts = ts
        if control_output >= 1.0:
            control_output = 1.0
            self._integral -= error * delta_time
        if control_output <= 0:
            control_output = 0.0
            self._integral -= error * delta_time

        return control_output, self._integral * self._Ki
    
    def reset(self):
        self._last_output = None
        self._integral = 0.0
        self._last_ts = None


class ZPcompensator(Compensator):
    def __init__(self, Kc=1.0, zeros=(1e-1,), poles=(1e-4, 1e-2), Tas=1.0):
        s = control.TransferFunction.s
        self.Gcs = Kc
        for zero in zeros:
            self.Gcs *= (s / zero + 1)
        for pole in poles:
            self.Gcs /= (s / pole + 1)

        self.Gcz = control.c2d(self.Gcs, Tas, method='tustin')
        self.num, self.den = self.Gcz.num[0][0], self.Gcz.den[0][0]

        print(self.Gcz)
        print("Difference Equation:")
        print("y[n] =", end=" ")
        for i in range(1, len(self.den)):
            print(f"- ({self.den[i]:.2e}) * y[n-{i}]", end=" ")
        for i in range(len(self.num)):
            print(f"+ ({self.num[i]:.2e}) * x[n-{i}]", end=" ")
        print()
        self.prev_output = deque([0] * (len(self.den) - 1))
        self.prev_input = deque([0] * len(self.num))

    @staticmethod
    def update_prev(prev, new_val):
        prev.appendleft(new_val)
        prev.pop()

    def update(self, error, ts):
        output = 0
        for i in range(1, len(self.den)):
            output -= self.den[i] * self.prev_output[i - 1]
        for i in range(len(self.num)):
            output += self.num[i] * self.prev_input[i - 1]
        output = min(max(0, output), 1)
        self.update_prev(self.prev_output, output)
        self.update_prev(self.prev_input, error)

        return output, None

    def reset(self):
        self.prev_output = deque([0] * (len(self.den) - 1))
        self.prev_input = deque([0] * len(self.num))


class FirstOrderLowPassFilter(LowPassFilter):
    def __init__(self, Tcut, sampling_time=0.5):
        self.Tcut = Tcut
        Tf = Tcut / (2 * np.pi)
        T = sampling_time
        self.b = T / (T + 2 * Tf)
        self.a = (T - 2 * Tf) / (T + 2 * Tf)
        
        self._prev_input = 0.0
        self._prev_output = 0.0
        
    def update(self, input_signal):
        output_signal = self.b * input_signal + self.b * self._prev_input - self.a * self._prev_output
        self._prev_input = input_signal
        self._prev_output = output_signal
        
        return output_signal
    
    def reset(self):
        self._prev_input = 0.0
        self._prev_output = 0.0
    
    @staticmethod
    def filter_list(input_list, Tauction=0.5, Tcut=10):
        result = [0] * len(input_list)
        lpf = FirstOrderLowPassFilter(Tcut=Tcut, sampling_time=Tauction)
        
        for index in range(len(input_list)):
            result[index] = lpf.update(input_list[index])
            
        return result
        

class ZeroOrderHold:
    def __init__(self, hold_interval=10):
        self._hold_interval = hold_interval
        self._hold_value = None
        self._last_ts = None
        
    def update(self, input_signal, ts):
        if self._hold_value is None or ts - self._last_ts >= self._hold_interval:
            self._hold_value = input_signal
            self._last_ts = ts
            
        return self._hold_value
        
    def reset(self):
        self._hold_value = None
        self._past_update = None

    
class ModeledAdsPlant(Plant):
    def __init__(self, Wn, noise_var_pect=0.0, noise_seed=103):
        self._Wn = Wn
        self._noise_var_pect = noise_var_pect
        self.rng = np.random.default_rng(noise_seed)
    
    def update(self, input_signal):
        raw_value = self._Wn * input_signal
        
        return self.rng.normal(loc=raw_value, scale=self._noise_var_pect * raw_value)
        

class TrafficFetcher:
    def __init__(self, ts_list, traffic_curve):
        start = int(ts_list[0])
        end = int(ts_list[-1])
        minute_intervals = (end - start) // 60 + 2

        interpolator = interp1d(ts_list, traffic_curve, kind='linear', bounds_error=False, fill_value="extrapolate")
        self._ticks = [start + index * 60 for index in range(minute_intervals)]
        self._traffic = interpolator(self._ticks)
        self._int_to_end = [0] * len(self._traffic)


        cumsum = 0
        for index in range(len(self._traffic) - 1, -1, -1):
            cumsum += self._traffic[index]
            self._int_to_end[index] = cumsum

        self._safe_divider = self._int_to_end[int(np.floor(0.95 * len(self._traffic)))]

    def get_traffic(self, ts):
        index = bisect.bisect_left(self._ticks, ts)
        current_traffic = self._traffic[index]
        remaining_traffic = self._int_to_end[index]
        
        return current_traffic, remaining_traffic, current_traffic / (remaining_traffic + self._safe_divider)


class Simulator:
    def __init__(self):
        self.spend = None
        self._traffic_fetcher = None
        self.daily_budget = None

    def _get_desired_velo(self, ts):
        """
        unit in dollar per minutes
        """
        daily_remain = self.daily_budget - self.spend
        traffic_ratio = self._traffic_fetcher.get_traffic(ts)
        
        return daily_remain * traffic_ratio[2], traffic_ratio


class PlantSimulator(Simulator):
    def __init__(self, compensator: Compensator, zoh: ZeroOrderHold, plant: ModeledAdsPlant, lpf: LowPassFilter,
                 traffic_fetcher: TrafficFetcher, daily_budget: float):
        super().__init__()
        self._compensator = compensator
        self._plant = plant
        self._lpf = lpf
        self._zoh = zoh
        self._traffic_fetcher = traffic_fetcher
        self.spend = 0.0
        self.daily_budget = daily_budget
        
    def run_sim(self, ts_list):
        observed_velo = 0.0
        
        desired_velo_list, observed_velo_list, error_list, raw_lambda_list, hold_lambda_list,  \
            actual_velo_list, traffic_ratio_list, spend_list, integral_list = [], [], [], [], [], [], [], [], []
        for ts in ts_list:
            desired_velo, traffic_ratio = self._get_desired_velo(ts)
            desired_velo_list.append(desired_velo)
            traffic_ratio_list.append(traffic_ratio)
            
            error = desired_velo - observed_velo    
            error_list.append(error)
            
            raw_lambda, integral = self._compensator.update(error, ts)
            raw_lambda_list.append(raw_lambda)
            integral_list.append(integral)

            hold_lambda = self._zoh.update(raw_lambda, ts)
            hold_lambda_list.append(hold_lambda)
            
            actual_velo = self._plant.update(hold_lambda)
            actual_velo_list.append(actual_velo)

            observed_velo = self._lpf.update(actual_velo)
            observed_velo_list.append(observed_velo)
            
            self.spend += actual_velo / 60
            spend_list.append(self.spend)
            if self.spend >= self.daily_budget:
                break

        return (
            desired_velo_list, 
            observed_velo_list, 
            error_list, 
            raw_lambda_list, 
            hold_lambda_list, 
            actual_velo_list, 
            traffic_ratio_list,
            spend_list,
            integral_list
        )


class AuctionSimulator(Simulator):
    def __init__(self, compensator: Compensator, zoh: ZeroOrderHold, lpf: LowPassFilter,
                 traffic_fetcher: TrafficFetcher, daily_budget: float):
        super().__init__()
        self._compensator = compensator
        self._lpf = lpf
        self._zoh = zoh
        self._traffic_fetcher = traffic_fetcher
        self.spend = 0.0
        self.daily_budget = daily_budget
        
    def run_sim(self, df, spending_velo_buffer_size=10, spending_velo_buffer_time=10):
        actual_velo_price_deque, actual_velo_time_deque = deque(), deque()

        observed_velo = 0.0
        (desired_velo_list, observed_velo_list, error_list, raw_lambda_list, hold_lambda_list, actual_velo_list,
         traffic_ratio_list, auction_status_list, spend_list, integral_list) = [], [], [], [], [], [], [], [], [], []

        for _, row in df.iterrows():
            ts = row["ts"].timestamp()
            auction_rank = row["auction_rank"]
            runner_up_bid = row["runner_up_bid"]
            p_event = row["p_event"]
            max_bid = row["max_bid"]
            eov = row["chosen_ad_organic_value"]
            winner_bid = row["winner_bid"]

            desired_velo, traffic_ratio = self._get_desired_velo(ts)
            traffic_ratio_list.append(traffic_ratio)
            desired_velo_list.append(desired_velo)
            
            error = desired_velo - observed_velo    
            error_list.append(error)
            
            raw_lambda, integral = self._compensator.update(error, ts)
            raw_lambda_list.append(raw_lambda)
            integral_list.append(integral)
            
            hold_lambda = self._zoh.update(raw_lambda, ts)
            hold_lambda_list.append(hold_lambda)

            final_bid = hold_lambda * max_bid * p_event + eov
            price = 0
            # print("Final_bid:", final_bid, "Winner_bid:", winner_bid)
            if final_bid > winner_bid or float_equal(final_bid, winner_bid):
                auction_status_list.append("WON")
                if auction_rank == 0:
                    price = min(hold_lambda * max_bid , runner_up_bid - eov)
                else:
                    price = min(hold_lambda * max_bid , winner_bid - eov)
            else:
                auction_status_list.append("LOST")
            
            if price > 0:
                actual_velo_price_deque.appendleft(price)
                actual_velo_time_deque.appendleft(ts)
            
            while (
                    len(actual_velo_price_deque) > 1
                    and (
                            ts - actual_velo_time_deque[-1] > spending_velo_buffer_time
                            or len(actual_velo_price_deque) > spending_velo_buffer_size
                    )
            ):
                actual_velo_price_deque.pop()
                actual_velo_time_deque.pop()
            # print(list(actual_velo_price_deque))
            # print(list(actual_velo_time_deque))
            
            actual_velo = 0
            if len(actual_velo_price_deque) > 1:
                actual_velo = sum(actual_velo_price_deque) / (actual_velo_time_deque[0] - actual_velo_time_deque[-1]) * 60  # dollar per minute

            # print("actual_velo", actual_velo)
            actual_velo_list.append(actual_velo)

            observed_velo = self._lpf.update(actual_velo)
            observed_velo_list.append(observed_velo)
            
            self.spend += price
            spend_list.append(self.spend)
            if self.spend >= self.daily_budget:
                break  
        
        return (
            desired_velo_list, 
            observed_velo_list, 
            error_list, 
            raw_lambda_list, 
            hold_lambda_list,
            actual_velo_list, 
            traffic_ratio_list, 
            auction_status_list,
            spend_list,
            integral_list
        )


class Metrics:
    @staticmethod
    def signal_line_pacing_error(desired_velo_list, actual_velo_list):
        desired_velo_array, actual_velo_array = np.array(desired_velo_list), np.array(actual_velo_list)
        abs_error = np.abs(desired_velo_array - actual_velo_array)
        error_pct = abs_error / desired_velo_array

        return np.mean(error_pct)

    @staticmethod
    def multi_line_pacing_error(list_of_desired_velo_list, list_of_actual_velo_list):
        pacing_error_list = []
        for i, desired_velo_list in enumerate(list_of_desired_velo_list):
            pacing_error_list.append(Metrics.signal_line_pacing_error(desired_velo_list, list_of_actual_velo_list[i]))

        return np.mean(pacing_error_list)

    @staticmethod
    def budget_weighted_pacing_error(list_of_desired_velo_list, list_of_actual_velo_list, weights):
        pacing_error_list = []
        for i, desired_velo_list in enumerate(list_of_desired_velo_list):
            pacing_error_list.append(Metrics.signal_line_pacing_error(desired_velo_list, list_of_actual_velo_list[i]))

        return np.mean(np.array(weights) / np.sum(weights) * np.array(pacing_error_list))
