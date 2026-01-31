import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def simulate_processes(n=400):
    np.random.seed(42)
    # Generate white noise with extra buffer for stability
    burn_in = 100
    total_n = n + burn_in
    w = np.random.normal(0, 1, total_n)
    
    # Base ARMA(2, 1) process: (1 + 0.7B - 0.6B^2)xt = (1 - 0.5B)wt
    # xt = -0.7xt-1 + 0.6xt-2 + wt - 0.5wt-1
    x_a = np.zeros(total_n)
    for t in range(2, total_n):
        x_a[t] = -0.7 * x_a[t-1] + 0.6 * x_a[t-2] + w[t] - 0.5 * w[t-1]
    
    # Slice to remove burn-in
    x_a = x_a[burn_in:]
    
    # b) ARIMA(2, 1, 1): Integrate x_a once
    x_b = np.cumsum(x_a)
    
    # c) ARIMA(2, 2, 1): Integrate x_a twice
    x_c = np.cumsum(x_b)
    
    # d) SARIMA (s=12): (1 - B^12)xt = x_a_t
    # xt = xt-12 + x_a_t
    x_d = np.zeros(len(x_a))
    for t in range(12, len(x_a)):
        x_d[t] = x_d[t-12] + x_a[t]
        
    # e) SARIMA (1-B)(1-B^12)xt = x_a_t
    x_e = np.cumsum(x_d)
    
    return [x_a, x_b, x_c, x_d, x_e]

titles = ["(a) ARMA(2,1)", "(b) ARIMA(2,1,1)", "(c) ARIMA(2,2,1)", 
          "(d) SARIMA(s=12)", "(e) Seasonal ARIMA"]
data_list = simulate_processes()

# Plotting ACF and PACF
fig, axes = plt.subplots(5, 2, figsize=(12, 20))
plt.subplots_adjust(hspace=0.5)

for i, data in enumerate(data_list):
    plot_acf(data, ax=axes[i, 0], lags=40, title=f"ACF: {titles[i]}")
    plot_pacf(data, ax=axes[i, 1], lags=40, title=f"PACF: {titles[i]}")

plt.show()
