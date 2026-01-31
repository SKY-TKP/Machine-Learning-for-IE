import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_ar2_acf(phi1, phi2, max_lag=10):
    """
    Computes rho(n) for an AR(2) process using recursive Yule-Walker steps.
    """
    rho = np.zeros(max_lag + 1)
    rho[0] = 1.0
    # rho(1) = phi1 / (1 - phi2)
    rho[1] = phi1 / (1 - phi2)
    
    for n in range(2, max_lag + 1):
        # rho(n) = phi1*rho(n-1) + phi2*rho(n-2)
        rho[n] = phi1 * rho[n-1] + phi2 * rho[n-2]
    
    return rho

# Define the cases from the assignment
cases = {
    "Case 2.a (phi1=-0.5, phi2=-0.5)": (-0.5, -0.5),
    "Case 2.b (phi1=0, phi2=0.2)": (0, 0.2),
    "Case 2.c (phi1=0.25, phi2=0.1)": (0.25, 0.1)
}

# Generate data
lags = np.arange(11)
results = {}

plt.figure(figsize=(12, 7))

for name, (p1, p2) in cases.items():
    rho_values = compute_ar2_acf(p1, p2, max_lag=10)
    results[name] = rho_values
    plt.stem(lags, rho_values, label=name, use_line_collection=True)

# Formatting the table for verification
df = pd.DataFrame(results, index=lags)
print("Autocorrelation Values (rho_x(n)):")
print(df.round(2))

# Plotting
plt.title('Autocorrelation Function (ACF) for AR(2) Processes', fontsize=14)
plt.xlabel('Lag (n)', fontsize=12)
plt.ylabel('rho_x(n)', fontsize=12)
plt.xticks(lags)
plt.axhline(0, color='black', linewidth=1)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
