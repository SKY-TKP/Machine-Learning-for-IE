# 💓 EKG & Time Series Analysis Report

## 1. Objective

This repository contains the analysis and forecasting of cardiac signals (EKG) and various stochastic processes. The primary goal is to apply **ARIMA/SARIMA** models to identify underlying patterns and predict future values accurately.

---

## 2. Theoretical Framework

To select the correct model, we follow a rigorous diagnostic workflow based on the **Box-Jenkins** methodology:

* **Stationarity:** Checked via the Augmented Dickey-Fuller (ADF) test.
* **Identification:** Using ACF/PACF plots to determine the lags ().
* **Seasonality:** Identifying repeating cycles () in rhythmic signals like EKG.

---

## 3. Results: Process Identification (Prog1)

We identified four distinct processes using **ACF/PACF Signature Analysis**:

| Process | Model | Key Indicator |
| --- | --- | --- |
| **P1** | **AR(2)** | PACF cuts off after lag 2. |
| **P2** | **ARIMA(0,1,1)** | 1st Difference is stationary; ACF spikes at lag 1. |
| **P3** | **ARIMA(0,2,0)** | Quadratic trend; requires double differencing. |
| **P4** | **Seasonal RW** | Periodic ACF spikes every 4 lags (). |

---

## 4. EKG Forecasting (SARIMA)

For the EKG signal, we utilized **SARIMA ** to capture the rhythmic heartbeats.

* **Training Set:** First 120 points (Baseline signal).
* **Testing Set:** Final 40 points (Forecast target).
* **Seasonal Period ():** Determined by the average distance between R-peaks in the EKG waveform.

---

## 5. Key Findings

1. **Model Robustness:** SARIMA effectively captures the "Phase" of the heartbeat but can be sensitive to heart rate variability (HRV).
2. **Differencing:** Crucial for removing "Baseline Wander" in medical signals.
3. **Residuals:** Final diagnostics showed white noise residuals, confirming the model has extracted all available information.

---

**Would you like me to help you write the `Installation` or `How to run` section for your GitHub README as well?**
