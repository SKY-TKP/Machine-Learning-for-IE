# Report

## 1. Objective

To identify the stochastic process type and order for four distinct datasets (**P1-P4**) using statistical diagnostics and visualization techniques.

---

## 2. Methodology

The identification process follows these key steps:

1. **Stationarity Check:** Using the **Augmented Dickey-Fuller (ADF) Test**.
2. **Order Identification:** Analyzing **ACF** and **PACF** plots to determine AR, MA, or ARIMA components.
3. **Transformation:** Applying first or second-order differencing () for non-stationary series.

---

## 3. Results Summary

| Process | Identification | Diagnostics & Reasoning |
| --- | --- | --- |
| **P1** | **AR(2)** | **Stationary** (). PACF cuts off at lag 2; ACF decays geometrically. |
| **P2** | **ARIMA(0,1,1)** | **Non-Stationary**. Becomes stationary after 1st differencing (). ACF shows a significant spike at lag 1. |
| **P3** | **ARIMA(0,2,0)** | **Non-Stationary**. Exhibits a quadratic trend. Requires 2nd-order differencing () to achieve stationarity. |
| **P4** | **Seasonal RW** | **Non-Stationary**. Strong seasonal ACF spikes at intervals of 4, indicating a **Seasonal Period ()**. |

---

## 4. Technical Interpretation

### **P1: Autoregressive (Order 2)**

P1 is a purely stationary process. The current value is directly influenced by its two immediate predecessors.

### **P2: Integrated Moving Average**

P2 represents a Random Walk with a localized error correction component (MA). The first difference reveals that the errors are correlated at lag 1.

### **P3: Second-Order Integrated**

P3 shows an accelerating trend. Physically, this represents a system where the "acceleration" is constant/random, requiring two levels of integration to describe its position.

### **P4: Seasonal Process**

P4 is dominated by a repeating 4-step cycle. This is common in quarterly data or systems with a fixed rhythmic vibration.

---

## 5. Conclusion

By combining the **ADF Test** for stationarity with **ACF/PACF** for order selection, we successfully mapped each raw dataset to its mathematical process. This identification is crucial for selecting the appropriate forecasting model in subsequent stages.

---
