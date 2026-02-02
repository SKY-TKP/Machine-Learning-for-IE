## 📚 Reports

### **[1. Lab 1: Model Selection & Validation Framework](https://github.com/SKY-TKP/Machine-Learning-for-IE/blob/main/2102-575%3A%20STAT%20INFER%20MODEL/Lab%20Coding/Lab%201%3A%20Model%20Selection/Lab1_G4.pdf)**

* **Core Objective:** Implementation of a Cross-Validation framework to evaluate feature settings and prevent overfitting.
* **Key Techniques:**  **Single Fold Operation:** custom implementation of training and validation logic within a cross-validation loop.
  * **Selection Scores:** Comparative analysis using **AIC, AICC, BIC, Adjusted**, and **Validation RMSE**.
  * **Validation Methods:** Comparison between **5-Fold Cross-Validation** and **Hold-out Validation** to verify model consistency.


* **Outcome:** Identified the optimal feature setting that minimizes validation error while maintaining model parsimony.

---

### **[2. Lab 2: Outlier Analysis & Robust Regression](https://github.com/SKY-TKP/Machine-Learning-for-IE/tree/main/2102-575%3A%20STAT%20INFER%20MODEL/Lab%20Coding/Lab%202%3A%20Outlier%20Analysis)**

* **Core Objective:** Identifying and managing influential observations that distort linear regression models.
* **Key Techniques:**
  * **H-Matrix (Hat Matrix):** Computation of diagonal elements to detect **High-Leverage points** (outliers in the feature space).
          $$H = X(X^T X)^{-1} X^T$$
  * **Cook's Distance:** Statistical measure to identify **Influential observations** that significantly change model coefficients.
          $$D_i = \frac{e_i^2}{p \cdot MSE} \left[ \frac{h_{ii}}{(1 - h_{ii})^2} \right]$$


* **Theoretical Comparison:**
  * **H-Matrix:** Focuses on stabilizing the regression geometry and reducing parameter variance.
  * **Cook's Distance:** Focuses on reducing bias by eliminating points with both high leverage and large residuals.


* **Outcome:** Demonstrated significant MSE improvement by selectively removing outliers to achieve a more robust fit.

---
