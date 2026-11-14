Domain: E-commerce

## Part 1 findings

R1.1 We reject H0 (z_stat = 2.8930, which is greater than the critical value of 1.96). A Type I error here means concluding the average order value is different from INR 800 when it actually is INR 800. The gap is 0.9382 column units (INR 93.82). This gap matters in practice because a ~12% difference in average order value would significantly alter category planning and revenue forecasts.

R1.2 The required sample size is n_required = 1293, but we only have 800 rows. This means our current estimate is less precise than the team's desired margin of error (INR 50). The planning team needs to collect roughly 500 more records to achieve their targeted precision.

R1.3 The gd_slope_gap is 0.0000. Gradient descent reached the closed-form line because the cost function for simple linear regression is convex (quadratic), so with a sufficiently small learning rate and enough steps, it converges exactly to the global minimum.

R1.4 resid_sd_ratio_raw = 3.7525 and resid_sd_ratio_log = 1.0396. The raw model shows a clear funnel (heteroscedasticity), as the spread of residuals grows significantly with the fitted value. The log model removes this funnel, as the ratio drops to near 1.0.

R1.5 The predictors with an absolute coefficient above 0.05 (excluding the constant) are: ln(X4) [1.1576], X2 [0.0940], X3 [0.1067], and X6 [0.1518]. Comparing mr_adj_r2 (0.7403) with loglog_adj_r2 (0.6561) shows that the extra predictors earn their place, as the adjusted R² improves by roughly 0.08 when adding them.


## Part 2 findings

R2.1 The losses from lowest to highest are: `eta_1.0` (0.3703), `eta_0.1` (0.3776), `eta_0.01` (0.5091). A learning rate that is too small (0.01) learns too slowly and has not converged after 200 updates, while the larger rate (1.0) converges much faster to a lower loss.

R2.2 The three predictors with the largest absolute weights are X3 (0.9013), X1 (-0.5554), and X2 (0.5505). The largest odds ratio is for X3 (2.4629), which means that for every one standard deviation increase in discount order share, the odds of a customer churning multiply by about 2.46 (customers who heavily use discounts are more likely to leave).

R2.3 The blind accuracy is 0.7875, which is close to the τ=0.5 accuracy of 0.8042. Accuracy alone is a weak score here because the dataset is imbalanced (only ~21% are churners), so a model predicting "no churn" for everyone scores highly while catching zero churners. Lowering τ from 0.5 to 0.3 increased recall from 0.4706 to 0.7451 (catching more churners) but dropped precision from 0.5455 to 0.4634 (more false alarms).

R2.4 The logistic model at τ=0.3 significantly outperforms the KNN model (best K=5). The logistic model achieves an f1 of 0.5714 and recall of 0.7451, compared to the KNN model's f1 of 0.3696 and recall of 0.3333. 

R2.5 I would deploy the logistic regression model with a threshold of τ=0.15. This threshold minimizes the financial cost to INR 6050 on the test set, which is far better than the cost at τ=0.5 (INR 11800) or the cost of flagging nobody (INR 20400). The threshold was chosen on the training rows (rather than test rows) to prevent data leakage and ensure the model generalizes to unseen data.
