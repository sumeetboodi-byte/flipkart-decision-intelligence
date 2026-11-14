Domain: E-commerce

## Part 1 findings

R1.1 We reject H0 (z_stat = 2.8930, which is greater than the critical value of 1.96). A Type I error here means concluding the average order value is different from INR 800 when it actually is INR 800. The gap is 0.9382 column units (INR 93.82). This gap matters in practice because a ~12% difference in average order value would significantly alter category planning and revenue forecasts.

R1.2 The required sample size is n_required = 1293, but we only have 800 rows. This means our current estimate is less precise than the team's desired margin of error (INR 50). The planning team needs to collect roughly 500 more records to achieve their targeted precision.

R1.3 The gd_slope_gap is 0.0000. Gradient descent reached the closed-form line because the cost function for simple linear regression is convex (quadratic), so with a sufficiently small learning rate and enough steps, it converges exactly to the global minimum.

R1.4 resid_sd_ratio_raw = 3.7525 and resid_sd_ratio_log = 1.0396. The raw model shows a clear funnel (heteroscedasticity), as the spread of residuals grows significantly with the fitted value. The log model removes this funnel, as the ratio drops to near 1.0.

R1.5 The predictors with an absolute coefficient above 0.05 (excluding the constant) are: ln(X4) [1.1576], X2 [0.0940], X3 [0.1067], and X6 [0.1518]. Comparing mr_adj_r2 (0.7403) with loglog_adj_r2 (0.6561) shows that the extra predictors earn their place, as the adjusted R² improves by roughly 0.08 when adding them.
