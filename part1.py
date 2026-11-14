import csv
import math
import os
import numpy as np

values, x1, x2, x3, x4, x5, x6, x7 = [], [], [], [], [], [], [], []
with open('data/records.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        x1.append(float(row[1]))
        x2.append(float(row[2]))
        x3.append(float(row[3]))
        x4.append(float(row[4]))
        x5.append(float(row[5]))
        x6.append(float(row[6]))
        x7.append(float(row[7]))
        values.append(float(row[9]))

y = np.array(values)
X4 = np.array(x4)
n = len(y)

mean_value = np.mean(y)
sd_value = np.std(y, ddof=1)
se_value = sd_value / math.sqrt(n)
ci95_low = mean_value - (1.96 * se_value)
ci95_high = mean_value + (1.96 * se_value)


z_stat = (mean_value - 8.0) / se_value
reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"


E = 0.5
n_required = math.ceil((1.96 * sd_value / E) ** 2)


mean_X4 = np.mean(X4)
reg_slope = np.sum((X4 - mean_X4) * (y - mean_value)) / np.sum((X4 - mean_X4)**2)
reg_intercept = mean_value - reg_slope * mean_X4
fitted = reg_intercept + reg_slope * X4
tss = np.sum((y - mean_value)**2)
ess = np.sum((fitted - mean_value)**2)
rss = np.sum((y - fitted)**2)
tss_equals = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
r2 = 1 - rss / tss
rmse = math.sqrt(rss / n)
mae = np.mean(np.abs(y - fitted))


sd_X4 = np.std(X4, ddof=1)
z_X4 = (X4 - mean_X4) / sd_X4
m, c, eta = 0.0, 0.0, 0.1
gd_J0 = np.mean((y - (m * z_X4 + c))**2)
gd_hist = {0: gd_J0}
for step in range(1, 501):
    pred = m * z_X4 + c
    e = y - pred
    dj_dm = -2/n * np.sum(z_X4 * e)
    dj_dc = -2/n * np.sum(e)
    m = m - eta * dj_dm
    c = c - eta * dj_dc
    if step in [1, 10, 100, 500]:
        gd_hist[step] = np.mean((y - (m * z_X4 + c))**2)
gd_slope = m / sd_X4
gd_intercept = c - m * mean_X4 / sd_X4
gd_slope_gap = abs(gd_slope - reg_slope)


def get_sd_ratio(fitted_vals, residuals):
    idx = np.argsort(fitted_vals, kind='stable')
    res_sorted = residuals[idx]
    lower = res_sorted[:400]
    upper = res_sorted[400:]
    sd_lower = np.std(lower, ddof=1)
    sd_upper = np.std(upper, ddof=1)
    return sd_lower, sd_upper, sd_upper / sd_lower

residuals_raw = y - fitted
sd_low_raw, sd_up_raw, ratio_raw = get_sd_ratio(fitted, residuals_raw)

ln_y = np.log(y)
ln_X4 = np.log(X4)
mean_ln_y = np.mean(ln_y)
mean_ln_X4 = np.mean(ln_X4)
loglog_slope = np.sum((ln_X4 - mean_ln_X4) * (ln_y - mean_ln_y)) / np.sum((ln_X4 - mean_ln_X4)**2)
loglog_intercept = mean_ln_y - loglog_slope * mean_ln_X4
fitted_log = loglog_intercept + loglog_slope * ln_X4
rss_log = np.sum((ln_y - fitted_log)**2)
tss_log = np.sum((ln_y - mean_ln_y)**2)
loglog_r2 = 1 - rss_log / tss_log
loglog_rmse = math.sqrt(rss_log / n)

residuals_log = ln_y - fitted_log
sd_low_log, sd_up_log, ratio_log = get_sd_ratio(fitted_log, residuals_log)


X_mr = np.column_stack((np.ones(n), ln_X4, x1, x2, x3, x5, x6, x7))
beta = np.linalg.solve(X_mr.T @ X_mr, X_mr.T @ ln_y)
pred_mr = X_mr @ beta
rss_mr = np.sum((ln_y - pred_mr)**2)
mr_r2 = 1 - rss_mr / tss_log
mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - 7 - 1)
loglog_adj_r2 = 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1)

os.makedirs('outputs', exist_ok=True)
with open('outputs/part1_results.txt', 'w') as f:
    f.write(f"n = {n}\n")
    f.write(f"mean_value = {mean_value:.4f}\n")
    f.write(f"sd_value = {sd_value:.4f}\n")
    f.write(f"se_value = {se_value:.4f}\n")
    f.write(f"ci95_low = {ci95_low:.4f}\n")
    f.write(f"ci95_high = {ci95_high:.4f}\n")
    f.write(f"z_stat = {z_stat:.4f}\n")
    f.write(f"reject_h0 = {reject_h0}\n")
    f.write(f"n_required = {n_required}\n")
    f.write(f"reg_slope = {reg_slope:.4f}\n")
    f.write(f"reg_intercept = {reg_intercept:.4f}\n")
    f.write(f"tss_equals_ess_plus_rss = {tss_equals}\n")
    f.write(f"r2 = {r2:.4f}\n")
    f.write(f"rmse = {rmse:.4f}\n")
    f.write(f"mae = {mae:.4f}\n")
    f.write(f"gd_J0 = {gd_hist[0]:.4f}\n")
    f.write(f"gd_J1 = {gd_hist[1]:.4f}\n")
    f.write(f"gd_J10 = {gd_hist[10]:.4f}\n")
    f.write(f"gd_J100 = {gd_hist[100]:.4f}\n")
    f.write(f"gd_J500 = {gd_hist[500]:.4f}\n")
    f.write(f"gd_slope = {gd_slope:.4f}\n")
    f.write(f"gd_intercept = {gd_intercept:.4f}\n")
    f.write(f"gd_slope_gap = {gd_slope_gap:.4f}\n")
    f.write(f"resid_sd_lower_raw = {sd_low_raw:.4f}\n")
    f.write(f"resid_sd_upper_raw = {sd_up_raw:.4f}\n")
    f.write(f"resid_sd_ratio_raw = {ratio_raw:.4f}\n")
    f.write(f"loglog_intercept = {loglog_intercept:.4f}\n")
    f.write(f"loglog_slope = {loglog_slope:.4f}\n")
    f.write(f"loglog_r2 = {loglog_r2:.4f}\n")
    f.write(f"loglog_rmse = {loglog_rmse:.4f}\n")
    f.write(f"resid_sd_lower_log = {sd_low_log:.4f}\n")
    f.write(f"resid_sd_upper_log = {sd_up_log:.4f}\n")
    f.write(f"resid_sd_ratio_log = {ratio_log:.4f}\n")
    for i in range(8):
        f.write(f"mr_beta_{i} = {beta[i]:.4f}\n")
    f.write(f"mr_r2 = {mr_r2:.4f}\n")
    f.write(f"mr_adj_r2 = {mr_adj_r2:.4f}\n")
    f.write(f"loglog_adj_r2 = {loglog_adj_r2:.4f}\n")

print("Success! Part 1 results saved to outputs/part1_results.txt")