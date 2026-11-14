import csv
import math
import os
import numpy as np


X_raw, y_raw = [], []
with open('data/records.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        X_raw.append([float(row[i]) for i in range(1, 8)])
        y_raw.append(float(row[8]))

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)
n = len(y_raw)


train_idx, test_idx = [], []
for outcome in [0.0, 1.0]:
    idx = np.where(y_raw == outcome)[0]
    split_point = (7 * len(idx)) // 10
    train_idx.extend(idx[:split_point])
    test_idx.extend(idx[split_point:])
train_idx.sort()
test_idx.sort()

X_train_raw, y_train = X_raw[train_idx], y_raw[train_idx]
X_test_raw, y_test = X_raw[test_idx], y_raw[test_idx]

train_mean = np.mean(X_train_raw, axis=0)
train_sd = np.std(X_train_raw, axis=0, ddof=1)
X_train_std = (X_train_raw - train_mean) / train_sd
X_test_std = (X_test_raw - train_mean) / train_sd


def calc_entropy(y):
    if len(y) == 0: return 0
    p1 = np.mean(y)
    p0 = 1 - p1
    ent = 0
    if p1 > 0: ent -= p1 * np.log2(p1)
    if p0 > 0: ent -= p0 * np.log2(p0)
    return ent

def calc_gini(y):
    if len(y) == 0: return 0
    p1 = np.mean(y)
    p0 = 1 - p1
    return 1 - (p1**2 + p0**2)

def evaluate_split(X_col, y, condition):
    parent_ent = calc_entropy(y)
    parent_gini = calc_gini(y)
    
    if condition == '>': mask = X_col > 0.5 if max(X_col) > 1 else X_col > 0.5 # S3 is X3>0.5
    elif condition == '==': mask = X_col == 1.0 # S1, S2
    elif condition == '<=': mask = X_col <= 24.0 # S4
    
    y_yes, y_no = y[mask], y[~mask]
    
    w_yes = len(y_yes) / len(y)
    w_no = len(y_no) / len(y)
    
    info_gain = parent_ent - (w_yes * calc_entropy(y_yes) + w_no * calc_entropy(y_no))
    gini_gain = parent_gini - (w_yes * calc_gini(y_yes) + w_no * calc_gini(y_no))
    
    return len(y_yes), int(np.sum(y_yes)), len(y_no), int(np.sum(y_no)), info_gain, gini_gain

splits = {
    'S1': (X_train_raw[:, 5], '=='),
    'S2': (X_train_raw[:, 6], '=='),
    'S3': (X_train_raw[:, 2], '>'),
    'S4': (X_train_raw[:, 0], '<=')
}

splits_data = []
best_info, best_info_name = -1, None
best_gini, best_gini_name = -1, None

for name, (col, cond) in splits.items():
    res = evaluate_split(col, y_train, cond)
    splits_data.append([name] + list(res))
    if res[4] > best_info: best_info, best_info_name = res[4], name
    if res[5] > best_gini: best_gini, best_gini_name = res[5], name

root_entropy = calc_entropy(y_train)
root_gini = calc_gini(y_train)


# Stump: Best split is S3 (X3 > 0.5)
mask_stump = X_train_raw[:, 2] > 0.5
rate_yes = np.mean(y_train[mask_stump])
rate_no = np.mean(y_train[~mask_stump])

def predict_stump(X_arr):
    return np.where(X_arr[:, 2] > 0.5, rate_yes, rate_no)

test_stump_prob = predict_stump(X_test_raw)

# Logistic
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

w_log, b_log = np.zeros(7), 0.0
for _ in range(1000):
    y_pred = sigmoid(np.dot(X_train_std, w_log) + b_log)
    dw = (1/len(y_train)) * np.dot(X_train_std.T, (y_pred - y_train))
    db = (1/len(y_train)) * np.sum(y_pred - y_train)
    w_log -= 0.1 * dw
    b_log -= 0.1 * db

test_log_prob = sigmoid(np.dot(X_test_std, w_log) + b_log)

# KNN (K=5)
def knn_prob(X_ref, y_ref, X_query, k=5):
    probs = []
    for q in X_query:
        dists = np.sqrt(np.sum((X_ref - q)**2, axis=1))
        idx = np.argsort(dists, kind='stable')[:k]
        probs.append(np.mean(y_ref[idx]))
    return np.array(probs)

test_knn_prob = knn_prob(X_train_std, y_train, X_test_std, 5)


pred_log = (test_log_prob >= 0.3).astype(float)
pred_knn = (test_knn_prob >= 0.6).astype(float)
pred_stump = (test_stump_prob >= 0.3).astype(float)

pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(float)
pred_soft_prob = (test_log_prob + test_knn_prob + test_stump_prob) / 3
pred_soft = (pred_soft_prob >= 0.3).astype(float)

def get_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp, fp, tn, fn, prec, rec, f1

voting_data = [
    ['logistic'] + list(get_metrics(y_test, pred_log)),
    ['knn'] + list(get_metrics(y_test, pred_knn)),
    ['stump'] + list(get_metrics(y_test, pred_stump)),
    ['hard_vote'] + list(get_metrics(y_test, pred_hard)),
    ['soft_vote'] + list(get_metrics(y_test, pred_soft))
]


y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

w1 = np.ones(8) / 8
err1 = np.sum(w1[h1 != y_ada])
alpha1 = 0.5 * np.log((1 - err1) / err1)

w2_unnorm = w1 * np.exp(-alpha1 * y_ada * h1)
w2 = w2_unnorm / np.sum(w2_unnorm)

err2 = np.sum(w2[h2 != y_ada])
alpha2 = 0.5 * np.log((1 - err2) / err2)

w3_unnorm = w2 * np.exp(-alpha2 * y_ada * h2)
w3 = w3_unnorm / np.sum(w3_unnorm)

ens_pred = np.sign(alpha1 * h1 + alpha2 * h2)
ens_acc = np.mean(ens_pred == y_ada)

ada_rows = []
for i in range(8):
    ada_rows.append([1, i+1, f"{w1[i]:.4f}", f"{w2_unnorm[i]:.4f}", f"{w2[i]:.4f}"])
for i in range(8):
    ada_rows.append([2, i+1, f"{w2[i]:.4f}", f"{w3_unnorm[i]:.4f}", f"{w3[i]:.4f}"])


oob_fraction = (1 - 1/560)**560


mlp_h1 = 784 * 16 + 16
mlp_h2 = 16 * 16 + 16
mlp_out = 16 * 10 + 10
mlp_total = mlp_h1 + mlp_h2 + mlp_out

cnn_c1 = (5*5*1)*6 + 6
cnn_c2 = (5*5*6)*16 + 16
cnn_d1 = (5*5*16)*32 + 32
cnn_d2 = 32*10 + 10
cnn_total = cnn_c1 + cnn_c2 + cnn_d1 + cnn_d2

fc_layer_params = 128*128*3*1024 + 1024
conv_layer_params = (3*3*3)*64 + 64

rnn_xh = 64 * 128
rnn_hh = 128 * 128
rnn_hy = 128 * 16
rnn_bh = 128
rnn_by = 16
rnn_total = rnn_xh + rnn_hh + rnn_hy + rnn_bh + rnn_by

arch_rows = [
    ['mlp_hidden1', mlp_h1],
    ['mlp_hidden2', mlp_h2],
    ['mlp_output', mlp_out],
    ['cnn_conv1', cnn_c1],
    ['cnn_conv2', cnn_c2],
    ['cnn_dense1', cnn_d1],
    ['cnn_dense2', cnn_d2]
]


os.makedirs('outputs', exist_ok=True)

with open('outputs/part4_splits.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['split','n_yes','positives_yes','n_no','positives_no','info_gain','gini_gain'])
    for r in splits_data:
        w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4f}", f"{r[6]:.4f}"])

with open('outputs/part4_voting.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model','tp','fp','tn','fn','precision','recall','f1'])
    for r in voting_data:
        w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4f}", f"{r[6]:.4f}", f"{r[7]:.4f}"])

with open('outputs/part4_adaboost.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['round','point','weight_before','weight_unnormalised','weight_after'])
    w.writerows(ada_rows)

with open('outputs/part4_architecture.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['layer','parameters'])
    w.writerows(arch_rows)

with open('outputs/part4_results.txt', 'w') as f:
    f.write(f"root_entropy = {root_entropy:.4f}\n")
    f.write(f"root_gini = {root_gini:.4f}\n")
    f.write(f"best_split_by_info_gain = {best_info_name}\n")
    f.write(f"best_split_by_gini_gain = {best_gini_name}\n")
    f.write(f"stump_rate_yes = {rate_yes:.4f}\n")
    f.write(f"stump_rate_no = {rate_no:.4f}\n")
    f.write(f"round1_error = {err1:.4f}\n")
    f.write(f"round1_alpha = {alpha1:.4f}\n")
    f.write(f"round2_error = {err2:.4f}\n")
    f.write(f"round2_alpha = {alpha2:.4f}\n")
    f.write(f"ensemble_accuracy = {ens_acc:.4f}\n")
    f.write(f"oob_fraction = {oob_fraction:.4f}\n")
    f.write(f"mlp_total = {mlp_total}\n")
    f.write(f"cnn_total = {cnn_total}\n")
    f.write(f"fc_layer_params = {fc_layer_params}\n")
    f.write(f"conv_layer_params = {conv_layer_params}\n")
    f.write(f"rnn_total = {rnn_total}\n")

print("Success! Part 4 outputs saved.")