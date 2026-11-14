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
X_train = (X_train_raw - train_mean) / train_sd
X_test = (X_test_raw - train_mean) / train_sd

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def train_logistic(X, y, eta, epochs):
    n_samples, n_features = X.shape
    w, b = np.zeros(n_features), 0.0
    losses = {}
    for epoch in range(epochs + 1):
        y_pred = sigmoid(np.dot(X, w) + b)
        if epoch in [0, 1, 10, 100, 200, 1000]:
            losses[epoch] = compute_loss(y, y_pred)
        if epoch == epochs: break
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        w -= eta * dw
        b -= eta * db
    return w, b, losses

w_main, b_main, losses_main = train_logistic(X_train, y_train, 0.1, 1000)
test_probs = sigmoid(np.dot(X_test, w_main) + b_main)

def get_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp, fp, tn, fn, acc, prec, rec, f1, spec

res_05 = get_metrics(y_test, (test_probs >= 0.5).astype(float))
res_03 = get_metrics(y_test, (test_probs >= 0.3).astype(float))
blind_acc = np.sum(y_test == 0) / len(y_test)

thresh_rows = []
for t in np.arange(0.1, 0.95, 0.1):
    tp, fp, tn, fn, _, _, _, _, _ = get_metrics(y_test, (test_probs >= t).astype(float))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    thresh_rows.append([t, tp, fp, tn, fn, tpr, fpr])

def knn_predict(X_ref, y_ref, X_query, k):
    preds = []
    for q in X_query:
        dists = np.sqrt(np.sum((X_ref - q)**2, axis=1))
        idx = np.argsort(dists, kind='stable')
        top_k_y = y_ref[idx[:k]]
        preds.append(1.0 if np.sum(top_k_y) >= (k + 1) / 2 else 0.0)
    return np.array(preds)

k_values = list(range(1, 22, 2))
cv_results = []
folds = np.arange(len(y_train)) % 5

for k in k_values:
    f1_scores = []
    for f in range(5):
        tr_mask, va_mask = folds != f, folds == f
        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_va, y_va = X_train[va_mask], y_train[va_mask]
        y_pred_va = knn_predict(X_tr, y_tr, X_va, k)
        _, _, _, _, _, _, _, f1_val, _ = get_metrics(y_va, y_pred_va)
        f1_scores.append(f1_val)
    cv_results.append([k, np.mean(f1_scores)])

best_k = max(cv_results, key=lambda x: (x[1], -x[0]))[0]
knn_test_preds = knn_predict(X_train, y_train, X_test, best_k)
knn_metrics = get_metrics(y_test, knn_test_preds)

train_probs = sigmoid(np.dot(X_train, w_main) + b_main)
best_tau, min_cost = 0, float('inf')

for k in range(1, 20):
    tau = k / 20.0
    tp, fp, tn, fn, _, _, _, _, _ = get_metrics(y_train, (train_probs >= tau).astype(float))
    cost = 50 * fp + 400 * fn
    if cost < min_cost:
        min_cost = cost
        best_tau = tau

def get_cost(y_true, y_prob, tau):
    tp, fp, tn, fn, _, _, _, _, _ = get_metrics(y_true, (y_prob >= tau).astype(float))
    return 50 * fp + 400 * fn

cost_test_tau = get_cost(y_test, test_probs, best_tau)
cost_test_05 = get_cost(y_test, test_probs, 0.5)
cost_test_nobody = 400 * np.sum(y_test)

os.makedirs('outputs', exist_ok=True)

with open('outputs/part2_thresholds.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['tau', 'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr'])
    for r in thresh_rows:
        w.writerow([f"{r[0]:.1f}", r[1], r[2], r[3], r[4], f"{r[5]:.4f}", f"{r[6]:.4f}"])

with open('outputs/part2_knn_cv.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['K', 'mean_cv_f1'])
    for r in cv_results:
        w.writerow([r[0], f"{r[1]:.4f}"])

with open('outputs/part2_results.txt', 'w') as f:
    # 2.1
    f.write(f"train_rows = {len(y_train)}\n")
    f.write(f"train_positives = {int(np.sum(y_train))}\n")
    f.write(f"test_rows = {len(y_test)}\n")
    f.write(f"test_positives = {int(np.sum(y_test))}\n")
    # 2.2
    f.write(f"loss_epoch_0 = {losses_main[0]:.4f}\n")
    f.write(f"loss_epoch_1 = {losses_main[1]:.4f}\n")
    f.write(f"loss_epoch_10 = {losses_main[10]:.4f}\n")
    f.write(f"loss_epoch_100 = {losses_main[100]:.4f}\n")
    f.write(f"loss_epoch_1000 = {losses_main[1000]:.4f}\n")
    for i in range(7):
        f.write(f"weight_X{i+1} = {w_main[i]:.4f}\n")
    f.write(f"bias = {b_main:.4f}\n")
    f.write(f"test_bce = {compute_loss(y_test, test_probs):.4f}\n")
    # 2.3
    for eta in [0.01, 0.1, 1.0]:
        _, _, l = train_logistic(X_train, y_train, eta, 200)
        f.write(f"loss_epoch_200_eta_{eta} = {l[200]:.4f}\n")
    # 2.4
    for i in range(7):
        f.write(f"odds_ratio_X{i+1} = {math.exp(w_main[i]):.4f}\n")
    # 2.5
    labels = ["tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"]
    for name, val in zip(labels, res_05): f.write(f"tau_0.5_{name} = {val:.4f}\n" if isinstance(val, float) else f"tau_0.5_{name} = {val}\n")
    for name, val in zip(labels, res_03): f.write(f"tau_0.3_{name} = {val:.4f}\n" if isinstance(val, float) else f"tau_0.3_{name} = {val}\n")
    f.write(f"blind_accuracy = {blind_acc:.4f}\n")
    # 2.7
    f.write(f"knn_best_k = {best_k}\n")
    for name, val in zip(labels, knn_metrics): f.write(f"knn_{name} = {val:.4f}\n" if isinstance(val, float) else f"knn_{name} = {val}\n")
    # 2.8
    f.write(f"cost_tau = {best_tau:.2f}\n")
    f.write(f"cost_train_at_tau = {min_cost}\n")
    f.write(f"cost_test_at_tau = {cost_test_tau}\n")
    f.write(f"cost_test_at_0.5 = {cost_test_05}\n")
    f.write(f"cost_test_flag_nobody = {cost_test_nobody}\n")

print("Success! Part 2 outputs saved.")