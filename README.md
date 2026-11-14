Domain: E-commerce

How to run THIS
To run this project from a clean clone:
1. `pip install -r requirements.txt`
2. `python generate_data.py`
3. `python part1.py`
4. `python part2.py`
5. `python part3.py`
6. `python part4.py`

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

## Part 3 findings
R3.1 The four successive WCSS drops are: WCSS(2)-WCSS(3) = 659.83, WCSS(3)-WCSS(4) = 108.07, WCSS(4)-WCSS(5) = 171.06, and WCSS(5)-WCSS(6) = 45.55. The elbow supports K=3 because it is the point where the drop becomes less than half of the previous drop. Note that the drops do not shrink steadily (the drop from 4 to 5 is larger than 3 to 4), which indicates that the pinned starting rows give a reproducible solution, not necessarily the global optimum.

R3.2 
- Segment A (Premium Loyal): Long-tenure customers with low complaints and low discount reliance. Action: Offer exclusive early access to new products to maintain loyalty.
- Segment B (High-Value Friction): High average order values but also high complaint counts. Action: Proactively reach out to resolve complaints and offer personalized retention offers.
- Segment C (Bargain Churners): Newer customers who rely heavily on discounts. Action: Send targeted discount coupons to prevent them from leaving.
Segment C has the highest outcome_rate at 0.4664, indicating it is the most at-risk segment.

R3.3 
- Single linkage clusters at cut: {E2, E3, E6}, {E1}, {E4}, {E5}.
- Complete linkage clusters at cut: {E2, E3, E6}, {E1, E4, E5}.
The two rules disagree because single linkage uses the minimum distance between any two points (which causes a "chaining" effect, linking E1 to E4+E5), whereas complete linkage uses the maximum distance, forcing clusters to be tightly packed and keeping E1 separate.

R3.4 The first component is driven primarily by X1 (0.5317) and X5 (0.4418) in the positive direction, and X3 (-0.5187) and X2 (-0.4100) in the negative direction. We neEd 3 components to keep at least 80% of the variance. The mean_pc1_segment_A/B/C values (1.7773, -0.7831, -1.6709) show that Segment A is strongly separated from Segments B and C along the first principal component, capturing the biggest differences in customer profiles.

## Part 4 findings
R4.1 The best split is S4 (X1 <= 24) because it yields the highest information gain (and Gini gain). A tree grown until every leaf is pure would overfit by memorizing the training data's noise, failing to generalize. One rule to stop this is setting a maximum depth limit for the tree.

R4.2 I would ship the standalone Logistic model at τ=0.3 (f1 = 0.5714, recall = 0.7451). Combining the models into soft or hard voting did not help; the base models (KNN and Stump) were too weak and dragged the ensemble's performance down compared to the standalone logistic model.

R4.3 The total weight of the two misclassified points after round 1 normalisation is 0.2500. Round 2 trusts h2 more than h1 because h2 has a lower weighted error (0.1667 vs 0.2500), which results in a higher alpha (0.8047 vs 0.5493).

R4.4 
Model A option 2 table:
| Layer | Output Shape | Parameters |
|---|---|---|
| Conv1 | 28 × 28 × 6 | 156 |
| Pool1 | 14 × 14 × 6 | 0 |
| Conv2 | 10 × 10 × 16 | 2416 |
| Pool2 | 5 × 5 × 16 | 0 |
| Dense1 | 32 | 12832 |
| Dense2 | 10 | 330 |

Shapes for Model B: W_xh (64 × 128), W_hh (128 × 128), W_hy (128 × 16). The rnn_total does not grow when sequences get longer because the same weight matrices are reused at every time step. For 10 classes, use ReLU in the hidden layers, Softmax at the output, and Categorical Cross-Entropy loss. A very deep plain network stops learning in early layers because of the vanishing gradient problem; skip connections help by providing a direct path for gradients to flow backwards.

## Limitations
1. Synthetic Data: The dataset is entirely synthetic and generated from a script. The relationships between variables (like discounts and churn) are artificially injected and may not reflect real Flipkart customer behavior.
2. Single Split: We used a single fixed 560/240 train-test split for evaluation. A k-fold cross-validation approach on the test set would provide a more robust estimate of model performance. 
3. **Small Positive Class:** There are only 170 positive (churned) records in total, and only 51 in the test set. This high class imbalance makes performance metrics (like precision and recall) sensitive to just a few predictions.
