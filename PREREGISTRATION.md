# Pre-registration: "Why Spotify is not using Bayesian A/B testing"

Written before any simulation was run. Committed to git so the timestamp is checkable.

## Source (Frånberg and Schultzberg, Spotify, 2026; arXiv 2608.12949)
- C1 (Setting A). Flat prior + stop when P(B>A) > 0.95 is the same as naive peeking: FPR 0.303. Bayes factor (BF) stopping keeps FPR below 0.05 for any proper prior (0.016 to 0.019). Expected-loss stopping (epsilon = 0.02) gives FPR about 0.50.
- C2 (Setting B). GST (O'Brien-Fleming, 14 looks) has the highest power, above BF with the oracle empirical Bayes (EB) prior. The oracle EB prior has the lowest MSE. A prior fitted on a corpus that pools two programs has higher MSE than plain GST.
- C3 (Setting C). A corpus that keeps only past winners gives FPR around 0.20, and more data from the same source does not fix it.

## Part 1: reproduction (their settings, Appendix D)
Normal data, sigma = 1 known, delta = 0 (null) or 0.2. Setting A: looks every 50 per arm, n_max = 5,000, 10,000 sims.
Setting B: n_max = 1,000, p = 0.30 non-null rate, delta ~ N(0, 0.2^2) when non-null, 5,000 sims.
Setting C: K = 10 metrics, corpus sizes 10 to 500, winner-selected corpus.

H1. My Setting A FPRs match theirs: each within 0.015 of the reported value (flat 0.303, P>0.50 0.501, expected loss 0.499, BF matched 0.019, BF weak 0.016).
H2. Setting B ordering holds: GST power > BF-oracle-EB power; oracle EB has the lowest MSE; pooled EB MSE > GST MSE. PASS if all three hold.
H3. Setting C: winner-selected corpus FPR is between 0.10 and 0.30 at every corpus size and does not fall as the corpus grows (Spearman rho over sizes not significantly negative).

## Part 2: does it survive real-world conditions? (my extensions)
Their simulations use normal data with known variance. Real product metrics are skewed and the variance is estimated as data comes in.
Real metric: activity per user from MovieLens (43,584 users, ratings per user, skew about 47). A/A tests are random splits of real users, resampled with replacement.

H4. With the variance estimated from the data and looks starting at 50 users per arm, BF stopping still keeps FPR <= 0.05 on the skewed real metric.
  PASS if FPR <= 0.05 (upper end of the 95% binomial interval below 0.06).
  I expect early looks with a heavy tail could break this.
H5. Novelty effect: the true lift starts at 2x its long-run value and decays to the long-run value (half-life one third of the run). Methods that stop early report the novelty, not the long-run lift.
  Prediction: among detected experiments, BF stopping's estimate overstates the long-run effect more than GST (14 looks). PASS if relative bias BF > relative bias GST with bootstrap 95% CI excluding 0.
H6. Shrinkage helps against the winner's curse only when the prior is right: among significant results, the EB posterior mean has lower bias than the MLE under a calibrated prior, and higher absolute error than GST's MLE under the pooled prior. PASS if both hold.
