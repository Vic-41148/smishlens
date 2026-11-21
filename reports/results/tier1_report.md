# Tier-1 experiments — UCI anchor, train split only

- Train rows: **3,301** (411 spam / 2,890 ham, prevalence 0.125)
- Cross-validation: StratifiedKFold(5, shuffle=True, seed=42)
- Metrics: PR-AUC (`ap`), F1, MCC, ROC-AUC. **Accuracy is not reported.**
- The test split was not opened. All numbers are 5-fold CV means ± std.
- E1 fixes LogisticRegression(C=1, class_weight=balanced); E2 fixes the same LR.
- E3 uses the production feature set (word + char TF-IDF + meta).
- Fold-to-fold std is shown next to every mean. Gaps smaller than roughly one std are not interpretable; UCI contains almost no obfuscation, so the E1 arms are expected to tie on this anchor.

## E1 — preprocessing ablation (same LR)

| variant | ap_mean | f1_mean | mcc_mean | roc_auc_mean |
|---|---|---|---|---|
| raw (no normaliser) | 0.9833 ± 0.0066 | 0.9458 ± 0.0148 | 0.9394 ± 0.0161 | 0.9946 ± 0.0038 |
| masked + min_df=3 | 0.9833 ± 0.0056 | 0.9452 ± 0.0136 | 0.9380 ± 0.0153 | 0.9952 ± 0.0027 |
| masked (default) | 0.9830 ± 0.0057 | 0.9425 ± 0.0162 | 0.9351 ± 0.0179 | 0.9950 ± 0.0028 |
| masked, global confusables | 0.9830 ± 0.0057 | 0.9425 ± 0.0162 | 0.9351 ± 0.0179 | 0.9950 ± 0.0028 |
| masked + deleet | 0.9829 ± 0.0057 | 0.9425 ± 0.0162 | 0.9351 ± 0.0179 | 0.9950 ± 0.0028 |
| masked + min_df=2 | 0.9829 ± 0.0058 | 0.9425 ± 0.0161 | 0.9351 ± 0.0178 | 0.9950 ± 0.0028 |
| masked + stem | 0.9827 ± 0.0059 | 0.9451 ± 0.0163 | 0.9380 ± 0.0180 | 0.9946 ± 0.0032 |
| masked + min_df=1 | 0.9814 ± 0.0058 | 0.9335 ± 0.0182 | 0.9251 ± 0.0199 | 0.9945 ± 0.0029 |
| masked + stopwords | 0.9801 ± 0.0044 | 0.9377 ± 0.0186 | 0.9295 ± 0.0205 | 0.9935 ± 0.0026 |

**Best by PR-AUC:** raw (no normaliser) (0.9833).

## E2 — feature ablation (same LR)

| variant | ap_mean | f1_mean | mcc_mean | roc_auc_mean | n_features |
|---|---|---|---|---|---|
| word + char | 0.9848 ± 0.0044 | 0.9485 ± 0.0135 | 0.9414 ± 0.0153 | 0.9955 ± 0.0024 | 27417 |
| word + char + meta | 0.9830 ± 0.0057 | 0.9425 ± 0.0162 | 0.9351 ± 0.0179 | 0.9950 ± 0.0028 | 27429 |
| char TF-IDF | 0.9825 ± 0.0038 | 0.9506 ± 0.0107 | 0.9439 ± 0.0120 | 0.9945 ± 0.0022 | 18679 |
| word TF-IDF | 0.9798 ± 0.0065 | 0.9405 ± 0.0198 | 0.9321 ± 0.0227 | 0.9940 ± 0.0027 | 8738 |
| BoW (word 1-2) | 0.9674 ± 0.0165 | 0.9246 ± 0.0210 | 0.9143 ± 0.0239 | 0.9889 ± 0.0056 | 8738 |

**Best by PR-AUC:** word + char (0.9848).

## E3 — classical model comparison

| variant | ap_mean | f1_mean | mcc_mean | roc_auc_mean |
|---|---|---|---|---|
| LinearSVC | 0.9845 ± 0.0064 | 0.9547 ± 0.0135 | 0.9492 ± 0.0147 | 0.9951 ± 0.0030 |
| LogisticRegression | 0.9830 ± 0.0057 | 0.9425 ± 0.0162 | 0.9351 ± 0.0179 | 0.9950 ± 0.0028 |
| ComplementNB | 0.9827 ± 0.0055 | 0.9365 ± 0.0177 | 0.9279 ± 0.0200 | 0.9921 ± 0.0040 |
| MultinomialNB | 0.9827 ± 0.0055 | 0.9598 ± 0.0050 | 0.9541 ± 0.0057 | 0.9921 ± 0.0040 |
| RandomForest | 0.9801 ± 0.0054 | 0.9235 ± 0.0227 | 0.9161 ± 0.0236 | 0.9939 ± 0.0024 |
| Dummy (prior) | 0.1245 ± 0.0005 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |

**Best by PR-AUC:** LinearSVC (0.9845).

## Champion

**LinearSVC** wins the train-split comparison on PR-AUC. It is promoted to the final retrain in C3, subject to the same threshold/conformal protocol.
