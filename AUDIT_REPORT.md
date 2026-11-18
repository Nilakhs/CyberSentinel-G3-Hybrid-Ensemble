# Strict Independent Code and Results Audit Report

**Auditor:** Independent Technical Auditor  
**Scope:** CyberSentinel AI — Track G3 (Hybrid / Ensemble Learning)  
**Target Directory:** `g3_work/` and upstream references  
**Audit Rule:** Strictly read-only inspection. No files modified, no models retrained.

---

### Audit Checklist & Point-by-Point Findings

---

#### 1. Was the Autoencoder trained separately inside each training fold during OOF generation?
- **Status:** **PASS**
- **File:** `autoencoder.py`
- **Function:** `generate_oof_ae_meta(X_train_ae, y_train, n_splits, seed)` (lines 232–278)
- **Evidence:**
  Inside the `StratifiedKFold` loop across training rounds 3, 4, 5:
  ```python
  for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_ae, y_int)):
      y_train_fold = y_train[train_idx]
      normal_mask  = y_train_fold == C.NORMAL_CLASS
      X_ae_normal  = X_train_ae[train_idx][normal_mask]
      ae = G3Autoencoder()
      ae.fit(X_ae_normal, verbose=False)
      meta = ae.meta_features(X_train_ae[val_idx])
      oof_meta[val_idx] = meta
  ```
  A fresh `G3Autoencoder` instance is instantiated and fitted strictly on `X_train_ae[train_idx][normal_mask]` (679 or 680 normal flows per fold). It generates meta-features only for the held-out validation slice `val_idx`.

---

#### 2. Were XGBoost predictions genuinely out-of-fold?
- **Status:** **PASS**
- **File:** `xgboost_model.py`
- **Function:** `generate_oof_xgb_proba(X_train, y_train, n_splits, seed)` (lines 80–116)
- **Evidence:**
  ```python
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
  for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_int)):
      X_tr_fold = X_train[train_idx]
      y_tr_fold = y_train[train_idx]
      X_val_fold = X_train[val_idx]
      m = G3XGBoostModel()
      m.fit(X_tr_fold, y_tr_fold)
      proba = m.predict_proba(X_val_fold)
      oof_proba[val_idx] = proba
  ```
  For each fold, a separate XGBoost model is trained on `train_idx` and evaluated exclusively on the unseen `val_idx`. The base model never predicts on samples present in its training fold.

---

#### 3. Was the meta-classifier trained exclusively on training-derived OOF features?
- **Status:** **PASS**
- **File:** `stacking.py`
- **Function:** `G3StackingClassifier.fit_meta(X_train_xgb, X_train_ae, y_train, n_splits)` (lines 61–120)
- **Evidence:**
  Inputs `X_train_xgb`, `X_train_ae`, and `y_train` represent the 7,941 samples of rounds 3, 4, 5. The meta-feature matrix is assembled from `oof_xgb_proba` (shape `7941, 4`) and `oof_ae_meta` (shape `7941, 9`):
  ```python
  meta_X = np.concatenate([oof_xgb_proba, oof_ae_meta], axis=1)  # (7941, 13)
  self.meta_scaler = StandardScaler()
  meta_X_scaled = self.meta_scaler.fit_transform(meta_X)
  self.meta_clf = LogisticRegression(**C.META_CLF_PARAMS)
  self.meta_clf.fit(meta_X_scaled, y_train)
  ```
  Round 6 test data is not passed to `fit_meta()` and has no influence on the meta-classifier weights or its scaler.

---

#### 4. Was the Autoencoder trained only on normal flows from rounds 3, 4, 5?
- **Status:** **PASS**
- **File:** `stacking.py` & `autoencoder.py`
- **Functions:** `generate_oof_ae_meta` (autoencoder.py line 260) and `refit_base_models` (stacking.py lines 145–149)
- **Evidence:**
  In both OOF generation and Phase B full refitting:
  ```python
  normal_mask = y_train == C.NORMAL_CLASS
  X_ae_normal = X_train_ae[normal_mask]
  self.ae_model = G3Autoencoder()
  self.ae_model.fit(X_ae_normal, verbose=True)
  ```
  Only the 1,019 normal flows from training rounds 3, 4, 5 are supplied to `ae.fit()`. All attack flows (scan, synflood, bruteforce) are masked out.

---

#### 5. Were scalers and preprocessing fitted only on training data?
- **Status:** **PASS**
- **File:** `data_loader.py` & `stacking.py`
- **Functions:** `fit_scaler` (data_loader.py lines 59–63), `load_splits` (lines 96–99), `fit_meta` (stacking.py line 104)
- **Evidence:**
  ```python
  ae_scaler = fit_scaler(X_train_ae_raw)       # fit strictly on X_train
  X_train_ae = ae_scaler.transform(X_train_ae_raw)
  X_test_ae  = ae_scaler.transform(X_test_ae_raw) # transform only on test
  ```
  Similarly, `self.meta_scaler.fit_transform(meta_X)` in `stacking.py` fits strictly on training OOF meta-features and calls only `.transform()` on test meta-features at inference time.

---

#### 6. Were the 22 selected features obtained without using round 6?
- **Status:** **PASS**
- **File:** `upstream/Network_Intrusion_AISD/cybersentinel_code/v2/run_phase3.py`
- **Function:** Module-level script execution (lines 21–36)
- **Evidence:**
  ```python
  TRAIN_ROUNDS, TEST_ROUND = [3, 4, 5], 6
  train = df[df["round"].isin(TRAIN_ROUNDS)].reset_index(drop=True)
  res = sf.select(train, seed=0)
  gkf = GroupKFold(n_splits=len(TRAIN_ROUNDS))
  ```
  Phase 3 filtered the dataset to rounds 3, 4, 5 prior to running constant-variance filtering, correlation pruning, and GroupKFold mutual information ranking. Round 6 was strictly held out and excluded from the feature selection procedure.

---

#### 7. Was round 6 used only for final evaluation?
- **Status:** **PASS**
- **File:** `run_experiments.py`
- **Function:** `run_all()` (lines 40–185)
- **Evidence:**
  `X_te_xgb`, `X_te_ae`, and `y_te` (Round 6, 2,735 flows) are passed exclusively to `.predict()` calls after model training is completely finished:
  - Line 56: `xgb_base.predict(X_te_xgb)`
  - Line 83: `ae_standalone.predict_anomaly(X_te_ae)`
  - Line 157: `stacker.predict(X_te_xgb, X_te_ae)`
  - Line 181: `soft_voter.predict(X_te_xgb, X_te_ae)`
  No test data is passed to any fitting, thresholding, or hyperparameter tuning function.

---

#### 8. Verify the 13 meta-features and explain their exact construction.
- **Status:** **PASS**
- **File:** `stacking.py` & `autoencoder.py`
- **Functions:** `G3StackingClassifier._build_meta_features` (stacking.py lines 163–171) & `G3Autoencoder.meta_features` (autoencoder.py lines 186–193)
- **Evidence:**
  The 13 meta-features are constructed as:
  1. **Features 1–4 ($4\text{D}$):** XGBoost class probabilities $[P_{\text{bruteforce}}, P_{\text{normal}}, P_{\text{scan}}, P_{\text{synflood}}]$ generated via `predict_proba()` on the 22 selected features.
  2. **Features 5–12 ($8\text{D}$):** Autoencoder bottleneck latent vector $z \in \mathbb{R}^8$ generated by `encoder(x)`.
  3. **Feature 13 ($1\text{D}$):** Scalar MSE reconstruction error $\frac{1}{66} \sum_{i=1}^{66} (x_i - \hat{x}_i)^2$.
  Concatenation: $4 + 8 + 1 = 13$ dimensions.

---

#### 9. Verify that all four-class metrics are calculated correctly.
- **Status:** **PASS**
- **File:** `evaluation.py`
- **Function:** `compute_metrics(y_true, y_pred, classes)` (lines 30–64)
- **Evidence:**
  - `accuracy`: standard multi-class accuracy $\frac{\text{correct}}{\text{total samples}}$.
  - `macro_f1`: unweighted arithmetic mean of F1 scores across all four classes (`average="macro"`), ensuring the minority class (`bruteforce`) has equal weight to majority classes.
  - `recall_{class}`: per-class recall $\frac{\text{True Positives for class } c}{\text{Total ground-truth samples of class } c}$ computed for each class in `["bruteforce", "normal", "scan", "synflood"]`.

---

#### 10. Explain why the standalone Autoencoder has binary metrics while the hybrid has multiclass metrics.
- **Status:** **PASS**
- **File:** `run_experiments.py`
- **Function:** `run_all()` (lines 80–115)
- **Evidence:**
  - **Standalone Autoencoder (E1):** Trained unsupervised strictly on benign traffic ($y = \text{normal}$). It computes a reconstruction error $\text{MSE}(x)$. If $\text{MSE}(x) > \text{threshold}$, it flags the flow as an anomaly ("attack"); otherwise "normal". Because it never observed attack labels during training, it cannot distinguish between scan, synflood, and bruteforce. Thus, its classification output is inherently binary.
  - **Hybrid Stacking (E2):** Feeds the Autoencoder's 8 latent dimensions and reconstruction error alongside XGBoost's 4-class probabilities into a supervised multinomial Logistic Regression meta-classifier. The meta-classifier is trained on multi-class targets and assigns full 4-class labels.

---

#### 11. Verify the false-alarm-rate calculation.
- **Status:** **PASS**
- **File:** `evaluation.py`
- **Function:** `compute_metrics` (lines 47–53)
- **Evidence:**
  ```python
  true_attack = y_true != C.NORMAL_CLASS
  pred_attack = y_pred != C.NORMAL_CLASS
  n_normal = int((~true_attack).sum())
  false_alarm = float((pred_attack & ~true_attack).sum() / n_normal) if n_normal else 0.0
  ```
  $\text{False Alarm Rate (FAR)} = \frac{\text{False Positives}}{\text{Total Actual Normal}} = \frac{\sum (\text{Pred is Attack } \land \text{ True is Normal})}{\sum (\text{True is Normal})}$.
  This matches standard networking Fall-out / FPR on normal traffic and replicates the upstream formula in `train_lib.py`.

---

#### 12. Verify prediction latency measurement. Does it include all three components?
- **Status:** **PASS**
- **File:** `evaluation.py` & `stacking.py`
- **Functions:** `measure_single_flow_latency` (evaluation.py lines 67–92) and `G3StackingClassifier.predict` (stacking.py lines 174–187)
- **Evidence:**
  `measure_single_flow_latency(stacker.predict, (X_te_xgb, X_te_ae))` times `stacker.predict(X_te_xgb[:1], X_te_ae[:1])` across 200 repetitions with high-resolution `time.perf_counter()`.
  Inside `stacker.predict`:
  1. Component 1: `self.xgb_model.predict_proba(X_xgb)` (XGBoost forward pass on 22 features).
  2. Component 2: `self.ae_model.meta_features(X_ae)` (PyTorch forward pass computing latent vector + MSE on 66 features).
  3. Component 3: `self.meta_scaler.transform(meta_X)` and `self.meta_clf.predict(meta_X_scaled)` (Logistic Regression inference).
  All three model inference components are included. It does not include upstream network packet capture (which is external to the model).

---

#### 13. Verify model-size calculations.
- **Status:** **PASS**
- **File:** `evaluation.py`
- **Function:** `get_model_size_kb(obj)` (lines 94–100)
- **Evidence:**
  Calculates in-memory serialized size via `round(len(pickle.dumps(obj)) / 1024.0, 1)`.
  - E0 (XGBoost): 632.1 KB
  - E1 (Autoencoder): 28.6 KB
  - E2 (Stacking): 662.6 KB
  - E3 (Soft Voting): 660.9 KB
  Physical on-disk checkpoints in `models/` are `g3_xgb_full.json` (489.2 KB), `g3_ae_full.pt` (27.9 KB), and `g3_stacking_meta.pkl` (1.8 KB), confirming consistent footprint.

---

#### 14. Independently inspect the six tests. What exactly do they test and what do they not test?
- **Status:** **PASS**
- **File:** `tests/test_g3_pipeline.py`
- **Evidence:**
  - **What they test:**
    1. `test_config_definitions`: verifies feature counts (66, 22), classes (4), round split definitions ([3,4,5], 6), and meta-dimension (13).
    2. `test_data_loader_structure`: verifies raw CSV load (10,676 rows, 74 columns) and split partitioning.
    3. `test_xgboost_fit_predict`: verifies XGBoost training, prediction, and probability normalization (sums to 1.0).
    4. `test_autoencoder_reconstruction_and_meta`: verifies Autoencoder MSE error non-negativity, latent dimension (8), and meta-feature dimension (9).
    5. `test_compute_metrics`: verifies arithmetic accuracy of multi-class metrics and zero false-alarm detection against synthetic ground truth.
    6. `test_single_flow_latency`: verifies latency timing execution and sanity bounds.
  - **What they do NOT test:**
    - They do not test the full end-to-end 3-fold OOF cross-validation execution inside pytest (omitted to keep unit tests fast; end-to-end integration is tested in `run_experiments.py`).
    - They do not test soft-voting ensemble probabilities or checkpoint reloading from disk.

---

#### 15. Confirm that the reported results CSV matches the actual evaluation code.
- **Status:** **PASS**
- **File:** `results/g3_comparison.csv`
- **Evidence:**
  Direct byte and text inspection of `g3_comparison.csv` shows:
  ```csv
  accuracy,macro_f1,attack_recall,normal_false_alarm,recall_bruteforce,recall_normal,recall_scan,recall_synflood,latency_ms,size_kb,experiment,type
  1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.53,632.1,E0: XGBoost Baseline,Supervised ML
  0.9941,0.9996,0.9996,0.0337,0.9697,0.9663,1.0,1.0,0.289,28.6,E1: Standalone Autoencoder,Unsupervised DL Anomaly
  1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.355,662.6,E2: G3 Hybrid Stacking,Hybrid Ensemble (ML+DL)
  1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.898,660.9,E3: G3 Soft Voting,Voting Ensemble (ML+DL)
  ```
  The values match the execution log of `run_experiments.py` exactly to four decimal places.

---

### Overall Audit Verdict

**FINAL VERDICT: PASS — SAFE TO PRESENT TO COLLEGE GUIDE**

**Key Justifications for Guide Presentation:**
1. **Zero Data Leakage:** Split boundaries between training (rounds 3, 4, 5) and test (round 6) were strictly maintained. Out-Of-Fold generation was used for meta-features, scalers were fitted only on training data, and metadata envelope columns were never exposed to the models.
2. **Defensible 1.0000 Scores:** The near-perfect accuracy on test Round 6 is explicitly documented as a natural consequence of a clean virtual testbed where attack tools (`nmap`, `hydra`, `hping3`) produce clear behavioral signatures. This is framed honestly as a lab baseline rather than claimed as unconstrained real-world generalization.
3. **Sound Architecture:** Justified the use of an Autoencoder over an LSTM due to the tabular nature of completed flow records.
4. **Reproducibility:** All artifacts, tests (6/6 passing), plots, and trained models exist and can be re-verified with a single command.
