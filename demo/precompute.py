"""Precompute SHAP and LIME attributions for the live demo.

Reproduces the paper pipeline on the local artifacts in data/:
  - 500-instance sample stratified by income (random_state=42)
  - SHAP TreeExplainer, positive-class slice  -> (500, 14)
  - LIME LimeTabularExplainer, class-1 weights -> (500, 14)

Outputs go to demo/artifacts/ and are loaded by demo/app.py.
Run once:  python demo/precompute.py
"""
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "demo", "artifacts")
os.makedirs(OUT, exist_ok=True)

N_SAMPLE = 500
SEED = 42

print("Loading model and data ...")
model = pickle.load(open(os.path.join(DATA, "model_rf.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(DATA, "feature_names.pkl"), "rb"))
test = pd.read_csv(os.path.join(DATA, "adult_test.csv"))
train = pd.read_csv(os.path.join(DATA, "adult_train.csv"))

# Stratified 500-instance sample by income class, matching notebook 02a
idx = (
    test.groupby("income", group_keys=False)["income"]
    .apply(lambda s: s.sample(n=int(round(N_SAMPLE * len(s) / len(test))), random_state=SEED))
    .index
)
sample = test.loc[idx].sort_index()
X_sample = sample[feature_names]
np.save(os.path.join(OUT, "sample_indices.npy"), sample.index.values)
print(f"Sample: {X_sample.shape}, positive rate {sample['income'].mean():.3f}")

# ---------------- SHAP ----------------
import shap

t0 = time.time()
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_sample, check_additivity=False)
sv = np.array(sv)
# shap 0.4x returns list [neg, pos]; 0.45+ returns (n, f, 2)
shap_pos = sv[..., 1] if sv.ndim == 3 else sv[1]
np.save(os.path.join(OUT, "shap_values.npy"), shap_pos)
print(f"SHAP done in {time.time() - t0:.0f}s -> {shap_pos.shape}")

# ---------------- LIME ----------------
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

X_train = train[feature_names].values
lime_explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["<=50K", ">50K"],
    mode="classification",
    random_state=SEED,
)

t0 = time.time()
lime_vals = np.zeros(X_sample.shape)
rows = X_sample.values
for i in tqdm(range(len(rows)), desc="LIME explanations"):
    exp = lime_explainer.explain_instance(
        rows[i], model.predict_proba, num_features=len(feature_names), num_samples=1000
    )
    for feat_idx, weight in exp.local_exp[1]:
        lime_vals[i, feat_idx] = weight
np.save(os.path.join(OUT, "lime_values.npy"), lime_vals)
print(f"LIME done in {time.time() - t0:.0f}s -> {lime_vals.shape}")
print("All artifacts written to", OUT)
