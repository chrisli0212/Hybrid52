
import sys, os, joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

PROD_ROOT = Path("/workspace/Final_production_model")
MODEL_DIR = PROD_ROOT / "models"
SAVE_PATH = MODEL_DIR / "stage3/stage3_logreg_2026.joblib"

print("Loading predictions.npz...")
d = np.load("/workspace/Final_production_model/predictions.npz", allow_pickle=True)
agent_mat = d['agent_mat']   # (14069, 7)
labels    = d['labels']
print(f"agent_mat: {agent_mat.shape}, UP%: {labels.mean()*100:.1f}%")

# Also load val split agent_mat by re-running on val
# For now use 60/40 stratified split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    agent_mat, labels, test_size=0.4, random_state=42, stratify=labels)
print(f"Train: {len(y_train)} UP%={y_train.mean()*100:.1f}%")
print(f"Test:  {len(y_test)}  UP%={y_test.mean()*100:.1f}%")

def build_meta(mat):
    _mean   = mat.mean(axis=1, keepdims=True)
    _std    = mat.std(axis=1, keepdims=True)
    _spread = mat.max(axis=1, keepdims=True) - mat.min(axis=1, keepdims=True)
    _maj    = (mat > 0.5).mean(axis=1, keepdims=True).astype(np.float32)
    _max    = mat.max(axis=1, keepdims=True)
    _min    = mat.min(axis=1, keepdims=True)
    return np.concatenate([mat, _mean, _std, _spread, _maj, _max, _min], axis=1)

X_tr = build_meta(X_train)
X_te = build_meta(X_test)

# class_weight='balanced' prevents always-UP prediction
logreg = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs',
                             class_weight='balanced', random_state=42)
logreg.fit(X_tr, y_train)
val_probs = logreg.predict_proba(X_tr)[:,1]

# find best threshold on train
best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.30, 0.65, 0.01):
    preds_t = (val_probs > t).astype(int)
    # require both classes predicted
    if len(np.unique(preds_t)) < 2: continue
    f = f1_score(y_train, preds_t)
    if f > best_f1:
        best_f1, best_thresh = f, t
print(f"Best threshold: {best_thresh:.2f}")

test_probs = logreg.predict_proba(X_te)[:,1]
test_preds = (test_probs > best_thresh).astype(int)

print(f"\n── New LogReg 2026 (balanced) ──")
print(f"Accuracy: {accuracy_score(y_test, test_preds)*100:.2f}%")
print(f"F1:       {f1_score(y_test, test_preds):.4f}")
print(classification_report(y_test, test_preds, target_names=['DOWN','UP']))

# compare old
old_meta  = build_meta(agent_mat)
old_model = joblib.load(MODEL_DIR / "stage3/stage3_logreg.joblib")
old_preds = (old_model.predict_proba(old_meta[len(X_train):])[:,1] > 0.36).astype(int)
print(f"── Old LogReg baseline ──")
print(f"Accuracy: {accuracy_score(y_test, old_preds)*100:.2f}%")
print(f"F1:       {f1_score(y_test, old_preds):.4f}")

os.makedirs(MODEL_DIR / "stage3", exist_ok=True)
joblib.dump(logreg, SAVE_PATH)
np.save(MODEL_DIR / "stage3/threshold_2026.npy", np.array([best_thresh]))
print(f"\nSaved: {SAVE_PATH}")
