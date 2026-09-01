"""Quick separability probe on cached DID features: class-conditional means of
d1 / d2 energy, plus a logistic-regression baseline on 6 summary stats.
"""
import glob, os, sys
import numpy as np

cache = sys.argv[1] if len(sys.argv) > 1 else "cache/wildfake"
split = sys.argv[2] if len(sys.argv) > 2 else "train"

X, y = [], []
for label, yi in (("real", 0), ("fake", 1)):
    for p in glob.glob(os.path.join(cache, split, "clean", label, "*.npz")):
        z = np.load(p)
        d1 = z["d1"].astype(np.float32); d2 = z["d2"].astype(np.float32)
        X.append([d1.mean(), d1.std(), np.abs(d2).mean(), d2.mean(), d2.std(),
                  np.percentile(d1, 90)])
        y.append(yi)
X = np.array(X); y = np.array(y)
print(f"n={len(y)}  real={np.sum(y==0)}  fake={np.sum(y==1)}")
names = ["d1.mean", "d1.std", "|d2|.mean", "d2.mean", "d2.std", "d1.p90"]
for j, nm in enumerate(names):
    print(f"  {nm:10s} real={X[y==0,j].mean():.4f}  fake={X[y==1,j].mean():.4f}")

if len(set(y)) == 2 and len(y) > 20:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    s = cross_val_score(clf, X, y, cv=5)
    print(f"5-fold LogReg acc on 6 summary stats: {s.mean():.3f} +- {s.std():.3f}")
