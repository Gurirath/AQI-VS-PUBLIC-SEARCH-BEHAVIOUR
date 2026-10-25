import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import features as F
from .config import CITIES, HEADLINE_TERMS

SEED = 42
TEST_START = "2024-01-01"
WARMUP = 8
CV_FOLDS = 4
ROLES = ("awareness", "protection", "health")

CALENDAR = ["month_sin", "month_cos", "woy_sin", "woy_cos"]
CITY = [f"city_{c}" for c in CITIES]
AP = [f"{r}_{k}" for r in ("awareness", "protection") for k in ("z", "z_lag1", "dz", "roll4")] + \
     ["attn_mean", "attn_mean_lag1", "attn_share_up"]
HEALTH = [f"health_{k}" for k in ("z", "z_lag1", "dz", "roll4")]

FEATURE_SETS = {
    "calendar": CALENDAR + CITY,
    "health": HEALTH + CITY,
    "search": AP + CITY,
    "search+calendar": AP + CALENDAR + CITY,
    "all": AP + HEALTH + CALENDAR + CITY,
}


def past_z(s):
    mu = s.expanding(WARMUP).mean().shift(1)
    sd = s.expanding(WARMUP).std().shift(1)
    return (s - mu) / sd.replace(0, np.nan)


def block(z, name):
    return pd.DataFrame({f"{name}_z": z, f"{name}_z_lag1": z.shift(1), f"{name}_dz": z.diff(),
                         f"{name}_roll4": z.rolling(4, min_periods=2).mean()})


def city_frame(long, city):
    parts = []
    for role in ROLES:
        s = F.headline(long, city, role)["search"].astype(float)
        parts.append(block(past_z(s), role))
    ap = long[long["role"] != "health"]
    summ = F.series_summary(ap)
    keep = summ[(summ["city"] == city) & summ["dense"]]
    Z = pd.concat([past_z(F.pair(ap, city, r.term, r.geo_level)["search"].astype(float)).rename(r.label)
                   for r in keep.itertuples()], axis=1)
    pm = F.headline(long, city, "awareness")["pm25_mean"]
    df = pd.concat(parts, axis=1)
    df["attn_mean"] = Z.mean(axis=1)
    df["attn_mean_lag1"] = df["attn_mean"].shift(1)
    df["attn_share_up"] = (Z > 0).mean(axis=1).where(Z.notna().any(axis=1))
    df["pm25"] = pm
    df["pm25_lag1"] = pm.shift(1)
    woy = df.index.isocalendar().week.astype(float).values
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
    df["city"] = city
    return df


def build_dataset(long):
    frames = []
    for city in CITIES:
        df = city_frame(long, city)
        dev = df[df.index < TEST_START]
        thr = dev["pm25"].median()
        df["threshold"] = thr
        df["high"] = (df["pm25"] > thr).astype(float).where(df["pm25"].notna())
        df["persist"] = (df["pm25_lag1"] > thr).astype(float).where(df["pm25_lag1"].notna())
        frames.append(df)
    data = pd.concat(frames).rename_axis("week").reset_index()
    for c in CITIES:
        data[f"city_{c}"] = (data["city"] == c).astype(float)
    data = data.dropna(subset=["high"] + AP + HEALTH).reset_index(drop=True)
    data["split"] = np.where(data["week"] < TEST_START, "dev", "test")
    return data.sort_values(["week", "city"]).reset_index(drop=True)


def _p(est):
    return make_pipeline(StandardScaler(), est)


def _rf(**kw):
    return RandomForestClassifier(random_state=SEED, n_jobs=1, **kw)


def _xgb(**kw):
    return XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss", verbosity=0, **kw)


def _lgbm(**kw):
    return LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1, **kw)


def _cat(**kw):
    return CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=1, allow_writing_files=False, **kw)


def _stack(est, final=None, passthrough=False):
    return StackingClassifier(est, final_estimator=LogisticRegression(max_iter=2000) if final is None else final,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), passthrough=passthrough, n_jobs=1)


def model_zoo():
    lr = lambda: LogisticRegression(max_iter=2000)
    svm = lambda: SVC(probability=True, random_state=SEED)
    z = []
    add = lambda fam, name, est: z.append((fam, name, est))
    add("Single", "Logistic regression", lr())
    add("Single", "Logistic regression (L1)", LogisticRegression(penalty="l1", solver="liblinear", C=0.5))
    add("Single", "Logistic regression (C=0.1)", LogisticRegression(C=0.1, max_iter=2000))
    add("Single", "Ridge classifier", RidgeClassifier())
    add("Single", "Linear discriminant analysis", LinearDiscriminantAnalysis())
    add("Single", "Gaussian naive Bayes", GaussianNB())
    add("Single", "Quadratic discriminant analysis", QuadraticDiscriminantAnalysis(reg_param=0.5))
    add("Single", "k-nearest neighbours (k=5)", KNeighborsClassifier(5))
    add("Single", "k-nearest neighbours (k=15)", KNeighborsClassifier(15))
    add("Single", "SVM (RBF kernel)", svm())
    add("Single", "SVM (linear)", LinearSVC(C=0.5, max_iter=5000))
    add("Single", "Decision tree (depth 3)", DecisionTreeClassifier(max_depth=3, random_state=SEED))
    add("Single", "Decision tree (depth 6)", DecisionTreeClassifier(max_depth=6, random_state=SEED))
    add("Single", "MLP (16 units)", MLPClassifier((16,), alpha=1e-2, max_iter=2000, random_state=SEED))
    add("Single", "MLP (64-32 units)", MLPClassifier((64, 32), alpha=1e-2, max_iter=2000, random_state=SEED))
    tree3 = lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED)
    add("Bagging", "Bagged decision trees", BaggingClassifier(DecisionTreeClassifier(random_state=SEED), 200, random_state=SEED))
    add("Bagging", "Random subspace trees", BaggingClassifier(DecisionTreeClassifier(random_state=SEED), 200, max_features=0.5, bootstrap=False, random_state=SEED))
    add("Bagging", "Pasting decision trees", BaggingClassifier(DecisionTreeClassifier(random_state=SEED), 200, max_samples=0.6, bootstrap=False, random_state=SEED))
    add("Bagging", "Bagged logistic regression", BaggingClassifier(lr(), 100, random_state=SEED))
    add("Bagging", "Bagged k-NN", BaggingClassifier(KNeighborsClassifier(15), 100, random_state=SEED))
    add("Bagging", "Bagged SVM", BaggingClassifier(SVC(random_state=SEED), 30, max_samples=0.7, random_state=SEED))
    add("Bagging", "Random forest (300, depth 4)", _rf(n_estimators=300, max_depth=4, min_samples_leaf=5))
    add("Bagging", "Random forest (500, full depth)", _rf(n_estimators=500))
    add("Bagging", "Random forest (balanced)", _rf(n_estimators=300, max_depth=6, class_weight="balanced"))
    add("Bagging", "Random forest (entropy)", _rf(n_estimators=300, max_depth=6, criterion="entropy"))
    add("Bagging", "Extra trees (300)", ExtraTreesClassifier(300, random_state=SEED))
    add("Bagging", "Extra trees (500, depth 6)", ExtraTreesClassifier(500, max_depth=6, random_state=SEED))
    add("Boosting", "AdaBoost (stumps)", AdaBoostClassifier(n_estimators=200, random_state=SEED))
    add("Boosting", "AdaBoost (depth-3 trees)", AdaBoostClassifier(tree3(), n_estimators=200, learning_rate=0.5, random_state=SEED))
    add("Boosting", "AdaBoost (logistic regression)", AdaBoostClassifier(lr(), n_estimators=50, random_state=SEED))
    add("Boosting", "Gradient boosting (100)", GradientBoostingClassifier(random_state=SEED))
    add("Boosting", "Gradient boosting (300, lr 0.05)", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED))
    add("Boosting", "Stochastic gradient boosting", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, subsample=0.7, random_state=SEED))
    add("Boosting", "Gradient boosting (stumps, 500)", GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=1, random_state=SEED))
    add("Boosting", "Histogram gradient boosting", HistGradientBoostingClassifier(random_state=SEED))
    add("Boosting", "Histogram gradient boosting (L2)", HistGradientBoostingClassifier(l2_regularization=1.0, max_depth=4, learning_rate=0.05, random_state=SEED))
    add("Boosting", "XGBoost (depth 3)", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05))
    add("Boosting", "XGBoost (depth 6)", _xgb(n_estimators=300, max_depth=6, learning_rate=0.05))
    add("Boosting", "XGBoost (regularised)", _xgb(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, reg_lambda=5))
    add("Boosting", "XGBoost random forest", _xgb(n_estimators=1, num_parallel_tree=300, max_depth=6, learning_rate=1.0, subsample=0.8, colsample_bynode=0.8))
    add("Boosting", "LightGBM (31 leaves)", _lgbm(n_estimators=300, learning_rate=0.05))
    add("Boosting", "LightGBM (15 leaves)", _lgbm(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10))
    add("Boosting", "LightGBM (DART)", _lgbm(n_estimators=300, learning_rate=0.05, boosting_type="dart"))
    add("Boosting", "LightGBM (GOSS)", _lgbm(n_estimators=300, learning_rate=0.05, data_sample_strategy="goss"))
    add("Boosting", "CatBoost (depth 4)", _cat(iterations=300, depth=4, learning_rate=0.05))
    add("Boosting", "CatBoost (depth 6)", _cat(iterations=300, depth=6, learning_rate=0.05))
    add("Voting", "Hard vote: LR, RF, GB", VotingClassifier([("lr", lr()), ("rf", _rf(n_estimators=300, max_depth=6)), ("gb", GradientBoostingClassifier(random_state=SEED))], voting="hard"))
    add("Voting", "Soft vote: LR, RF, GB", VotingClassifier([("lr", lr()), ("rf", _rf(n_estimators=300, max_depth=6)), ("gb", GradientBoostingClassifier(random_state=SEED))], voting="soft"))
    add("Voting", "Soft vote: SVM, k-NN, RF", VotingClassifier([("svm", svm()), ("knn", KNeighborsClassifier(15)), ("rf", _rf(n_estimators=300, max_depth=6))], voting="soft"))
    add("Voting", "Soft vote: RF, ET, HGB", VotingClassifier([("rf", _rf(n_estimators=300, max_depth=6)), ("et", ExtraTreesClassifier(300, max_depth=6, random_state=SEED)), ("hgb", HistGradientBoostingClassifier(random_state=SEED))], voting="soft"))
    add("Voting", "Soft vote: XGB, LGBM, CatBoost", VotingClassifier([("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05)), ("lgbm", _lgbm(n_estimators=300, learning_rate=0.05, num_leaves=15)), ("cat", _cat(iterations=300, depth=4, learning_rate=0.05))], voting="soft"))
    add("Voting", "Soft vote: LR, SVM, RF, XGB, CatBoost", VotingClassifier([("lr", lr()), ("svm", svm()), ("rf", _rf(n_estimators=300, max_depth=6)), ("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05)), ("cat", _cat(iterations=300, depth=4, learning_rate=0.05))], voting="soft"))
    add("Voting", "Hard vote: LR, SVM, RF, XGB, CatBoost", VotingClassifier([("lr", lr()), ("svm", svm()), ("rf", _rf(n_estimators=300, max_depth=6)), ("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05)), ("cat", _cat(iterations=300, depth=4, learning_rate=0.05))], voting="hard"))
    add("Voting", "Weighted soft vote: RF, XGB, LR", VotingClassifier([("rf", _rf(n_estimators=300, max_depth=6)), ("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05)), ("lr", lr())], voting="soft", weights=[2, 2, 1]))
    add("Stacking", "Stack: LR, RF, GB → LR", _stack([("lr", lr()), ("rf", _rf(n_estimators=300, max_depth=6)), ("gb", GradientBoostingClassifier(random_state=SEED))]))
    add("Stacking", "Stack: LR, k-NN, SVM → RF", _stack([("lr", lr()), ("knn", KNeighborsClassifier(15)), ("svm", svm())], _rf(n_estimators=300, max_depth=3)))
    add("Stacking", "Stack: XGB, LGBM, CatBoost → LR", _stack([("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05)), ("lgbm", _lgbm(n_estimators=300, learning_rate=0.05, num_leaves=15)), ("cat", _cat(iterations=300, depth=4, learning_rate=0.05))]))
    add("Stacking", "Stack: RF, ET, bagged trees → LR", _stack([("rf", _rf(n_estimators=300, max_depth=6)), ("et", ExtraTreesClassifier(300, max_depth=6, random_state=SEED)), ("bag", BaggingClassifier(DecisionTreeClassifier(random_state=SEED), 100, random_state=SEED))]))
    add("Stacking", "Stack: LR, SVM, RF, XGB → LR", _stack([("lr", lr()), ("svm", svm()), ("rf", _rf(n_estimators=300, max_depth=6)), ("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05))]))
    add("Stacking", "Stack with passthrough: LR, RF, XGB → LR", _stack([("lr", lr()), ("rf", _rf(n_estimators=300, max_depth=6)), ("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05))], passthrough=True))
    add("Stacking", "Stack: LR, RF, SVM → GB", _stack([("lr", lr()), ("rf", _rf(n_estimators=300, max_depth=6)), ("svm", svm())], GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=SEED)))
    add("Stacking", "Stack: MLP, SVM, RF → LR", _stack([("mlp", MLPClassifier((16,), alpha=1e-2, max_iter=2000, random_state=SEED)), ("svm", svm()), ("rf", _rf(n_estimators=300, max_depth=6))]))
    add("Stacking", "Stack: NB, k-NN, tree → LR", _stack([("nb", GaussianNB()), ("knn", KNeighborsClassifier(15)), ("dt", tree3())]))
    add("Stacking", "Stack: LR, SVM, RF, ET, XGB, CatBoost → LR", _stack([("lr", lr()), ("svm", svm()), ("rf", _rf(n_estimators=300, max_depth=6)), ("et", ExtraTreesClassifier(300, max_depth=6, random_state=SEED)), ("xgb", _xgb(n_estimators=300, max_depth=3, learning_rate=0.05)), ("cat", _cat(iterations=300, depth=4, learning_rate=0.05))]))
    return [(fam, name, _p(est)) for fam, name, est in z]


def scores(est, X):
    for fn in ("predict_proba", "decision_function"):
        try:
            out = getattr(est, fn)(X)
            return out[:, 1] if out.ndim == 2 else out
        except AttributeError:
            continue
    return est.predict(X).astype(float)


def metrics(y, pred, score):
    out = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
           "f1": f1_score(y, pred)}
    try:
        out["auc"] = roc_auc_score(y, score)
    except ValueError:
        out["auc"] = np.nan
    return out


def time_folds(data, n=CV_FOLDS):
    weeks = np.sort(data["week"].unique())
    edges = np.linspace(0, len(weeks), n + 2).astype(int)
    for k in range(1, n + 1):
        train_w = weeks[:edges[k]]
        test_w = weeks[edges[k]:edges[k + 1]]
        yield data.index[data["week"].isin(train_w)], data.index[data["week"].isin(test_w)]


def fit_predict(est, Xtr, ytr, Xte):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = clone(est).fit(Xtr, ytr)
        return m, m.predict(Xte), scores(m, Xte)


def cross_validate(data, zoo, feature_sets=FEATURE_SETS, log=print):
    dev = data[data["split"] == "dev"].reset_index(drop=True)
    folds = list(time_folds(dev))
    rows, fold_rows = [], []
    for fs, cols in feature_sets.items():
        for fam, name, est in zoo:
            t0 = time.time()
            per = []
            for k, (tr, te) in enumerate(folds):
                _, pred, sc = fit_predict(est, dev.loc[tr, cols], dev.loc[tr, "high"], dev.loc[te, cols])
                m = metrics(dev.loc[te, "high"], pred, sc)
                per.append(m)
                fold_rows.append({"feature_set": fs, "family": fam, "model": name, "fold": k + 1, **m})
            agg = pd.DataFrame(per).mean().to_dict()
            rows.append({"feature_set": fs, "family": fam, "model": name, **{f"cv_{k}": v for k, v in agg.items()},
                         "cv_accuracy_sd": pd.DataFrame(per)["accuracy"].std(), "seconds": time.time() - t0})
        log(f"{fs}: done")
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def test_evaluate(data, zoo, feature_sets=FEATURE_SETS):
    dev, test = data[data["split"] == "dev"], data[data["split"] == "test"]
    rows, preds, fitted = [], {}, {}
    for fs, cols in feature_sets.items():
        for fam, name, est in zoo:
            m, pred, sc = fit_predict(est, dev[cols], dev["high"], test[cols])
            rows.append({"feature_set": fs, "family": fam, "model": name, **{f"test_{k}": v for k, v in metrics(test["high"], pred, sc).items()}})
            preds[(fs, name)] = (pred, sc)
            fitted[(fs, name)] = m
    return pd.DataFrame(rows), preds, fitted


def bootstrap_ci(y, pred, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    acc = [(y[i] == pred[i]).mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(acc, [2.5, 97.5])


def mcnemar(y, a, b):
    y, a, b = map(np.asarray, (y, a, b))
    ra, rb = a == y, b == y
    n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p": p}


def friedman(fold_table, feature_set, models):
    t = fold_table[(fold_table["feature_set"] == feature_set) & fold_table["model"].isin(models)]
    wide = t.pivot(index="fold", columns="model", values="accuracy")[models]
    stat, p = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
    return stat, p, wide.rank(axis=1, ascending=False).mean().sort_values()


def per_city(data, pred):
    test = data[data["split"] == "test"].assign(pred=pred)
    return test.groupby("city").apply(lambda g: pd.Series({"weeks": len(g), "accuracy": (g["pred"] == g["high"]).mean(),
                                                           "share_high": g["high"].mean()}), include_groups=False)


def importance(model, data, cols, repeats=50):
    test = data[data["split"] == "test"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(model, test[cols], test["high"], n_repeats=repeats, random_state=SEED, scoring="accuracy")
    return pd.DataFrame({"feature": cols, "mean": r.importances_mean, "sd": r.importances_std}).sort_values("mean", ascending=False)
