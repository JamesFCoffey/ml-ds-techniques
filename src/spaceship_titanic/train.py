from sklearn import pipeline, preprocessing

from .featurize import SpaceshipTransformer


# Build a pipeline with/without scaling
def make_pipe(feat, model, scale_needed=False):
    steps = [("feat", feat)]
    if scale_needed:
        # StandardScaler(with_mean=False) keeps sparse → dense explosions away
        steps.append(("sc", preprocessing.StandardScaler(with_mean=False)))
    steps.append(("clf", model))
    return pipeline.Pipeline(steps)


# reuse one transformer instance per fold
def make_fold_data(tr_idx, va_idx, X, y):
    tfm = SpaceshipTransformer(min_freq=0.01)
    X_tr = tfm.fit_transform(X.iloc[tr_idx])
    X_va = tfm.transform(X.iloc[va_idx])
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    return X_tr, X_va, y_tr, y_va
