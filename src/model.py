from xgboost import XGBClassifier


def get_model():
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_thread=-1,
        eval_metric="logloss"
    )
    return model


