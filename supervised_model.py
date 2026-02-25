from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf
