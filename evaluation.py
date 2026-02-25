from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

def evaluate_model(y_true, y_pred, y_scores):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_scores))
    print("PR-AUC:", average_precision_score(y_true, y_scores))
