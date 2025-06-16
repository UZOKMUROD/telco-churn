# Evaluation metrics

def evaluate(y_true, y_pred):
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))
