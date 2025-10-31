from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_metrics(y_true,y_score,threshold=0.5):
    return {'auroc':roc_auc_score(y_true,y_score), 'auprc':average_precision_score(y_true,y_score), f'f1@{threshold}': f1_score(y_true,(y_score>=threshold).astype(int))}
