from ..utils.metrics import compute_metrics

def evaluate_demo():
    y_true=[0,1,0,1,0,0,1,0,1,0]
    y_score=[0.1,0.8,0.2,0.7,0.3,0.4,0.9,0.2,0.6,0.1]
    return compute_metrics(y_true,y_score)
