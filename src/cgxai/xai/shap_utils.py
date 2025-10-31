import numpy as np

def simple_shap_like(feature_values):
    import numpy as np
    vals=np.abs(feature_values - np.mean(feature_values))
    s=vals.sum(); return (vals/(s+1e-8)).tolist()
