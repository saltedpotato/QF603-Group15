# to compute for the ensemble model

class EnsembleModel:
  def __init__(self, estimators):
    self.estimators = estimators

  def compute_weightage(self, qlike, eps=1e-12): # weightage computed by using inverse qlike
    inverse = {k: 1.0/max(v,eps) for k,v in qlike.items()}
    total = sum(inverse.values()) if inverse else 0.0
    weight = {k: v/total for k,v in inverse.items()}
    return weight
