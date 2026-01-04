from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import StackingClassifier

class LateFusionStacker(BaseEstimator, ClassifierMixin):
    """
    Late Fusion Stacker for meta-learning ensemble.
    Uses scikit-learn's StackingClassifier to combine predictions of base estimators.
    """
    def __init__(self, base_learners, meta_learner, cv=5, n_jobs=None):
        """
        Args:
            base_learners (list): List of (str, estimator) tuples for the base models.
            meta_learner: Estimator object used for meta-learning (the stacker).
            cv (int): Number of folds for cross-validation.
            n_jobs (int, optional): Number of jobs for parallel processing. Default is None.
        """
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.cv = cv
        self.n_jobs = n_jobs
        self.model_ = None

    def fit(self, X, y):
        self.model_ = StackingClassifier(
            estimators=self.base_learners,
            final_estimator=clone(self.meta_learner),
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise Exception("The model is not fitted yet!")
        return self.model_.predict(X)

    def predict_proba(self, X):
        if self.model_ is None:
            raise Exception("The model is not fitted yet!")
        return self.model_.predict_proba(X)
