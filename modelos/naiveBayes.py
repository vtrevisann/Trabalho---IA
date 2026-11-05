import lib

class NaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X_train, y_train):
        self.model = lib.GaussianNB(
            var_smoothing=self.var_smoothing
        )

        self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)