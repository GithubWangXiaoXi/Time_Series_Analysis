import numpy as np
import matplotlib.pyplot as plt

class GM:

    def paramSetter(self):
        # 使用GM(1,1)模型，无参数设置
        pass

    def fit(self):
        # GM模型无需训练
        print("GM fit")

    def predict(self, X, K):

        print("GM predict")

        """
            Return the predictions through GM(1, 1).
    
            Parameters
            ----------
            X_0 : np.ndarray
                Raw data, a one-dimensional array.
            K : int
                number of predictions.
            display : bool
                Whether to display the results.
    
            Returns
            -------
            X_0_pred[N:] : np.ndarray
                predictions, a one-dimensional array.

        """
        X_0 = X
        N = X_0.shape[0]
        X_1 = np.cumsum(X_0)

        B = np.ones((N - 1, 2))
        B[:, 0] = [-(X_1[i] + X_1[i + 1]) / 2 for i in range(N - 1)]
        y = X_0[1:]

        assert np.linalg.det(np.matmul(B.transpose(), B)) != 0
        a, u = np.matmul(
            np.matmul(np.linalg.inv(np.matmul(B.transpose(), B)), B.transpose()),
            y
        )

        pred_range = np.arange(N + K)
        X_1_pred = (X_1[0] - u / a) * np.exp(-a * pred_range) + u / a
        X_0_pred = np.insert(np.diff(X_1_pred), 0, X_0[0])
        # print(X_1_pred)
        # print(X_0_pred)

        return X_0_pred[N:]



