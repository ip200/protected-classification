import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


def gen_alpha(ind, size, alpha):
    """ Calculates Alpha parameters for the Cox calibrator

            Parameters
            ----------
            ind : int
                Integer corresponding to a given one-hot encoded class

            size : int
                Total number of classes

            alpha : int
                alpha parameter

            Returns
            ----------
            alphas: list
                A list of alpha parameters for the Cox calibrator
        """
    alp = np.zeros(size)
    alp[ind] = alpha
    alphas = list(alp)
    return alphas


def cox_multiclass(p, alpha, beta):
    """ Generates Cox calibration probability outputs

        For more details of the use of Cox calibrating functions in Protected calibration see Section 2 in [1]
        The original description of Cox calibrating functions can be found in Section 3 in [2]

            Parameters
            ----------
            p : {array-like}, shape (n_samples, n_classes)
                A set of probability outputs

            alpha : float
                Alpha parameter for the Cox calibrator

            beta : float
                Beta parameter for the Cox calibrator

            Returns
            ----------
            cox: {array-like}
                Cox calibrator probability outputs

            References
            ----------
            [1] Vovk, Vladimir, Ivan Petej, and Alex Gammerman. "Protected probabilistic classification."
            In Conformal and Probabilistic Prediction and Applications, pp. 297-299. PMLR, 2021.
            (arxiv version https://arxiv.org/pdf/2107.01726.pdf)
            [2] Cox, David R. Two further applications of a model for binary regression. Biometrika, 45:562–565, 1958.

        """
    cox = [prob ** beta for prob in p] * np.exp(alpha)
    cox = cox / np.sum(cox)
    return cox


def y_encode(y):
    """ Calculates Brier loss for multi-class probability outputs

            Parameters
            ----------
            y : {array-like}, shape (n_samples)
                A set of class labels


            Returns
            ----------
            y_encoded: {array-like}, shape (n_samples, n_classes)
                One-hot-encoded values of y
        """
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1, 1))
    y_encoded = enc.transform(y.reshape(-1, 1)).toarray()
    return y_encoded


def brier_loss(y, p):
    """ Calculates Brier loss for multi-class probability outputs

        Parameters
        ----------
        y : {array-like}, shape (n_samples, n_classes)
            A set of one hot encoded class labels

        p : {array-like}, shape (n_samples, n_classes)
            A set of class probabilities

        Returns
        ----------
        loss: float
            Mean Brier loss
    """
    loss = np.mean(np.sum((y - p) ** 2, axis=1))
    return loss


def log_mean(x):
    m = np.max(x)
    return m + np.log10(np.mean(np.exp(np.log(10) * (x - m))))


class ProtectedClassification:
    """ A wrapper for Protected Probabilistic Classification

            A class implementing Protected Probabilistic Classification as described in [1].

            For more details of usage see Examples below.

            Parameters
            __________

            estimator : sci-kit learn estimator instance, default=None
                The classifier whose output needs to be calibrated

            inductive: bool, default = False
                Flag specifying whether to return martingale statistics instead of calibrated probabilities

            alphas : {array-like}, default = None
                Alpha parameters of Cox calibration functions (see Section 2 in [1])

            betas : {array-like}, shape (n_samples,), default = None
                Beta parameters of Cox calibration functions (see Section 2 in [1])

            jumping_rates: {array-like}, default = None
                Jumping rates for the Composite Jumper algorithm described in [1]

            pi : float, default = None
                Passive capital parameter for the Composite Jumper algorithm described in [1]


            References
            ----------
             [1] Vovk, Vladimir, Ivan Petej, and Alex Gammerman. "Protected probabilistic classification."
            In Conformal and Probabilistic Prediction and Applications, pp. 297-299. PMLR, 2021.
            (arxiv version https://arxiv.org/pdf/2107.01726.pdf)


             Examples
            --------

    """

    def __init__(
            self,
            estimator=None,
            alphas=None,
            betas=None,
            jumping_rates=None,
            pi=None,
    ):
        self.estimator = estimator
        self.alphas = alphas
        self.betas = betas
        self.jumping_rates = jumping_rates
        self.pi = pi
        self.cal_probs = None
        self.classes = None
        self.active_weight = None
        self.passive_weight = None
        self.log_sj_martingale = None
        self.log_cj_martingale = None
        self.mart_capital = None
        self.n_classes = None
        self.return_probs = None

        if self.betas is None:
            self.betas = [1, 0.5, 2]

        if self.jumping_rates is None:
            self.jumping_rates = [10 ** (-2), 10 ** (-3), 10 ** (-4)]

        if self.pi is None:
            self.pi = 0.5

    def calibrate_probs(self,
                        p_pred,
                        y
                        ):
        """ Calculates protected calibrated probabilities.

                    Parameters
                    ----------
                    p_pred : {array-like}, shape (n_samples, n_classes)
                        A set of probabilities to be calibrated

                    y : {array-like}, shape (n_samples,)
                        Associated class labels

                    Returns
                    ----------
                    p_prime: {array-like}, shape (n_samples,)
                        Calibrated probabilities

                    References
                    __________
                    [1] Vovk, Vladimir, Ivan Petej, and Alex Gammerman. "Protected probabilistic classification."
                    In Conformal and Probabilistic Prediction and Applications, pp. 297-299. PMLR, 2021.
                    (arxiv version https://arxiv.org/pdf/2107.01726.pdf)
                    """
        if y is not None:
            self.classes = np.unique(y)

        self.n_classes = p_pred.shape[1]

        if self.alphas is None:
            self.alphas = \
                [list(np.zeros(self.n_classes))] + \
                [gen_alpha(ind, self.n_classes, 1) for ind in [self.n_classes - 1]] + \
                [gen_alpha(ind, self.n_classes, -1) for ind in [self.n_classes - 1]]

        n_jumping_rates = len(self.jumping_rates)
        n_calibrators = len(self.betas) * len(self.alphas)

        prob_list = np.zeros((self.n_classes, self.n_classes))

        for i in range(self.n_classes):
            prob_list[i, i] = 1
        prob_list = list(prob_list)

        cal_list = [list((a, b)) for a in self.alphas for b in self.betas]

        if y is not None:
            n_test = len(y)
        else:
            n_test = p_pred.shape[0]

        # Defining storage for martingales
        if self.log_sj_martingale is None:
            self.log_sj_martingale = np.zeros((n_jumping_rates, n_test + 1))
        if self.log_cj_martingale is None:
            self.log_cj_martingale = np.zeros(n_test + 1)

        # the normalized capital in each state (after normalization at the previous step)
        if self.mart_capital is None:
            self.mart_capital = np.zeros((n_jumping_rates, n_calibrators))
            self.mart_capital[:, 0] = 1

        # Processing the dataset
        if self.passive_weight is None:
            self.passive_weight = self.pi  # amount set aside (passive weight)
        if self.active_weight is None:
            self.active_weight = np.zeros((n_jumping_rates, n_calibrators))  # the weight of each active state
            self.active_weight[:, 0] = (1 - self.pi) / n_jumping_rates  # initial weights

        if y is not None:
            y_encoded = y_encode(y)
            p_prime = np.zeros((n_test, self.n_classes))
            for n in range(n_test):  # going through all test observations
                # Jump-mixing starts
                for j_index in range(n_jumping_rates):
                    capital = np.sum(self.active_weight[j_index, :])  # active capital for this jumping rate
                    j_rate = self.jumping_rates[j_index]
                    self.active_weight[j_index, :] = (1 - j_rate) * self.active_weight[j_index, :] \
                                                    + capital * j_rate / n_calibrators
                    self.mart_capital[j_index, :] = (1 - j_rate) * self.mart_capital[j_index, :] \
                                                    + (j_rate / n_calibrators)
                # Jump-mixing ends
                g = np.empty(self.n_classes)  # pseudo prediction initialized
                for prob_index, prob_i in enumerate(prob_list):
                    # initializing the pseudo prediction to its passive component
                    g[prob_index] = self.passive_weight \
                                    * np.exp(-brier_loss(np.array(prob_i).reshape(1, -1), p_pred[n]))
                    for k in range(n_calibrators):
                        # prediction calibrated by the k-th calibrator
                        cal_pp_k = cox_multiclass(p_pred[n], cal_list[k][0], cal_list[k][1])
                        for j_index in range(n_jumping_rates):
                            # accumulating predictions calibrated by the calibrators
                            g[prob_index] += self.active_weight[j_index, k] * np.exp(
                                -brier_loss(np.array(prob_i).reshape(1, -1), cal_pp_k))
                    g[prob_index] = -np.log(g[prob_index])
                # We need to solve equation for s, let's first try a shortcut:
                s = (2 + np.sum(g)) / self.n_classes
                for k_index in range(self.n_classes):
                    p_prime[n, k_index] = (s - g[k_index]) / 2  # my prediction
                # Updating the weights:
                # updating the passive capital
                self.passive_weight *= np.exp(-brier_loss(y_encoded[n].reshape(1, -1), p_pred[n]))
                for k in range(n_calibrators):
                    # base prediction calibrated by the k-th calibrator
                    cal_pp_k = cox_multiclass(p_pred[n], cal_list[k][0], cal_list[k][1])
                    for j_index in range(n_jumping_rates):
                        # updating the active capital
                        self.active_weight[j_index, k] *= np.exp(-brier_loss(y_encoded[n].reshape(1, -1), cal_pp_k))
                        self.mart_capital[j_index, k] *= \
                            np.exp(-brier_loss(y_encoded[n].reshape(1, -1), cal_pp_k)) / \
                            np.exp(-brier_loss(y_encoded[n].reshape(1, -1), p_pred[n]))
                for j_index in range(n_jumping_rates):
                    self.log_sj_martingale[j_index, n + 1] = self.log_sj_martingale[j_index, n] + \
                                                        np.log10(np.sum(self.mart_capital[j_index, :]))
                    self.mart_capital[j_index, :] /= np.sum(self.mart_capital[j_index, :])
                # Normalizing at each step (not needed):
                capital = self.passive_weight + np.sum(self.active_weight[:, :])  # the overall weight
                self.passive_weight /= capital  # normalization of the passive weight
                self.active_weight[:, :] /= capital  # normalization of the active weights

            if np.sum(p_prime < 0) > 0:
                p_prime[p_prime < 0] = 0
            if np.sum(p_prime > 1) > 0:
                p_prime[p_prime > 1] = 1

            for n in range(n_test + 1):
                self.log_cj_martingale[n] = log_mean([0, log_mean(self.log_sj_martingale[:, n])])

        else:
            n_test = p_pred.shape[0]
            p_prime = np.zeros((n_test, self.n_classes))
            for n in range(n_test):
                for k in range(n_calibrators):
                    # prediction calibrated by the k-th calibrator
                    cal_pp_k = cox_multiclass(p_pred[n], cal_list[k][0], cal_list[k][1])
                    for j_index in range(n_jumping_rates):
                        p_prime[n, :] += cal_pp_k * self.active_weight[j_index, k]

        p_prime = p_prime / np.repeat(p_prime.sum(axis=1).reshape(-1, 1), self.n_classes, axis=1)

        return p_prime

    def fit(self, x, y, test_probs=None):

        """ Fits the Protected calibration algorithm to an underlying dataset.

                Parameters
                ----------
                
                x : {array-like}, shape (n_samples,)
                    Test set features (if class is constructed by passing
                    an underlying sci-kit learn classifier)

                y : {array-like}, shape (n_samples,)
                    Class labels

                test_probs : {array-like}, shape (n_samples, n_classes), default = None
                    Test set probabilities (if class is constructed without passing
                    an underlying sci-kit learn classifier)


                References
                ----------
                [1] Vovk, Vladimir, Ivan Petej, and Alex Gammerman. "Protected probabilistic classification."
                In Conformal and Probabilistic Prediction and Applications, pp. 297-299. PMLR, 2021.
                (arxiv version https://arxiv.org/pdf/2107.01726.pdf)
        """

        if self.estimator is not None and x is None:
            raise Exception("Please provide a set of test examples to calibrate")

        if self.estimator is None and x is not None:
            raise Exception(
                "Please initialise the ProtectedCalibration class with an underlying classification algorithm"
            )

        if x is None and test_probs is None:
            raise Exception("Please provide either a set of test examples to calibrate or test set probabilities")

        if test_probs is None:
            try:
                check_is_fitted(self.estimator)
                test_probs = self.estimator.predict_proba(x)
            except NotFittedError:
                raise Exception("Please fit the underlying estimator first")

        self.cal_probs = self.calibrate_probs(
                    p_pred=test_probs,
                    y=y
                    )

    def predict_proba(self, x=None, y=None, test_probs=None, return_stats=False):
        """ Outputs calibrated probabilities.

            Returns
            ----------
            p_prime: {array-like}, shape (n_samples,n_classes)
                Protected calibrated probabilities
        """
        if y is not None and x is None and test_probs is None:
            raise Exception("For online protected classification please provide test points or probability outputs")
        if y is None and x is None and test_probs is None:
            if self.cal_probs is None:
                raise Exception("Please fit the ProtectedClassification algorithm first")
            p_prime = np.asarray(self.cal_probs)
        else:
            if test_probs is None:
                test_probs = self.estimator.predict_proba(x)

            self.return_probs = self.calibrate_probs(p_pred=test_probs, y=y)
            p_prime = self.return_probs

        if return_stats is False:
            return p_prime
        else:
            return p_prime, [self.log_sj_martingale, self.log_cj_martingale, self.mart_capital, self.active_weight,
                             self.passive_weight]

    def predict(self, x=None, y=None, test_probs=None, one_hot=False):

        """ Outputs calibrated predictions.

            Parameters
            ----------
            one_hot: bool, default=True
            If True returns one hot encoded labels, class labels otherwise

            Returns
            ----------
            y_pred: {array-like}, shape (n_samples, n_classes) if one_hot=True otherwise shape (n_samples)
                Protected calibrated probabilities
        """

        if self.return_probs is None:
            self.return_probs = self.predict_proba(x=x, y=y, test_probs=test_probs)

        idx = np.argmax(self.return_probs, axis=-1)
        if one_hot:
            y_pred = np.zeros(self.cal_probs.shape)
            y_pred[np.arange(y_pred.shape[0]), idx] = int(1)
        else:
            y_pred = np.array([self.classes[i] for i in idx])
        return y_pred
