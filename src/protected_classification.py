import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


def gen_alpha(ind, size, alpha):
    """ Calculates Alpha paramaters for the Cox calibrator

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
        The original decription of Cox calibrating functions can be found in Section 3 in [2]

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


def calibrate_probs(p_pred,
                    y_test,
                    alphas,
                    betas,
                    jumping_rates,
                    pi,
                    return_martingale=False,
                    inductive=False
                    ):
    """ Calculates protected calibraated probabilities.

                Parameters
                ----------
                p_pred : {array-like}, shape (n_samples, n_classes)
                    A set of probabilities to be calibrated

                y_test : {array-like}, shape (n_samples,)
                    Associated class labels

                alphas : {array-like}
                    Alpha parameters of Cox calibtration functions (see Section 2 in [1])

                betas : {array-like}, shape (n_samples,)
                    Beta parameters of Cox calibtration functions (see Section 2 in [1])

                jumping_rates: {array-like}
                    Jumping rates for the Composite Jumper algorithm described in [1]

                pi : float
                    Passive capital parameter for the Composite Jumper algorithm described in [1]

                return_martingale : bool, default = False
                    Flag specifies whether to calculate and return martingale values

                inductive: bool, default = False
                    Flag specifying whether to return active_weight and passive_weight instead of probabilities

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

    n_jumping_rates = len(jumping_rates)
    n_test = len(y_test)
    y_test_encoded = y_encode(y_test)
    n_calibrators = len(betas) * len(alphas)
    n_classes = len(np.unique(y_test))

    prob_list = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        prob_list[i, i] = 1
    prob_list = list(prob_list)

    cal_list = [list((a, b)) for a in alphas for b in betas]

    p_prime = np.empty((n_test, n_classes))
    # Defining storage for martingales
    log_sj_martingale = np.zeros((n_jumping_rates, n_test + 1))
    log_cj_martingale = np.zeros(n_test + 1)

    # the normalized capital in each state (after normalization at the previous step)
    mart_capital = np.zeros((n_jumping_rates, n_calibrators))
    mart_capital[:, 0] = 1

    # Processing the dataset
    passive_weight = pi  # amount set aside (passive weight)
    active_weight = np.zeros((n_jumping_rates, n_calibrators))  # the weight of each active state
    active_weight[:, 0] = (1 - pi) / n_jumping_rates  # initial weights
    for n in range(n_test):  # going through all test observations
        # Jump-mixing starts
        for j_index in range(n_jumping_rates):
            capital = np.sum(active_weight[j_index, :])  # active capital for this jumping rate
            j_rate = jumping_rates[j_index]
            active_weight[j_index, :] = (1 - j_rate) * active_weight[j_index, :] + capital * j_rate / n_calibrators
            mart_capital[j_index, :] = (1 - j_rate) * mart_capital[j_index, :] + (j_rate / n_calibrators)
        # Jump-mixing ends
        g = np.empty(n_classes)  # pseudo prediction initialized
        for prob_index, prob_i in enumerate(prob_list):
            # initializing the pseudo prediction to its passive component
            g[prob_index] = passive_weight * np.exp(-brier_loss(np.array(prob_i).reshape(1, -1), p_pred[n]))
            for k in range(n_calibrators):
                # prediction calibrated by the k-th calibrator
                cal_pp_k = cox_multiclass(p_pred[n], cal_list[k][0], cal_list[k][1])
                for j_index in range(n_jumping_rates):
                    # accumulating predictions calibrated by the calibrators
                    g[prob_index] += active_weight[j_index, k] * np.exp(
                        -brier_loss(np.array(prob_i).reshape(1, -1), cal_pp_k))
            g[prob_index] = -np.log(g[prob_index])
        # We need to solve equation for s, let's first try a shortcut:
        s = (2 + np.sum(g)) / n_classes
        for k_index in range(n_classes):
            p_prime[n, k_index] = (s - g[k_index]) / 2  # my prediction
        # Updating the weights:
        # updating the passive capital
        passive_weight *= np.exp(-brier_loss(y_test_encoded[n].reshape(1, -1), p_pred[n]))
        for k in range(n_calibrators):
            # base prediction calibrated by the k-th calibrator
            cal_pp_k = cox_multiclass(p_pred[n], cal_list[k][0], cal_list[k][1])
            for j_index in range(n_jumping_rates):
                # updating the active capital
                active_weight[j_index, k] *= np.exp(-brier_loss(y_test_encoded[n].reshape(1, -1), cal_pp_k))
                mart_capital[j_index, k] *= \
                    np.exp(-brier_loss(y_test_encoded[n].reshape(1, -1), cal_pp_k)) /\
                    np.exp(-brier_loss(y_test_encoded[n].reshape(1, -1), p_pred[n]))
        for j_index in range(n_jumping_rates):
            log_sj_martingale[j_index, n + 1] = log_sj_martingale[j_index, n] + \
                                                np.log10(np.sum(mart_capital[j_index, :]))
            mart_capital[j_index, :] /= np.sum(mart_capital[j_index, :])
        # Normalizing at each step (not needed):
        capital = passive_weight + np.sum(active_weight[:, :])  # the overall weight
        passive_weight /= capital  # normalization of the passive weight
        active_weight[:, :] /= capital  # normalization of the active weights

    if np.sum(p_prime < 0) > 0:
        p_prime[p_prime < 0] = 0
    if np.sum(p_prime > 1) > 0:
        p_prime[p_prime > 1] = 1

    if return_martingale:
        for n in range(n_test + 1):
            log_cj_martingale[n] = log_mean([0, log_mean(log_sj_martingale[:, n])])

    p_prime = p_prime / np.repeat(p_prime.sum(axis=1).reshape(-1, 1), n_classes, axis=1)

    if not inductive:
        if return_martingale:
            return p_prime, log_sj_martingale, log_cj_martingale, mart_capital
        else:
            return p_prime
    else:
        if return_martingale:
            return active_weight, passive_weight, log_sj_martingale, log_cj_martingale, mart_capital
        else:
            return active_weight, passive_weight


class ProtectedClassification:
    """ A wrapper for Protected Probabilistic Classification

            A class implementing Protected Probabilistic Classification as decribed in [1].

            For more details of usage see Examples below.

            Parameters
            __________

            estimator : sci-kit learn estimator instance, default=None
                The classifier whose output needs to be calibrated


            References
            ----------
             [1] Vovk, Vladimir, Ivan Petej, and Alex Gammerman. "Protected probabilistic classification."
            In Conformal and Probabilistic Prediction and Applications, pp. 297-299. PMLR, 2021.
            (arxiv version https://arxiv.org/pdf/2107.01726.pdf)


             Examples
            --------

    """

    def __init__(self, estimator=None):
        self.estimator = estimator
        self.cal_probs = None
        self.classes = None
        self.active_weight = None
        self.passive_weight = None

    def calibrate(
            self,
            y_test,
            x_test=None,
            test_probs=None,
            alphas=None,
            betas=None,
            jumping_rates=None,
            pi=None,
            return_martingale=False
    ):

        """ Fits the Protected calibration algorithm to an underlying test set.

                Parameters
                ----------

                y_test : {array-like}, shape (n_samples,)
                    Class labels

                x_test : {array-like}, shape (n_samples,), default = None
                    Test set features (if class is constructed by passing
                    an underlying sci-kit learn classifier)

                test_probs : {array-like}, shape (n_samples, n_classes), default = None
                    Test set probabilities (if class is constructed without passing
                    an underlying sci-kit learn classifier)

                alphas : {array-like}, default = None
                    Alpha parameters of Cox calibtration functions (see Section 2 in [1])

                betas : {array-like}, shape (n_samples,), default = None
                    Beta parameters of Cox calibtration functions (see Section 2 in [1])

                jumping_rates: {array-like}, default = None
                    Jumping rates for the Composite Jumper algorithm described in [1]

                pi : float, default = None
                    Passive capital parameter for the Composite Jumper algorithm described in [1]

                return_martingale : bool, default = False
                    Flag specifies whether to calculate and return martingale values


                References
                ----------
                [1] Vovk, Vladimir, Ivan Petej, and Alex Gammerman. "Protected probabilistic classification."
                In Conformal and Probabilistic Prediction and Applications, pp. 297-299. PMLR, 2021.
                (arxiv version https://arxiv.org/pdf/2107.01726.pdf)
        """

        if self.estimator is not None and x_test is None:
            raise Exception("Please provide a set of test examples to calibrate")

        if self.estimator is None and x_test is not None:
            raise Exception(
                "Please initialise the ProtectedCalibration class with an underlying classification algorithm"
            )

        if x_test is None and test_probs is None:
            raise Exception("Please provide either a set of test examples to calibrate or test set probabilities")

        if betas is None:
            betas = [1, 0.5, 2]
        if test_probs is None:
            try:
                check_is_fitted(self.estimator)
                test_probs = self.estimator.predict_proba(x_test)
            except NotFittedError:
                raise Exception("Please fit the underlying calibrator on the training set first")

        if jumping_rates is None:
            jumping_rates = [10 ** (-2), 10 ** (-3), 10 ** (-4)]

        if pi is None:
            pi = 0.5

        self.classes = np.unique(y_test)
        n_classes = len(self.classes)

        if alphas is None:
            alphas = \
                [list(np.zeros(n_classes))] + \
                [gen_alpha(ind, n_classes, 1) for ind in [n_classes - 1]] + \
                [gen_alpha(ind, n_classes, -1) for ind in [n_classes - 1]]

        self.cal_probs = calibrate_probs(
            p_pred=test_probs,
            y_test=y_test,
            alphas=alphas,
            betas=betas,
            jumping_rates=jumping_rates,
            pi=pi,
            return_martingale=return_martingale
        )

    def predict(self, one_hot=True):

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

        if self.cal_probs is None:
            raise Exception("Please calibrate the underlying probabilities first")

        idx = np.argmax(self.cal_probs, axis=-1)
        if one_hot:
            y_pred = np.zeros(self.cal_probs.shape)
            y_pred[np.arange(y_pred.shape[0]), idx] = int(1)
        else:
            y_pred = np.array([self.classes[i] for i in idx])
        return y_pred

    def predict_proba(self):
        """ Outputs calibrated probabilities.

            Returns
            ----------
            p_prime: {array-like}, shape (n_samples,n_classses)
                Protected calibrated probabilities
        """
        if self.cal_probs is None:
            raise Exception("Please calibrate the underlying probabilities first")
        p_prime = np.asarray(self.cal_probs)
        return p_prime
