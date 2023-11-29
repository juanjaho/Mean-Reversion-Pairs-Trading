import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import warnings
import pmdarima as pm


class ArimaStrategy:

    def __init__(self, x_data, y_data):
        # convert the data to numpy array
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # assign the data to the class
        self.x_data = x_data
        self.y_data = y_data
        self.beta = x_data / y_data

        # generate the model and in-sample predict beta
        self.model = self._generate_model()
        self.pred_log_beta, self.pred_log_sigma2 = self._insample_predict_beta()
        self.pred_beta = np.exp(self.pred_log_beta)

    # generate the arima model
    def _generate_model(
        self,
    ):
        """
        Generate/re-generate the ARIMA model using the given data
        """
        # Use AR model to find the hedge ratio
        log_x_data = np.log(self.x_data)
        log_y_data = np.log(self.y_data)

        # log(x)-log(y) = log(beta) + s, where s noise
        log_beta = log_y_data - log_x_data

        # https://alkaline-ml.com/pmdarima/tips_and_tricks.html
        # test if ARIMA is appropriate to estimate beta
        model = pm.auto_arima(
            log_beta,
            start_p=1,
            start_q=1,
            max_p=5,
            max_q=5,
            m=1,
            seasonal=True,
            trace=True,
        )
        # model = pm.auto_arima(log_beta, m=12, seasonal=True, trace=True)

        def define_arima_class(model) -> pm.arima.arima.ARIMA:
            """
            Define ARIMA class based on the model
            """
            return model

        model = define_arima_class(model)
        return model

    def _insample_predict_beta(self):
        """
        Predict beta in-sample
        """

        model = self.model
        # get the p, d, q from the model
        p = model.order[0]
        d = model.order[1]
        q = model.order[2]

        pred_log_beta = model.predict_in_sample()
        # set the first few predictions to be NaN
        to_skip = max(p, d, q)
        pred_log_beta[:to_skip] = np.nan

        # get the variance of the residuals sigma^2
        pred_log_sigma2 = np.ones(len(self.x_data)) * model.params()[-1]

        return pred_log_beta, pred_log_sigma2

    def update(self, x_data_point, y_data_point):
        """
        Update the model with next data point
        """
        # convert the data to numpy array
        x_data_point = np.array(x_data_point)
        y_data_point = np.array(y_data_point)

        # insert the data point to the data
        self.x_data = np.append(self.x_data, x_data_point)
        self.y_data = np.append(self.y_data, y_data_point)
        self.beta = np.append(self.beta, x_data_point / y_data_point)

        # predict beta before updating the model
        pred_log_beta, pred_log_sigma2 = self.predict_beta()
        self.pred_log_beta = np.append(self.pred_log_beta, pred_log_beta)
        self.pred_log_sigma2 = np.append(self.pred_log_sigma2, pred_log_sigma2)

        # update the model after prediction to prevent forward-looking bias
        self.model.update(np.log(y_data_point / x_data_point))

    def predict_beta(self):
        """
        Predict y based on the x data

        Returns:
        pred_log_beta: predicted log beta
        pred_log_sigma2: predicted variance of log beta
        """
        pred_log_beta = self.model.predict(n_periods=1)[0]
        pred_log_sigma2 = self.model.params()[-1]

        return pred_log_beta, pred_log_sigma2

    def make_decision(
        self,
        x_data_point,
        y_data_point,
        z_score_entry_threshold=0,
        z_score_exit_threshold=0
    ):
        """
        Get the decision based on the current data
        x_data_point: the current x data point
        y_data_point: the current y data point
        z_score_entry_threshold: the z-score threshold to enter the position
        z_score_exit_threshold: the z-score threshold to exit the position
        carry_position: whether to carry the position from the previous day if there is no entry/exit signal on current day

        Returns:
        decision: 1 for long, -1 for short, 0 for close position None for no position
        """

        # get the predicted beta
        pred_log_beta, pred_log_sigma2 = self.predict_beta()
        pred_beta = np.exp(pred_log_beta)

        # get the spread
        log_spread = np.log(y_data_point / x_data_point) - pred_log_beta

        # get the z-score
        z_score = log_spread / np.sqrt(pred_log_sigma2)

        # get the decision
        long_decision = (None, None)
        short_decision = (None, None)
        # -1 for short, 1 for long, 0 for close position, None for no/hold position
        # enter the position based on entry threshold
        if z_score > z_score_entry_threshold:
            # if z-score is greater than the threshold, short the spread
            # long the x, short the y
            short_decision = (1, -1)
        elif z_score < -z_score_entry_threshold:
            # if z-score is less than the -threshold, long the spread
            # short the x, long the y
            long_decision = (-1, 1)

        # exit the position based on exit threshold
        if z_score <= z_score_exit_threshold:
            # close the short position
            short_decision = (0, 0)
        if z_score >= -z_score_exit_threshold:
            # close the long position
            long_decision = (0, 0)

        return pred_beta, long_decision, short_decision
