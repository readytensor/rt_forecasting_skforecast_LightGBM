import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the LGBM Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "LGBM Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        n_estimators: int = 100,
        boosting_type: Optional[str] = "gbdt",
        num_leaves: Optional[int] = 31,
        learning_rate: Optional[float] = 0.1,
        lags: Union[int, List[int]] = 7,
        random_state: int = 0,
    ):
        """Construct a new LGBM Forecaster

        Args:

            data_schema (ForecastingSchema): Schema of the data used for training.

            n_estimators (int): Number of boosted trees to fit.

            boosting_type (Optional[str]):
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.

            num_leaves (Optional[int]): Maximum tree leaves for base learners.

            learning_rate (Optional[float]): Boosting learning rate.
                You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback.
                Note, that this will ignore the learning_rate argument in training.

            lags (Union[int, List[int]]): Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                - int: include lags from 1 to lags (included).
                - list, 1d numpy ndarray or range: include only lags present in lags, all elements must be int.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.n_estimators = n_estimators
        self.leaarning_rate = learning_rate
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.lags = lags
        self._is_trained = False
        self.models = {}
        self.data_schema = data_schema
        self.end_index = {}

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        history_length: int = None,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate LGBM model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
            history_length (int): The length of the series used for training.
        """
        np.random.seed(self.random_state)
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.models = {}

        for id, series in zip(all_ids, all_series):
            if history_length:
                series = series[-history_length:]
            model = self._fit_on_series(history=series, data_schema=data_schema, id=id)
            self.models[id] = model

        self.all_ids = all_ids
        self._is_trained = True
        self.data_schema = data_schema

    def _fit_on_series(
        self, history: pd.DataFrame, data_schema: ForecastingSchema, id: int
    ):
        """Fit LGBM model to given individual series of data"""
        model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.leaarning_rate,
            num_leaves=self.num_leaves,
            boosting_type=self.boosting_type,
            random_state=self.random_state,
            verbose=-1,
        )
        forecaster = ForecasterAutoreg(regressor=model, lags=self.lags)

        covariates = data_schema.future_covariates

        history.index = pd.RangeIndex(start=0, stop=len(history))

        self.end_index[id] = len(history)
        exog = None
        if covariates:
            exog = history[covariates]

        forecaster.fit(y=history[data_schema.target], exog=exog)

        return forecaster

    def predict(self, test_data: pd.DataFrame, prediction_col_name: str) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.data_schema.id_col)
            for id_ in self.all_ids
        ]
        # forecast one series at a time
        all_forecasts = []
        for id_, series_df in zip(self.all_ids, all_series):
            forecast = self._predict_on_series(
                key_and_future_df=(id_, series_df), id=id_
            )
            forecast.insert(0, self.data_schema.id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)

        all_forecasts.rename(
            columns={self.data_schema.target: prediction_col_name}, inplace=True
        )
        return all_forecasts

    def _predict_on_series(self, key_and_future_df, id):
        """Make forecast on given individual series of data"""
        key, future_df = key_and_future_df

        start = self.end_index[id]
        future_df.index = pd.RangeIndex(start=start, stop=start + len(future_df))
        exog = None
        covariates = self.data_schema.future_covariates
        if covariates:
            exog = future_df[covariates]

        if self.models.get(key) is not None:
            forecast = self.models[key].predict(
                steps=len(future_df),
                exog=exog,
            )
            future_df[self.data_schema.target] = forecast.values

        else:
            # no model found - key wasnt found in history, so cant forecast for it.
            future_df = None

        return future_df

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    history_length = None
    history_forecast_ratio = hyperparameters.get("history_forecast_ratio")
    if history_forecast_ratio:
        history_length = data_schema.forecast_length * history_forecast_ratio
    if "history_forecast_ratio" in hyperparameters:
        hyperparameters.pop("history_forecast_ratio")

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(history=history, data_schema=data_schema, history_length=history_length)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
