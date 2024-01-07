import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

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
        history_forecast_ratio: int = None,
        lags_forecast_ratio: int = None,
        n_estimators: int = 100,
        boosting_type: Optional[str] = "gbdt",
        num_leaves: Optional[int] = 31,
        max_depth: Optional[int] = -1,
        learning_rate: Optional[float] = 0.1,
        lags: Union[int, List[int]] = 20,
        use_exogenous: bool = True,
        verbose: int = 1,
        random_state: int = 0,
    ):
        """Construct a new LGBM Forecaster

        Args:

            data_schema (ForecastingSchema): Schema of the data used for training.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            lags_forecast_ratio (int):
                Sets the lags parameters depending on the forecast horizon.
                lags = forecast horizon * lags_forecast_ratio
                This parameters overides lags parameters.


            n_estimators (int): Number of boosted trees to fit.

            boosting_type (Optional[str]):
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.

            num_leaves (Optional[int]): Maximum tree leaves for base learners.

            max_depth (Optional[int]): Maximum tree depth for base learners, <=0 means no limit.

            learning_rate (Optional[float]): Boosting learning rate.
                You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback.
                Note, that this will ignore the learning_rate argument in training.

            lags (Union[int, List[int]]): Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                - int: include lags from 1 to lags (included).
                - list, 1d numpy ndarray or range: include only lags present in lags, all elements must be int.

            use_exogenous (bool):
                If true, uses covariates in training.

            verbose (int):
                controls the level of LightGBM's verbosity

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.random_state = random_state
        self.use_exogenous = use_exogenous
        self.verbose = verbose
        self.lags = lags
        self._is_trained = False
        self.models = {}
        self.data_schema = data_schema
        self.end_index = {}
        self.history_length = None

        has_covariates = len(
            data_schema.future_covariates + data_schema.static_covariates
        ) > 0 or data_schema.time_col_dtype in ["DATE", "DATETIME"]
        self.use_exogenous = use_exogenous and has_covariates

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )
        if lags_forecast_ratio:
            lags = self.data_schema.forecast_length * lags_forecast_ratio
            self.lags = lags

        self.base_model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            boosting_type=self.boosting_type,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        transformer_exog = MinMaxScaler() if has_covariates else None

        self.model = ForecasterAutoregMultiSeries(
            regressor=self.base_model,
            lags=self.lags,
            transformer_series=MinMaxScaler(),
            transformer_exog=transformer_exog,
        )

    def _add_future_covariates_from_date(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        is_training: bool = True,
    ) -> pd.DataFrame:
        """
        Creates future covariates from the data column.

        Args:
            history (pd.DataFrame):
                The data to create the covariates on.

            data_schema (ForecastingSchema):
                Schema of the data.

            is_training (bool):
                Set to true if the process is to be done on the training data.

        Returns (pd.DataFrame): The processed dataframe
        """

        future_covariates_names = data_schema.future_covariates
        if data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            date_col = pd.to_datetime(history[data_schema.time_col])
            year_col = date_col.dt.year
            month_col = date_col.dt.month
            year_col_name = f"{data_schema.time_col}_year"
            month_col_name = f"{data_schema.time_col}_month"
            history[year_col_name] = year_col
            history[month_col_name] = month_col
            if is_training:
                future_covariates_names += [year_col_name, month_col_name]

        return history

    def crop_data(self, all_series: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Reduces the data based on the history_length attribute.

        Args:
            all_series (List[pd.DataFrame]): List of the original data to be reduced.

        Returns (List[pd.DataFrame]): List of the reduced data.

        """

        new_length = []
        for series in all_series:
            series = series.iloc[-self.history_length :]
            new_length.append(series.copy())
        all_series = new_length
        return all_series

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate Adaboost model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
        """
        np.random.seed(self.random_state)
        data_schema = self.data_schema
        history = self._add_future_covariates_from_date(
            history=history, data_schema=data_schema, is_training=True
        )
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col).reset_index()
            for id_ in all_ids
        ]

        if self.history_length:
            all_series = self.crop_data(all_series)

        targets = [series[data_schema.target] for series in all_series]
        target_series = pd.DataFrame({f"id_{k}": v for k, v in zip(all_ids, targets)})

        exog = None

        if self.use_exogenous:
            covariates_names = (
                data_schema.future_covariates + data_schema.static_covariates
            )
            exog = [series[covariates_names] for series in all_series]
            exog = pd.concat(exog, axis=1)
            exog.columns = [str(i) for i in range(exog.shape[1])]
            self.train_end_index = all_series[0].index.values[-1]

        self.model.fit(series=target_series, exog=exog)

        self.all_ids = all_ids
        self._is_trained = True

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

        test_data = self._add_future_covariates_from_date(
            history=test_data, data_schema=self.data_schema, is_training=False
        )

        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_)
            .drop(columns=self.data_schema.id_col)
            .reset_index()
            for id_ in self.all_ids
        ]
        exog = None
        if self.use_exogenous:
            covariates_names = (
                self.data_schema.future_covariates + self.data_schema.static_covariates
            )
            exog = [series[covariates_names] for series in all_series]
            exog = pd.concat(exog, axis=1)
            exog.columns = [str(i) for i in range(exog.shape[1])]
            start = self.train_end_index + 1
            exog.index = pd.RangeIndex(
                start=start,
                stop=start + self.data_schema.forecast_length,
            )

        forecast = self.model.predict(steps=self.data_schema.forecast_length, exog=exog)
        forecast.columns = [c.split("id_")[1] for c in forecast.columns]
        predictions = []
        for column in forecast.columns:
            predictions += forecast[column].values.tolist()

        test_data[prediction_col_name] = predictions

        return test_data

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

    model = Forecaster(data_schema=data_schema, **hyperparameters)
    model.fit(history=history)
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
