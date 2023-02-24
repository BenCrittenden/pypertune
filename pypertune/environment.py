import numpy as np

from collections import OrderedDict
from sklearn.model_selection import cross_validate


class Environment:

    """Hyperparameter tuning Environment class

    Parameters
    ----------
    x_data: numpy array
        Numpy array corresponding to the training data for the model.

    y_data: numpy array
        Numpy array corresponding to the training targets for the model.

    model: sklearn model
        The machine learning model for which the hyperparameters will be
        tuned.

    parameters: dictionary
        Dictionary of hyperparameters to be tuned. The keys of this
        dictionary must correspond to valid hyperparameters of the supplied
        model. Each value is another dictionary with keys 'min',
        specifying the minimum value for the hyperparameter, 'max',
        specifying the maximum possible value for the hyperparameter, and
        'type', with valid values 'int' and 'float', specifying whether or
        not the hyperparameter takes discrete or continuous values.

    parameters_fixed: dictionary
        Dictionary of additional model hyperparameters that will be fixed
        instead of being tuned. The keys of this dictionary must correspond
        to valid hyperparameters of the supplied model. Each hyperparameter
        will be set to the corresponding value in this dictionary.

    """

    def __init__(
        self,
        x_data,
        y_data,
        model,
        parameters,
        parameters_fixed={},
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.model = model
        self.parameters = parameters
        self.parameters_fixed = parameters_fixed

        self.param_mins = OrderedDict(
            {param: options["min"] for param, options in parameters.items()}
        )

        self.param_maxs = OrderedDict(
            {param: options["max"] for param, options in parameters.items()}
        )

        self.reset()

    def reset(self):
        """Resets the environment by setting its hyperparameters to random
        valid values and updating the state and previous performance

        Returns
        -------

        self.state: numpy array
            state vector corresponding to the current parameters.
        """
        self.params = OrderedDict(
            {
                param: np.random.randint(
                    self.param_mins[param], self.param_maxs[param]
                )
                if self.parameters[param]["type"] == "int"
                else np.random.rand()
                * (self.param_maxs[param] - self.param_mins[param])
                + self.param_mins[param]
                for param in self.param_mins.keys()
            }
        )

    def test_model(self, params):
        """Tests the performance of a model with the specified set of
        hyperparameters.

        Parameters
        ----------

        params: np.array
            Array of hyperparameters.

        Returns
        -------

        reward: float
            Reward calculated via the reward function based on the
            performance of the model with the specified hyperparameters.
        """

        clf = self.model(**params)
        clf.fit(self.x_data, self.y_data)
        perf = cross_validate(
            clf, self.x_data, self.y_data, cv=3, return_train_score=False
        )["test_score"].mean()
        return perf

    def gen_classifier(self, optimal_param_dict={}):
        """Generates a model with a given set of hyperparameters

        Parameters
        ----------

        optimal_param_dict: dictionary of parameters
            If using this function to generate the optimal classifier,
            then this dictionary contains the optimal hyperparameters

        Returns
        -------

        model: sklearn model

        """
        classifier_gen_param_dict = self.parameters_fixed.copy()

        if bool(optimal_param_dict):
            classifier_gen_param_dict.update(optimal_param_dict)
        else:
            classifier_gen_param_dict.update(self.params)

        return self.model(**classifier_gen_param_dict)
