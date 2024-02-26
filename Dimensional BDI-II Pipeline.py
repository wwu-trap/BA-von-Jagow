import pandas as pd
import numpy as np
from photonai.base import Hyperpipe, PipelineElement, Switch, Preprocessing
from sklearn.model_selection import KFold

# Load data
df = pd.read_excel("C:/Users/paula/Desktop/PyCharm/features_targets_wholebrain_BDI_final_24_01_21.xlsx")

# Location of x=features, y=targets in Excel file
X = np.asarray(df.iloc[:, 2:])
y = np.asarray(df.iloc[:, 1])

# Specify how and where results are going to be saved
# Define hyperpipe
hyperpipe = Hyperpipe('00_Results_Wholebrain_regression_BDI',
                      project_folder='C:/Users/paula/Desktop/PyCharm/00_Results_Wholebrain_regression_BDI',
                      optimizer="grid_search",
                      optimizer_params={},
                      metrics=['mean_squared_error', 'mean_absolute_error', 'explained_variance', 'pearson_correlation',
                               'r2'],
                      best_config_metric='mean_squared_error',
                      outer_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                      inner_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                      verbosity=1)

# Add transformer elements
preprocessing_pipe = Preprocessing()
hyperpipe += preprocessing_pipe
preprocessing_pipe += PipelineElement("LabelEncoder")

hyperpipe += PipelineElement("SimpleImputer", hyperparameters={},
                             test_disabled=False, missing_values=np.nan, strategy='mean', fill_value=0)
hyperpipe += PipelineElement("RobustScaler", hyperparameters={},
                             test_disabled=False, with_centering=True, with_scaling=True)
# Add transformation - dimension reduction or feature selection element
transformer_switch = Switch('TransformerSwitch')
transformer_switch += PipelineElement("PCA", hyperparameters={"n_components": None}, test_disabled=True)
transformer_switch += PipelineElement("FRegressionSelectPercentile", hyperparameters={'percentile': [5, 10, 50]},
                                      test_disabled=False)
hyperpipe += transformer_switch


# Add estimator
estimator_switch = Switch('EstimatorSwitch')
estimator_switch += PipelineElement("SVR",
                                    hyperparameters={"C": [0.00000001, 0.000001, 0.0001, 0.01, 1, 100, 10000, 1000000,
                                                           100000000], 'kernel': ['linear', 'rbf', 'poly']},
                                    gamma='auto',
                                    max_iter=1000)
estimator_switch += PipelineElement("RandomForestRegressor", hyperparameters={"min_samples_leaf": [0.01, 0.1, 0.2],
                                    'max_features': ['sqrt', 'log2'], "criterion": ["squared_error", "absolute_error"]},
                                    n_estimators=int(100),
                                    min_samples_split=int(2), max_depth=None)
hyperpipe += estimator_switch


# Fit hyperpipe
hyperpipe.fit(X, y)
