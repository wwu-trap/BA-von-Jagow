import pandas as pd
import numpy as np
from photonai.base import Hyperpipe, PipelineElement, Switch, Preprocessing
from sklearn.model_selection import StratifiedKFold

# Load data
df = pd.read_excel("C:/Users/paula/Desktop/PyCharm/features_targets_wholebrain_four_classes_final_24_01_21.xlsx")

# Location of x=features, y=targets in Excel file
X = np.asarray(df.iloc[:, 2:])
y = np.asarray(df.iloc[:, 1])

# Specify how and where results are going to be saved
# Define hyperpipe
hyperpipe = Hyperpipe('Wholebrain_classification_four_classes',
                      project_folder='C:/Users/paula/Desktop/PyCharm/00_Results_Wholebrain_classification_four_classes',
                      optimizer="grid_search",
                      optimizer_params={},
                      metrics=['accuracy'],
                      best_config_metric='accuracy',
                      outer_cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                      inner_cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                      verbosity=1)

# Add transformer elements
preprocessing_pipe = Preprocessing()
hyperpipe += preprocessing_pipe
preprocessing_pipe += PipelineElement("LabelEncoder")

hyperpipe += PipelineElement("SimpleImputer", hyperparameters={},
                             test_disabled=False, missing_values=np.nan, strategy='mean', fill_value=0)
hyperpipe += PipelineElement("RobustScaler", hyperparameters={},
                             test_disabled=False, with_centering=True, with_scaling=True)
hyperpipe += PipelineElement("ImbalancedDataTransformer", hyperparameters={},
                             test_disabled=False, method_name='SMOTE')

# Add transformation - dimension reduction or feature selection element
transformer_switch = Switch('TransformerSwitch')
transformer_switch += PipelineElement("PCA", hyperparameters={"n_components": None}, test_disabled=True)
transformer_switch += PipelineElement("FClassifSelectPercentile", hyperparameters={'percentile': [5, 10, 50]},
                                      test_disabled=False)
hyperpipe += transformer_switch


# Add estimator
estimator_switch = Switch('EstimatorSwitch')
estimator_switch += PipelineElement("SVC",
                                    hyperparameters={"C": [0.00000001, 0.000001, 0.0001, 0.01, 1, 100, 10000, 1000000,
                                                           100000000], 'kernel': ['linear', 'rbf', 'poly']},
                                    gamma='auto', max_iter=1000)
estimator_switch += PipelineElement("RandomForestClassifier", hyperparameters={"min_samples_leaf": [0.01, 0.1, 0.2],
                                    'max_features': ['sqrt', 'log2']}, n_estimators=int(100),
                                    min_samples_split=int(2), criterion="gini", max_depth=None)
hyperpipe += estimator_switch


# Fit hyperpipe
hyperpipe.fit(X, y)
