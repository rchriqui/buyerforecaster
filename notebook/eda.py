# Function to evaluate: accuracy, precision, recall, f1-score
# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for the yeo-johnson transformation
import scipy.stats as stats

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


data=pd.read_csv('/Users/robin/Desktop/my_projects/buyerforecaster/notebook/data/online_shoppers_intention.csv')

data.select_dtypes(include=['int64', 'float64']).columns
#boolean to numeric
data['Revenue']=data['Revenue'].astype(int)
data.columns
data['BounceRates'].value_counts()





cat_vars = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
data[cat_vars].astype(object)

num_vars = [
    var for var in data.columns if var not in cat_vars and var != 'Revenue'
]

discrete_vars = [var for var in num_vars if len(
    data[var].unique()) < 15]


cont_vars = [
    var for var in num_vars if var not in discrete_vars]


CATEGORICAL_VARS=cat_vars
NUMERICAL_VARS=num_vars

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Revenue'], axis=1), # predictive variables
    data['Revenue'], # target
    test_size=0.2, # portion of dataset to allocate to test set
    random_state=0, # we are setting the seed here
)

""" X_train[CATEGORICAL_VARS].columns
X_test[CATEGORICAL_VARS].columns
 """


# Create an instance of the encoder
encoder = OneHotEncoder()

# Fit the encoder on the training data
encoder.fit(X_train[CATEGORICAL_VARS])

# Get the feature names from the training data
feature_names = encoder.get_feature_names_out()

# Transform the training data
encoded_training_data = encoder.transform(X_train[CATEGORICAL_VARS]).toarray()

# Transform the testing data
encoded_testing_data = encoder.transform(X_test).toarray()


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv('/mnt/data/online_shoppers_intention.csv')

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), data.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(), ['Month', 'VisitorType'])
    ])

# Apply transformations to the data
X = data.drop('Revenue', axis=1)
y = data['Revenue'].astype(int)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor on the training data and transform both the training and test data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Instantiate the model
rf_clf = RandomForestClassifier(random_state=42)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, confusion_mat, classification_rep




X_train.shape, X_test.shape

X_train[CATEGORICAL_VARS]
