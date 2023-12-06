import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def perform_simple_ols(data, target, feature):
    '''
    Perform a simple Ordinary Least Squares (OLS) regression.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the data
    target (str): The name of the target variable
    feature (str): The name of the feature variable

    Returns:
    model: The OLS regression model
    '''
    # Preparing the data
    X = data[feature].values.reshape(-1, 1)
    y = data[target].values

    # Adding a constant to the model (intercept) and fitting the model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def perform_multiple_ols(data, target, features):
    '''
    Perform a multiple Ordinary Least Squares (OLS) regression.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the data
    target (str): The name of the target variable
    features (list): The list of feature variable names

    Returns:
    model: The OLS regression model
    '''
    # Preparing the data
    X = data[features]
    y = data[target].values

    # Adding a constant to the model (intercept) and fitting the model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def plot_actual_vs_predicted(model, data, target, feature=None, title='Actual vs Predicted'):
    '''
    Generate and save a graph of actual vs predicted values from the regression model.

    Parameters:
    model: The regression model
    data (DataFrame): The pandas DataFrame containing the data
    target (str): The name of the target variable
    feature (str, optional): The name of the feature variable for simple regression
    title (str): The title of the graph

    Returns:
    None
    '''
    # Actual vs. Predicted Values
    if isinstance(feature, list) or isinstance(feature, tuple):
            if len(feature) > 1:
                title = 'Actual vs Predicted (MLR)'
                c = 'blue'
            else:
                title = 'Actual vs Predicted (Simple OLS)'
                c = 'violet'
    else:
        title = 'Actual vs Predicted (Simple OLS)'
        c = 'violet'
    plt.figure(figsize=(12, 6))
    plt.scatter(data[target], model.predict(), color=c, alpha=0.4)
    plt.title(title)
    plt.xlabel('Actual ' + target)
    plt.ylabel('Predicted ' + target)
    plt.savefig(f'outputs/{title.replace(" ", "_")}.png')
    plt.show()