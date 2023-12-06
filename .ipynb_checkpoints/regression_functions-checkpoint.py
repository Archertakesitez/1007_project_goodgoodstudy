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

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fitting the model
    model = sm.OLS(y, X).fit()
    return model

def generate_loss_graph(model, title):
    '''
    Generate and save a graph of the model's loss.

    Parameters:
    model: The regression model
    title (str): The title of the graph

    Returns:
    None
    '''
    # Calculate predictions
    predictions = model.predict()
    
    # Calculate loss
    loss = mean_squared_error(model.endog, predictions)

    # Plotting the loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Loss')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{title}_loss.png')
    plt.close()
