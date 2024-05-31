import pandas as pd
"""
pandas: A powerful data manipulation library.

This library provides data structures like DataFrame for efficient data manipulation and analysis.
For more information, refer to the pandas documentation: https://pandas.pydata.org/pandas-docs/stable/index.html
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
"""
scikit-learn: Simple and efficient tools for data mining and data analysis.

train_test_split: A function to split arrays or matrices into random train and test subsets.
mean_squared_error: A function to compute the mean squared error regression loss.

For more information, refer to the scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
"""
import matplotlib.pyplot as plt
"""
matplotlib: A 2D plotting library.

This library provides functions for creating various types of plots and visualizations.
For more information, refer to the matplotlib documentation: https://matplotlib.org/stable/contents.html
"""
import seaborn as sns
"""
seaborn: Statistical data visualization.

This library provides a high-level interface for drawing attractive and informative statistical graphics.
For more information, refer to the seaborn documentation: https://seaborn.pydata.org/
"""
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
"""
KMeans: K-Means clustering algorithm for unsupervised learning.
GaussianMixture: Gaussian Mixture Model for clustering.

For more information, refer to the scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
"""
from xgboost import XGBRegressor
"""
XGBoost: An optimized distributed gradient boosting library.

XGBRegressor: XGBoost implementation for regression tasks.

For more information, refer to the XGBoost documentation: https://xgboost.readthedocs.io/en/latest/
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
"""
TensorFlow: An open-source machine learning library.

Sequential: A linear stack of layers for building neural network models.
Dense: A fully connected layer for neural network models.

For more information, refer to the TensorFlow documentation: https://www.tensorflow.org/api_docs/python
"""

class SalaryPredictionModel:
    """
    A class for predicting salaries based on job-related data.

    Attributes:
    - data_path (str): The file path to the CSV containing job-related data.
    - df (pandas.DataFrame): The DataFrame storing the loaded data.
    - numeric_cols (list): List of column names containing numeric data.
    - categorical_cols (list): List of column names containing categorical data.

    Methods:
    - remove_unnamed_column(): Removes the 'Unnamed: 0' column if present.
    - fill_missing_values(): Handles missing values in numeric and categorical columns.
    - add_correlation_heatmap(): Generates a heatmap of correlations for numeric columns.
    - visualize_categorical_data(): Creates count plots for each categorical column.
    - find_column_with_lowest_correlation(): Identifies the column with the lowest correlation to the target variable.
    - drop_column_with_lowest_correlation(): Drops the column with the lowest correlation.
    - update_numeric_and_categorical_cols(dropped_column): Updates numeric and categorical column lists after dropping a column.
    - split_data_numeric(): Splits data into numeric features and target variable for regression models.
    - train_xgboost_model(X_train, y_train): Trains an XGBoost regression model.
    - train_ann_model(X_train, y_train): Trains an Artificial Neural Network regression model.
    - evaluate_numeric_model(model, X_test, y_test, model_type): Evaluates and visualizes the performance of a numeric regression model.
    - calculate_rmse(y_true, y_pred): Calculates the Root Mean Squared Error (RMSE) for model evaluation.
    - visualize_numeric_predictions(y_true, y_pred, model_type): Visualizes actual vs predicted numeric values.
    - visualize_data(): Creates subplots to visualize the distribution of each column.
    - apply_kmeans_clustering(num_clusters): Applies K-Means clustering and visualizes results.
    - apply_gmm_clustering(num_components): Applies Gaussian Mixture Model clustering and visualizes results.
    - visualize_clustering_results(cluster_column, title): Visualizes clustering results.
    - plot_salary_distribution(): Creates a histogram to visualize the salary distribution.
    - visualize_cluster_salaries(cluster_column, title): Creates box plots to show salary distributions within clusters.
    - evaluate_cluster_performance(cluster_column): Calculates and prints the average salary for each cluster.
    - run(): Executes the entire workflow, from data preprocessing to model training and evaluation.

    """
    def __init__(self, data_path):
        """
    Initializes a SalaryPredictionModel instance.

    Parameters:
    - data_path (str): The file path to the CSV containing job-related data.

    Attributes:
    - df (pandas.DataFrame): The DataFrame storing the loaded data.
    - numeric_cols (list): List of column names containing numeric data.
    - categorical_cols (list): List of column names containing categorical data.

    """
        self.df = pd.read_csv(data_path)
        self.numeric_cols = ['salary_in_usd']
        self.categorical_cols = ['job_title', 'job_category', 'salary_currency', 'employee_residence',
                                 'experience_level', 'employment_type', 'work_setting']

    def remove_unnamed_column(self):
        """
    Removes the 'Unnamed: 0' column from the DataFrame if it exists.

    This method checks if the 'Unnamed: 0' column is present in the DataFrame columns.
    If found, it drops the column in-place, modifying the DataFrame.
    """
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

    def fill_missing_values(self):
        """
    Fills missing values in the DataFrame.

    This method handles missing values by filling them with appropriate statistics:
    - For numeric columns, missing values are filled with the mean of each respective column.
    - For categorical columns, missing values are filled with the mode (most frequent value) of each respective column.

    The DataFrame is modified in-place.
    """
        # Fill missing values with the mean for numeric columns
        self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].mean())

        # Fill missing values in categorical columns with mode
        self.df[self.categorical_cols] = self.df[self.categorical_cols].fillna(self.df[self.categorical_cols].mode().iloc[0])

    def add_correlation_heatmap(self):
        """
    Generates and displays a correlation heatmap for numeric columns.

    This method calculates the correlation matrix for numeric columns in the DataFrame
    and creates a heatmap to visualize the relationships between variables.

    The heatmap includes annotations with correlation values, and it uses the 'coolwarm'
    colormap for better visibility. The figure is displayed with a title indicating that
    only numeric columns are considered.
    """
        # Exclude non-numeric columns
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.df[numeric_columns].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap (Numeric Columns Only)')
        plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
        plt.show()

    def visualize_categorical_data(self):
        """
    Creates count plots to visualize the distribution of categorical columns.

    This method iterates through the specified categorical columns and generates count plots
    to show the distribution of each category. Each plot includes a title with the column name
    and is displayed in a maximized window for better visibility.

    """
        for column in self.categorical_cols:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=column, data=self.df)
            plt.title(f'{column} Distribution')
            plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
            plt.show()

    def find_column_with_lowest_correlation(self):
        """
    Identifies the column with the lowest correlation to the target variable.

    This method calculates the correlation matrix for numeric columns in the DataFrame
    and excludes the target variable column ('salary_in_usd') from consideration.
    It then identifies the column with the minimum correlation to the target variable.

    The result is printed, and the column name is returned.
    """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.df[numeric_columns].corr()

        # Exclude the target variable column ('salary_in_usd') from consideration
        target_column = 'salary_in_usd'
        numeric_columns = numeric_columns[numeric_columns != target_column]

        min_corr_col = correlation_matrix.loc[numeric_columns, target_column].idxmin()
        print(f"Column with the lowest correlation to the target variable: {min_corr_col}")
        return min_corr_col

    def drop_column_with_lowest_correlation(self):
        """
    Drops the column with the lowest correlation to the target variable.

    This method utilizes the find_column_with_lowest_correlation method to identify
    the column with the lowest correlation to the target variable ('salary_in_usd').
    It then drops that column from the DataFrame in-place.

    The method prints messages indicating the intention and completion of the operation.
    """
        min_corr_col = self.find_column_with_lowest_correlation()
        print(f"About to drop column with closest correlation to 0: {min_corr_col}")
        self.df.drop(columns=min_corr_col, inplace=True)
        print(f"Dropped column with closest correlation to 0: {min_corr_col}")

    def update_numeric_and_categorical_cols(self, dropped_column):
        """
    Updates the lists of numeric and categorical columns after dropping a column.

    This method is called after dropping a column with the lowest correlation to the target variable.
    It updates the 'numeric_cols' list by removing the dropped column, and appends the dropped column
    to the 'categorical_cols' list.

    Parameters:
    - dropped_column (str): The name of the column that was dropped.
    """
        self.numeric_cols = [col for col in self.numeric_cols if col != dropped_column]
        self.categorical_cols.append(dropped_column)

    def split_data_numeric(self):
        """
    Splits the data into numeric features and the target variable for regression models.

    This method extracts numeric features (X_numeric) and the target variable ('salary_in_usd')
    from the DataFrame and performs a train-test split. The split data is returned as a tuple
    containing X_train, X_test, y_train, and y_test.

    Returns:
    - X_train (pandas.DataFrame): Training set of numeric features.
    - X_test (pandas.DataFrame): Testing set of numeric features.
    - y_train (pandas.Series): Training set of the target variable.
    - y_test (pandas.Series): Testing set of the target variable.
    """
        X_numeric = self.df[self.numeric_cols]
        y_numeric = self.df['salary_in_usd']
        return train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

    def train_xgboost_model(self, X_train, y_train):
        """
    Trains an XGBoost regression model.

    This method initializes an XGBRegressor with the specified parameters, fits the model
    on the training data, and returns the trained model.

    Parameters:
    - X_train (pandas.DataFrame): Training set of numeric features.
    - y_train (pandas.Series): Training set of the target variable.

    Returns:
    - model (XGBRegressor): Trained XGBoost regression model.
    """
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_ann_model(self, X_train, y_train):
        """
    Trains an Artificial Neural Network (ANN) regression model.

    This method constructs a Sequential model with three densely connected layers:
    - Input layer with 64 neurons and ReLU activation.
    - Hidden layer with 32 neurons and ReLU activation.
    - Output layer with 1 neuron and linear activation.

    The model is compiled using the 'adam' optimizer and 'mean_squared_error' loss function.
    It is then trained on the provided training data for 50 epochs with a batch size of 32.

    Parameters:
    - X_train (pandas.DataFrame): Training set of numeric features.
    - y_train (pandas.Series): Training set of the target variable.

    Returns:
    - model (Sequential): Trained ANN regression model.
    """
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        return model

    def evaluate_numeric_model(self, model, X_test, y_test, model_type):
        """
    Evaluates and visualizes the performance of a numeric regression model.

    This method takes a trained regression model, predicts the target variable on the test set,
    calculates the Root Mean Squared Error (RMSE), and visualizes the actual vs predicted values.

    Parameters:
    - model: Trained regression model (e.g., XGBoost, ANN).
    - X_test (pandas.DataFrame): Test set of numeric features.
    - y_test (pandas.Series): Test set of the target variable.
    - model_type (str): Type of the regression model (e.g., 'XGBoost', 'ANN').

    Returns:
    - rmse (float): Root Mean Squared Error.
    """
        y_pred = model.predict(X_test).flatten()  # Flatten the predictions
        rmse = self.calculate_rmse(y_test, y_pred)
        self.visualize_numeric_predictions(y_test, y_pred, model_type)

        # Print actual vs predicted values for the first 20 rows
        print(f"\nActual vs Predicted values for the first 20 rows ({model_type}):")
        print(pd.DataFrame({'Actual': y_test[:20], 'Predicted': y_pred[:20]}))

        # Print RMSE
        print(f"RMSE ({model_type}): {rmse}")

        return rmse


    def calculate_rmse(self, y_true, y_pred):
        """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    This method computes the RMSE using the mean_squared_error function from scikit-learn.

    Parameters:
    - y_true (array-like): True values of the target variable.
    - y_pred (array-like): Predicted values of the target variable.

    Returns:
    - rmse (float): Root Mean Squared Error.
    """
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print('Model RMSE:', rmse)
        return rmse

    def visualize_numeric_predictions(self, y_true, y_pred, model_type):
        """
    Visualizes actual vs predicted numeric values using a scatter plot.

    This method creates a scatter plot comparing the actual vs predicted values of the target variable.
    It also includes a red dashed line representing a perfect match between actual and predicted values.

    Parameters:
    - y_true (array-like): True values of the target variable.
    - y_pred (array-like): Predicted values of the target variable.
    - model_type (str): Type of the regression model (e.g., 'XGBoost', 'ANN').
    """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', lw=2)
        plt.title(f'Model: Actual vs Predicted Salary (in USD) - {model_type}')
        plt.xlabel('Actual Salary (in USD)')
        plt.ylabel('Predicted Salary (in USD)')
        plt.grid(True)
        plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
        plt.show()

    def visualize_data(self):
        """
    Visualizes the distribution of categorical and discrete features in the dataset.

    This method creates a grid of count plots for each column in the DataFrame, with each subplot
    representing the distribution of values. It uses different visualization styles based on the
    number of unique values in each column.
    """
        r = 4  # Subplot grid row count
        c = 6  # Subplot grid column count
        it = 1  # Iterator for subplot position

        plt.figure(figsize=(18, 12))  # Set the overall figure size

        for i in self.df.columns:
            plt.subplot(r, c, it)
            if self.df[i].nunique() > 6:
                sns.countplot(x=self.df[i])
                plt.grid()
            else:
                sns.countplot(x=self.df[i])
            plt.xlabel(i)
            plt.ylabel('Count')
            it += 1

        plt.tight_layout()
        plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
        plt.show()

    def apply_kmeans_clustering(self, num_clusters):
        """
    Applies K-Means clustering to numeric features and visualizes the results.

    This method uses the KMeans algorithm to cluster the numeric features of the DataFrame
    into the specified number of clusters. The cluster labels are added to the DataFrame
    under the column name 'kmeans_cluster'. The method then visualizes the clustering results.

    Parameters:
    - num_clusters (int): The number of clusters to create.
    """
        X_numeric = self.df[self.numeric_cols]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.df['kmeans_cluster'] = kmeans.fit_predict(X_numeric)

        # Visualize K-Means Clustering Results
        self.visualize_clustering_results('kmeans_cluster', 'K-Means Clustering')

    def apply_gmm_clustering(self, num_components):
        """
    Applies Gaussian Mixture Model (GMM) clustering to numeric features and visualizes the results.

    This method uses the GaussianMixture algorithm to fit a model to the numeric features of the DataFrame
    assuming a specified number of components (clusters). The cluster labels are added to the DataFrame
    under the column name 'gmm_cluster'. The method then visualizes the clustering results.

    Parameters:
    - num_components (int): The number of components (clusters) to assume.
    """
        X_numeric = self.df[self.numeric_cols]
        gmm = GaussianMixture(n_components=num_components, random_state=42)
        self.df['gmm_cluster'] = gmm.fit_predict(X_numeric)

        # Visualize GMM Clustering Results
        self.visualize_clustering_results('gmm_cluster', 'GMM Clustering')

    def visualize_clustering_results(self, cluster_column, title):
        """
    Visualizes the clustering results using a scatter plot.

    This method creates a scatter plot with the target variable 'salary_in_usd' on the x-axis and
    the cluster assignments from the specified column on the y-axis. Each cluster is represented by
    a different color on the plot.

    Parameters:
    - cluster_column (str): The column containing cluster assignments.
    - title (str): Title for the visualization.
    """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df['salary_in_usd'], y=self.df[cluster_column], hue=self.df[cluster_column], palette='viridis')
        plt.title(title)
        plt.xlabel('Salary (in USD)')
        plt.ylabel('Cluster')
        plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
        plt.show()

    def plot_salary_distribution(self):
        """
    Plots the distribution of salaries in the dataset.

    This method creates a histogram with a kernel density estimate (KDE) of the 'salary_in_usd' column,
    providing insights into the distribution of salaries.
    """
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df['salary_in_usd'], kde=True, color='blue', bins=30)
        plt.title('Salary Distribution')
        plt.xlabel('Salary (in USD)')
        plt.ylabel('Frequency')
        plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
        plt.show()

    def visualize_cluster_salaries(self, cluster_column, title):
        """
    Visualizes the distribution of salaries within each cluster using a box plot.

    This method creates a box plot with the target variable 'salary_in_usd' on the y-axis
    and the cluster assignments from the specified column on the x-axis. Each box represents
    the salary distribution within a cluster.

    Parameters:
    - cluster_column (str): The column containing cluster assignments.
    - title (str): Title for the visualization.
    """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=self.df[cluster_column], y=self.df['salary_in_usd'])
        plt.title(f'{title} - Salary Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Salary (in USD)')
        plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
        plt.show()

    def evaluate_cluster_performance(self, cluster_column):
        """
    Evaluates the average salary within each cluster and prints the results.

    This method calculates the average salary for each unique value in the specified
    cluster column and prints the results.

    Parameters:
    - cluster_column (str): The column containing cluster assignments.
    """
        avg_salary_by_cluster = self.df.groupby(cluster_column)['salary_in_usd'].mean()
        print(f'\nAverage Salary by {cluster_column}:\n{avg_salary_by_cluster}')

    def run(self):
        """
    Executes the complete pipeline for salary prediction model analysis.
    This method performs the following steps:
    1. Visualizes the overall distribution of categorical and discrete features in the dataset.
    2. Visualizes the distribution of each categorical feature.
    3. Adds a correlation heatmap for numeric columns.
    4. Removes any unnamed column if present.
    5. Fills missing values in numeric columns with the mean and in categorical columns with the mode.
    6. Drops the column with the lowest correlation to the target variable.
    7. Updates numeric and categorical columns after dropping a column.
    8. Applies K-Means clustering with a specified number of clusters.
    9. Applies GMM clustering with a specified number of components.
    10. Visualizes the distribution of salaries using a histogram.
    11. Visualizes the distribution of salaries within each cluster for K-Means clustering.
    12. Evaluates the average salary within each cluster for K-Means clustering.
    13. Visualizes the distribution of salaries within each cluster for GMM clustering.
    14. Evaluates the average salary within each cluster for GMM clustering.
    15. Performs numeric regression with XGBoost and evaluates the model.
    16. Performs numeric regression with an Artificial Neural Network (ANN) and evaluates the model.
    """
        self.visualize_data()
        self.visualize_categorical_data()
        self.add_correlation_heatmap()
        self.remove_unnamed_column()
        self.fill_missing_values()
        self.drop_column_with_lowest_correlation()

        # Update numeric and categorical columns after dropping a column
        dropped_column = self.find_column_with_lowest_correlation()
        self.update_numeric_and_categorical_cols(dropped_column)

        # Apply K-Means clustering
        num_clusters = 3
        self.apply_kmeans_clustering(num_clusters)

        # Apply GMM clustering
        num_components = 3
        self.apply_gmm_clustering(num_components)

        # Visualize Salary Distribution
        self.plot_salary_distribution()

        # Visualize Cluster Salaries for K-Means Clustering
        self.visualize_cluster_salaries('kmeans_cluster', 'K-Means Clustering')

        # Evaluate K-Means Clustering Performance
        self.evaluate_cluster_performance('kmeans_cluster')

        # Visualize Cluster Salaries for GMM Clustering
        self.visualize_cluster_salaries('gmm_cluster', 'GMM Clustering')

        # Evaluate GMM Clustering Performance
        self.evaluate_cluster_performance('gmm_cluster')

        # Numeric Regression with XGBoost
        X_numeric_train, X_numeric_test, y_numeric_train, y_numeric_test = self.split_data_numeric()
        xgboost_model = self.train_xgboost_model(X_numeric_train, y_numeric_train)
        self.evaluate_numeric_model(xgboost_model, X_numeric_test, y_numeric_test, model_type='XGBoost')

        # Numeric Regression with Artificial Neural Network
        ann_model = self.train_ann_model(X_numeric_train, y_numeric_train)
        self.evaluate_numeric_model(ann_model, X_numeric_test, y_numeric_test, model_type='ANN')

if __name__ == "__main__":
    data_path = 'jobs_in_data.csv'
    salary_model = SalaryPredictionModel(data_path)
    salary_model.run()
