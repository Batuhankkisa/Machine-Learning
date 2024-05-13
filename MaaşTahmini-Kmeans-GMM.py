import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SalaryPredictionModel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.numeric_cols = ['salary_in_usd']
        self.categorical_cols = ['job_title', 'job_category', 'salary_currency', 'employee_residence',
                                 'experience_level', 'employment_type', 'work_setting']

        # Initialize kmeans as an instance variable
        self.kmeans = None

    def remove_unnamed_column(self):
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

    def fill_missing_values(self):
        self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].mean())
        self.df[self.categorical_cols] = self.df[self.categorical_cols].fillna(self.df[self.categorical_cols].mode().iloc[0])

    def apply_kmeans_clustering(self, num_clusters):
        X_numeric = self.df[self.numeric_cols]
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.df['kmeans_cluster'] = self.kmeans.fit_predict(X_numeric)

    def apply_gmm_clustering(self, num_components):
        X_numeric = self.df[self.numeric_cols]
        gmm = GaussianMixture(n_components=num_components, random_state=42)
        self.df['gmm_cluster'] = gmm.fit_predict(X_numeric)

    def train_xgb_model(self, X_train, y_train):
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_ann_model(self, X_train, y_train):
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)  # Only one epoch
        return model

    def evaluate_numeric_model(self, model, X_test, y_test, model_type):
        y_pred = model.predict(X_test)
        rmse = self.calculate_rmse(y_test, y_pred)
        self.visualize_numeric_predictions(y_test, y_pred, model_type)
        return rmse

    def calculate_rmse(self, y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print('Model RMSE:', rmse)
        return rmse

    def visualize_ann_training(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()

    def visualize_numeric_predictions(self, y_true, y_pred, model_type):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, cmap='viridis')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', lw=2)
        plt.title(f'Model: Actual vs Predicted Salary (in USD) - {model_type}')
        plt.xlabel('Actual Salary (in USD)')
        plt.ylabel('Predicted Salary (in USD)')
        plt.grid(True)
        plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window
        plt.show()

    def visualize_feature_importance(self, model, feature_names):
        plt.figure(figsize=(10, 6))
        importance = model.feature_importances_
        sns.barplot(x=importance, y=feature_names, orient='h')
        plt.title('XGBoost Feature Importance')
        plt.show()

    def visualize_cluster_distributions(self, data, cluster_col, salary_col):
        plt.figure(figsize=(15, 8))
        for cluster in data[cluster_col].unique():
            cluster_data = data[data[cluster_col] == cluster]
            sns.kdeplot(cluster_data[salary_col], label=f'Cluster {cluster}', fill=True, alpha=0.5)

        plt.title(f'Salary Distribution within Clusters ({cluster_col})')
        plt.xlabel('Salary (in USD)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_additional_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        r_squared = r2_score(y_true, y_pred)
        print('Mean Absolute Error:', mae)
        print('R-squared:', r_squared)

    def visualize_kmeans_clusters(self):
        if self.kmeans is None:
            print("Please run apply_kmeans_clustering before visualizing clusters.")
            return

        plt.figure(figsize=(15, 8))
        for cluster in self.df['kmeans_cluster'].unique():
            cluster_data = self.df[self.df['kmeans_cluster'] == cluster]
            plt.scatter(cluster_data['salary_in_usd'], cluster_data['kmeans_cluster'], alpha=0.5, label=f'Cluster {cluster}', cmap='viridis')

        centroids = self.kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], [len(self.df['kmeans_cluster'].unique()) - 1] * len(centroids), marker='X', s=200, color='black', label='Centroids')
        
        sns.kdeplot(self.df['salary_in_usd'], color='gray', fill=True, alpha=0.2, label='Salary Distribution')
        
        plt.title('K-Means Clustering: Salary vs Cluster')
        plt.xlabel('Salary (in USD)')
        plt.ylabel('K-Means Cluster')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_gmm_clusters(self):
        plt.figure(figsize=(15, 8))
        for cluster in self.df['gmm_cluster'].unique():
            cluster_data = self.df[self.df['gmm_cluster'] == cluster]
            plt.scatter(cluster_data['salary_in_usd'], cluster_data['gmm_cluster'], alpha=0.5, label=f'Cluster {cluster}', cmap='viridis')

        sns.kdeplot(self.df['salary_in_usd'], color='gray', fill=True, alpha=0.2, label='Salary Distribution')
        
        plt.title('Gaussian Mixture Model Clustering: Salary vs Cluster')
        plt.xlabel('Salary (in USD)')
        plt.ylabel('GMM Cluster')
        plt.legend()
        plt.grid(True)
        plt.show()

    def split_data_numeric(self):
        X_numeric = self.df[self.numeric_cols]
        y_numeric = self.df['salary_in_usd']
        return train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

    def run(self):
        self.remove_unnamed_column()
        self.fill_missing_values()

        # Apply K-Means clustering
        num_clusters = 3
        self.apply_kmeans_clustering(num_clusters)
        self.visualize_kmeans_clusters()

        # Apply GMM clustering
        num_components = 3
        self.apply_gmm_clustering(num_components)
        self.visualize_gmm_clusters()

        # Numeric Regression with XGBoost
        X_numeric_train, X_numeric_test, y_numeric_train, y_numeric_test = self.split_data_numeric()
        xgboost_model = self.train_xgb_model(X_numeric_train, y_numeric_train)
        self.visualize_feature_importance(xgboost_model, X_numeric_train.columns)
        rmse_xgboost = self.evaluate_numeric_model(xgboost_model, X_numeric_test, y_numeric_test, model_type='XGBoost')
        print(f'XGBoost RMSE: {rmse_xgboost}')

        # Numeric Regression with Artificial Neural Network
        ann_model = self.train_ann_model(X_numeric_train, y_numeric_train)
        self.calculate_additional_metrics(y_numeric_test, ann_model.predict(X_numeric_test))
        rmse_ann = self.evaluate_numeric_model(ann_model, X_numeric_test, y_numeric_test, model_type='ANN')
        print(f'ANN RMSE: {rmse_ann}')

if __name__ == "__main__":
    data_path = 'jobs_in_data.csv'
    salary_model = SalaryPredictionModel(data_path)
    salary_model.run()
