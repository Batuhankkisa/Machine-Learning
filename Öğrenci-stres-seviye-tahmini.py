import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_palette("pastel")

class StressLevelModel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.linear_cols = ['anxiety_level', 'mental_health_history', 'depression',
                            'headache', 'blood_pressure', 'breathing_problem', 'noise_level', 'living_conditions',
                            'safety', 'basic_needs', 'academic_performance', 'study_load',
                            'teacher_student_relationship', 'future_career_concerns', 'social_support',
                            'peer_pressure', 'extracurricular_activities', 'bullying']
        self.logistic_cols = ['stress_level', 'sleep_quality']

    def preprocess_data(self):
        self.df.drop(columns=["Unnamed: 0"], errors='ignore', inplace=True)
        self.df.fillna(self.df.mean(), inplace=True)

        min_corr_col = self.find_column_with_min_correlation()
        self.df.drop(columns=min_corr_col, inplace=True)
        
        print(f"Dropped column with closest correlation to 0: {min_corr_col}")
        self.linear_cols = [col for col in self.linear_cols if col != min_corr_col]

    def find_column_with_min_correlation(self):
        return self.df.corr().iloc[:-1, -1].idxmin()

    def split_data(self, X, y, model_type='linear'):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, model_type='linear'):
        model = LinearRegression() if model_type == 'linear' else LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, model_type='linear'):
        y_pred = model.predict(X_test)
        used_columns = []

        if model_type == 'linear':
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            print(f'Linear Regression RMSE: {rmse:.4f}')
            print('\nLinear Regression Used Columns:')
            used_columns = list(X_test.columns)
            for col in used_columns:
                print(f'  - {col}')

            self.plot_scatter_actual_vs_predicted(y_test, y_pred, 'Linear Regression: Actual vs Predicted')

        elif model_type == 'logistic':
            accuracy, conf_matrix, classification_rep = self.get_classification_metrics(y_test, y_pred)
            print('Logistic Regression Classification Model:')
            print(f'  Accuracy: {accuracy:.4f}')
            print('  Confusion Matrix:')
            print(conf_matrix)
            print('  Classification Report:')
            print(classification_rep)

            print('\nLogistic Regression Used Columns:')
            used_columns = list(X_test.columns)
            for col in used_columns:
                print(f'  - {col}')

            self.plot_roc_curve(model, X_test, y_test)

        else:
            raise ValueError("Invalid model_type. Supported values: 'linear' or 'logistic'.")



    def run(self):
        self.visualize_data()
        self.preprocess_data()

        # Lineer regresyon
        X_linear_train, X_linear_test, y_linear_train, y_linear_test = self.split_data(self.df[self.linear_cols], self.df['stress_level'])
        linear_model = self.train_model(X_linear_train, y_linear_train)
        self.evaluate_model(linear_model, X_linear_test, y_linear_test)

        # Lojistik regresyon
        X_logistic_train, X_logistic_test, y_logistic_train, y_logistic_test = self.split_data(self.df[self.logistic_cols], (self.df['sleep_quality'] == 1).astype(int), model_type='logistic')
        logistic_model = self.train_model(X_logistic_train, y_logistic_train, model_type='logistic')
        self.evaluate_model(logistic_model, X_logistic_test, y_logistic_test, model_type='logistic')

        
        self.plot_logistic_regression_results(X_logistic_test, y_logistic_test, logistic_model)

    def plot_logistic_regression_results(self, X_test, y_test, model):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            y_test,
            model.predict_proba(X_test)[:, 1],
            c=X_test['sleep_quality'],
            cmap='viridis', 
            alpha=0.8,
        )
        
        plt.plot([0, 1], [1, 0], color='red', linestyle='--', lw=2)
        plt.title('Logistic Regression: Actual Sleep Quality vs. Predicted Stress Level')
        plt.xlabel('Actual Sleep Quality (Class 1)')
        plt.ylabel('Predicted Stress Level')
        plt.grid(True)
        plt.show()

    def visualize_data(self):
        r, c = 4, 6
        it = 1

        plt.figure(figsize=(18, 12))

        for col in self.df.columns:
            plt.subplot(r, c, it)
            self.plot_distribution(col)
            it += 1

        plt.tight_layout(pad=2.0)  
        plt.show()

    def plot_distribution(self, col):
        plt.xlabel(col)
        plt.ylabel('Count' if self.df[col].nunique() <= 6 else 'Density')
        if self.df[col].nunique() > 6:
            sns.histplot(self.df[col], bins=20, kde=True, color='skyblue')  
            plt.grid()
        else:
            sns.countplot(x=self.df[col])

    def plot_scatter_actual_vs_predicted(self, y_true, y_pred, title):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', lw=2)
        plt.title(title)
        plt.xlabel('Actual Stress Level')
        plt.ylabel('Predicted Stress Level')
        plt.grid(True)
        plt.show()

    def plot_roc_curve(self, model, X_test, y_test):
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, linestyle='-', label='ROC curve (AUC = {:.2f})'.format(roc_auc))  # Customize ROC curve
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.fill_between(fpr, tpr, color='skyblue', alpha=0.2)  
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.show()

    def get_classification_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred)
        return accuracy, conf_matrix, classification_rep

if __name__ == "__main__":
    data_path = 'StressLevelDataset.csv'
    stress_model = StressLevelModel(data_path)
    stress_model.run()
