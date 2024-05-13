import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

class RegressionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.cat_cols = ['vin', 'year', 'make', 'model', 'trim', 'body_type', 'vehicle_type',
                         'drivetrain', 'transmission', 'fuel_type', 'engine_size', 'engine_block',
                         'seller_name', 'street', 'city', 'state', 'zip']
        self.encoder = OrdinalEncoder()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = RandomForestRegressor(n_estimators=100)
        self.y_pred = None
        self.rmse = None
        self.df_pred_actual = None

    def load_data(self):
        try:
            # Specify low_memory=False to suppress DtypeWarnings
            self.df = pd.read_csv(self.data_path, dtype={'vin': str}, low_memory=False)
            print(self.df.info())  # Display basic info about the dataset
            self.df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
            print(self.df.head())
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")

    def preprocess_data(self):
        # Drop rows with missing target variable ('price')
        self.df.dropna(subset=['price'], inplace=True)

        # Handle mixed types more explicitly for columns with mixed types
        columns_with_mixed_types = ['year']  # Add other columns with mixed types if needed
        self.df[columns_with_mixed_types] = self.df[columns_with_mixed_types].apply(pd.to_numeric, errors='coerce')

        # Use a more sophisticated imputer or encoder if needed
        self.df[self.cat_cols] = self.encoder.fit_transform(self.df[self.cat_cols])

        # Convert non-numeric values to NaN and drop rows with NaN values
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        self.df.dropna(inplace=True)

        print(self.df.head())

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df.drop(columns=['price']),
            self.df['price'],
            test_size=0.2,
            random_state=42  # Set a specific random state for reproducibility
        )

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def calculate_rmse(self):
        self.rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)
        print('RMSE:', self.rmse)

    def analyze_results(self):
        self.df_pred_actual = pd.DataFrame({
            'Actual Price': self.y_test.values,
            'Predicted Price': self.y_pred
        })
        print(self.df_pred_actual.head(10))

    def visualize_predictions(self):
        sns.set_style("darkgrid")
        plt.scatter(self.y_test, self.y_pred)
        plt.plot([min(self.y_test), max(self.y_test)],
                 [min(self.y_test), max(self.y_test)], "--", color="red")
        plt.xlim([min(self.y_test), max(self.y_test)])
        plt.ylim([min(self.y_test), max(self.y_test)])
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Predicted vs Actual Price (RMSE={:.2f})".format(self.rmse))
        plt.show()

    def visualize_feature_importance(self, top_n=10):
        feat_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
        top_feats = feat_importances.nlargest(top_n)

        top_feats.plot(kind='barh')
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title(f"Top {top_n} Important Features")
        plt.show()



# Create an instance of the RegressionModel class and use it
model = RegressionModel('us-dealers-used.csv')
model.load_data()
model.preprocess_data()
model.split_data()
model.train_model()
model.predict()
model.calculate_rmse()
model.analyze_results()
model.visualize_predictions()
model.visualize_feature_importance(top_n=21)