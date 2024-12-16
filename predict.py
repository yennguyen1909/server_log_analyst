import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline  
import seaborn as sns
df = pd.read_csv("D:\code-python\project_practice\server_log\dataframe\dataa.csv")
print(df.head(10))
class HourlyErrorPredictor: 
    def __init__(self): 
        self.remote_address_encoder = LabelEncoder() 
        self.user_agent_encoder = LabelEncoder() 
        self.path_encoder = LabelEncoder() 
    def preprocess_data(self, df): 
        """ Preprocess the dataframe by encoding categorical variables """ 
        df = df.copy() # Create a copy to avoid modifying the original dataframe 
        df['remote_address_encoded'] = self.remote_address_encoder.fit_transform(df['remote_address']) 
        df['user_agent_encoded'] = self.user_agent_encoder.fit_transform(df['user_agent']) 
        df['path_encoded'] = self.path_encoder.fit_transform(df['path']) 
        df['error'] = (df['status'] != 200).astype(int) 
        return df 
    def analyze_hourly_errors(self, df): 
        """ Analyze errors by hour """
        error_statuses = [400, 401, 404, 500] 
        df['error'] = df['status'].isin(error_statuses).astype(int) 
        hourly_stats = df.groupby('hour').agg( 
            error_count=('error', 'sum'), 
            # Total number of errors 
            total_requests=('status', 'count'), # Total requests (success + error) 
            error_prob=('error', 'mean') 
            ).reset_index() 
        hourly_stats['error_rate'] = (hourly_stats['error_count'] / hourly_stats['total_requests']) * 100 
            # Sort by hour for consistent visualization 
        hourly_stats = hourly_stats.sort_values('hour') 
        return hourly_stats 
    
    def build_predictive_model(self, df): 
        """ Build predictive model for error prediction """ 
        features = [ 'remote_address_encoded', 'user_agent_encoded', 'path_encoded', 'bytes_sent', 'day', 'month' ] 
        X = df[features] 
        y = df['error'] 
        # Split the data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
        # Create preprocessing steps for different feature types 
        preprocessor = ColumnTransformer(
            transformers=[ ('num', StandardScaler(), ['bytes_sent', 'day', 'month']),
                           ('cat', 'passthrough', ['remote_address_encoded', 'user_agent_encoded', 'path_encoded']) 
                           ]) 
        # Create a pipeline with scaling and random forest 
        pipeline = Pipeline([ ('preprocessor', preprocessor), 
                             ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) ]) 
        # Train the model 
        pipeline.fit(X_train, y_train) 
        # Predictions and evaluation 
        y_pred = pipeline.predict(X_test) 
        print("Classification Report:") 
        print(classification_report(y_test, y_pred)) 
        return pipeline 
    def visualize_hourly_errors(self, hourly_stats): 
        """ Visualize hourly error statistics """ 
        plt.figure(figsize=(15,10)) 
        # Error Probability Plot 
        plt.subplot(2, 1, 1) 
        plt.bar(hourly_stats['hour'], hourly_stats['error_prob'], color='coral') 
        plt.title('Hourly Error Probability') 
        plt.xlabel('Hour of Day') 
        plt.ylabel('Error Probability') 
        plt.xticks(hourly_stats['hour']) 
        # Total Requests vs Error Count Plot plt.subplot(2, 1, 2) 
        plt.plot(hourly_stats['hour'], hourly_stats['total_requests'], label='Total Requests', marker='o') 
        plt.plot(hourly_stats['hour'], hourly_stats['error_count'], label='Error Count', marker='x') 
        plt.title('Hourly Requests and Errors') 
        plt.xlabel('Hour of Day') 
        plt.ylabel('Count') 
        plt.legend() 
        plt.xticks(hourly_stats['hour']) 
        plt.tight_layout() 
        plt.show() 
    def predict_hour_error_probability(self, df, model):
        """ Predict error probabilities for each hour """
         # Prepare a dataframe with all unique hours 
        unique_hours = pd.DataFrame({'hour': range(24)}) 
        # Sample features from the original dataset 
        sample_row = df.iloc[0] 
        # Create a sample feature set for each hour 
        prediction_features = unique_hours.copy()
        prediction_features['remote_address_encoded'] = sample_row['remote_address_encoded'] 
        prediction_features['user_agent_encoded'] = sample_row['user_agent_encoded'] 
        prediction_features['path_encoded'] = sample_row['path_encoded'] 
        prediction_features['bytes_sent'] = sample_row['bytes_sent'] 
        prediction_features['day'] = sample_row['day'] 
        prediction_features['month'] = sample_row['month'] 
        # Predict probabilities 
        error_probabilities = model.predict_proba(prediction_features[ ['remote_address_encoded', 'user_agent_encoded', 'path_encoded', 'bytes_sent', 'day', 'month'] ])[:, 1] 
        # Combine hours with probabilities 
        hour_probabilities = pd.DataFrame({ 'hour': unique_hours['hour'], 'error_probability': error_probabilities * 100 }) 
        hour_probabilities['error_probability'] = hour_probabilities['error_probability'].round(5)

        return hour_probabilities 
def main(): 
    # Load your actual dataset here 
    df = pd.read_csv('D:\code-python\project_practice\server_log\dataframe\dataa.csv') 
        # Read your dataset into a dataframe 
    predictor = HourlyErrorPredictor()
         # Preprocess data 
    processed_df = predictor.preprocess_data(df) 
        # Analyze hourly errors 
    hourly_stats = predictor.analyze_hourly_errors(processed_df) 
    print("\nHourly Error Statistics:") 
    print(hourly_stats) 
        # Build predictive model 
    model = predictor.build_predictive_model(processed_df) 
        # Predict hour-wise error probabilities 
    hour_probabilities = predictor.predict_hour_error_probability(processed_df, model) 
    hour_probabilities['error_probability'] = hour_probabilities['error_probability'].round(5)
    print("\nHour-wise Error Probabilities:")
    print(hour_probabilities) 
        # Visualize results 
    predictor.visualize_hourly_errors(hourly_stats) 

if __name__ == "__main__": 
    main()