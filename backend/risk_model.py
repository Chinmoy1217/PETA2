
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class RiskModel:
    def __init__(self, model_dir="../model"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "risk_classifier.pkl")
        self.model = None
        self.feature_cols = ['External_Risk_Score', 'WEATHER_RISK_SCORE', 'PEAK_SEASON_SCORE', 
                             'PORT_CONGESTION_SCORE', 'LABOR_STRIKE_SCORE']

    def train(self, df):
        """
        Trains a Random Forest Classifier to predict if a shipment will be DELAYED
        (Actual Duration > Base Duration + 3 Days buffer).
        """
        try:
            print("Training Risk Classification Model...")
            
            # 1. Define Target: Delayed if Actual > 3 days (72h) vs baseline logic
            # For hackathon data, we use median as baseline proxy if BASE_ETA missing
            median_duration = df['Actual_Duration_Hours'].median()
            df['Is_Delayed'] = (df['Actual_Duration_Hours'] > median_duration * 1.2).astype(int)
            
            # 2. Prepare Features
            # Ensure columns exist, fill with 0
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
            
            X = df[self.feature_cols].fillna(0)
            y = df['Is_Delayed']
            
            # 3. Train
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X, y)
            
            # 4. Save
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                
            with open(self.model_path, "wb") as f:
                pickle.dump(clf, f)
                
            self.model = clf
            print("✅ Risk Model Trained & Saved.")
            return True
            
        except Exception as e:
            print(f"Risk Model Training Failed: {e}")
            return False

    def load(self):
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        except:
            return False

    def predict_risk_proba(self, features_dict):
        """
        Returns probability of delay (0.0 to 1.0)
        """
        if not self.model:
            if not self.load():
                return 0.5 # Default uncertainty
        
        # Convert dict to DF row
        input_data = pd.DataFrame([features_dict])
        
        # Access missing columns safely
        for col in self.feature_cols:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Predict Class 1 (Delayed) probability
        prob = self.model.predict_proba(input_data[self.feature_cols])[0][1]
        return round(prob, 2)
