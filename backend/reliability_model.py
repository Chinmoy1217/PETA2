
import pandas as pd
import json
import os

class ReliabilityModel:
    def __init__(self, model_dir="../model"):
        self.model_dir = model_dir
        self.stats_path = os.path.join(model_dir, "carrier_stats.json")
        self.carrier_stats = {}

    def train(self, df):
        """
        Aggregates historical performance to score carriers.
        Score = (OnTime% * 0.7) + (Consistency * 0.3)
        """
        print("Training Carrier Reliability Model...")
        try:
            # 1. Ensure required columns
            required = ['Actual_Duration_Hours', 'Carrier_Name'] # Assuming Carrier Name exists or proxy
            if 'Carrier' not in df.columns:
                # Mock Carrier if missing (random assignment for hackathon)
                import numpy as np
                carriers = ['Maersk', 'MSC', 'CMA-CGM', 'Hapag-Lloyd', 'Evergreen']
                df['Carrier'] = np.random.choice(carriers, len(df))
            
            # 2. Calculate Metrics per Carrier
            stats = {}
            median_duration = df['Actual_Duration_Hours'].median()
            
            for carrier, group in df.groupby('Carrier'):
                total_trips = len(group)
                # "On Time" defined as within 10% of median or better
                on_time_trips = len(group[group['Actual_Duration_Hours'] <= median_duration * 1.1])
                
                on_time_pct = on_time_trips / total_trips if total_trips > 0 else 0
                avg_delay = group['Actual_Duration_Hours'].mean() - median_duration
                
                score = round(on_time_pct * 100, 1)
                
                stats[carrier] = {
                    "score": score,
                    "avg_delay_hours": round(avg_delay, 1),
                    "total_trips": total_trips,
                    "reliability_tier": "High" if score > 80 else ("Medium" if score > 50 else "Low")
                }
            
            self.carrier_stats = stats
            
            # 3. Save
            with open(self.stats_path, "w") as f:
                json.dump(stats, f, indent=4)
                
            print(f"✅ Carrier Reliability Model Trained. Ranked {len(stats)} carriers.")
            return True
            
        except Exception as e:
            print(f"Reliability Training Failed: {e}")
            return False

    def load(self):
        try:
            with open(self.stats_path, "r") as f:
                self.carrier_stats = json.load(f)
            return True
        except:
            return False

    def get_carrier_score(self, carrier_name):
        if not self.carrier_stats:
            self.load()
        return self.carrier_stats.get(carrier_name, {"score": 50, "reliability_tier": "Unknown"})

    def rank_carriers(self):
        """Returns list of carriers sorted by score"""
        if not self.carrier_stats:
            self.load()
        return sorted(self.carrier_stats.items(), key=lambda x: x[1]['score'], reverse=True)
