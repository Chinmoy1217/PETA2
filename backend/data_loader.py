import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Snowflake Config
USER = "HACKATHON_DT"
PASSWORD = "eyJraWQiOiIxOTMxNTY4MzQxMDAzOTM3OCIsImFsZyI6IkVTMjU2In0.eyJwIjoiMjk0NzMzOTQwNDEzOjI5NDczMzk0MjUzMyIsImlzcyI6IlNGOjIwMTciLCJleHAiOjE3NzEyMjY3MTF9.O0OTFEyQPIqpdCsNuV881UG1RtQQLBMIyUt-0kfESVYaI0J_u3S4fysE7lee7lWMIMoezOhd2t7gUItdoHC0UA"
ACCOUNT = "COZENTUS-DATAPRACTICE"
DATABASE = "HACAKATHON"
SCHEMA = "DT_PREP"
ROLE = "SYSADMIN"
WAREHOUSE = "COZENTUS_WH"

class DataLoader:
    def __init__(self):
        self.trips = None
        self.lanes = None
        self.carriers = None
        self.vehicles = None
        self.ext_conditions = None
        self.load_data()

    def load_data(self):
        """Generates Synthetic Data for Testing"""
        print("DataLoader: GENERATING MOCK DATA (No Database Connection)...")
        
        # 1. Mock LANES
        self.lanes = pd.DataFrame([
            {'LID': 'L001', 'POL': 'USLAX', 'POD': 'CNSHA', 'Distance': 10500},
            {'LID': 'L002', 'POL': 'NLROT', 'POD': 'USNYC', 'Distance': 6000},
            {'LID': 'L003', 'POL': 'SGSIN', 'POD': 'AEDXB', 'Distance': 3500},
            {'LID': 'L004', 'POL': 'CNHKG', 'POD': 'USLAX', 'Distance': 11000},
        ])

        # 2. Mock CARRIERS
        self.carriers = pd.DataFrame([
            {'CID': 'C001', 'CNm': 'Maersk Line', 'Reliability': 0.95},
            {'CID': 'C002', 'CNm': 'MSC', 'Reliability': 0.92},
            {'CID': 'C003', 'CNm': 'CMA CGM', 'Reliability': 0.88},
            {'CID': 'C004', 'CNm': 'FedEx Air', 'Reliability': 0.99},
        ])

        # 3. Mock VEHICLES
        self.vehicles = pd.DataFrame([
            {'VID': 'V001', 'VNm': 'Maersk Mc-Kinney', 'VType': 'Ocean'},
            {'VID': 'V002', 'VNm': 'MSC Gulsub', 'VType': 'Ocean'},
            {'VID': 'V003', 'VNm': 'Boeing 777F', 'VType': 'Air'},
            {'VID': 'V004', 'VNm': 'Volvo FH16', 'VType': 'Truck'},
        ])

        # 4. Mock EXT CONDITIONS
        self.ext_conditions = pd.DataFrame([
            {'LID': 'L001', 'Factors': 'Congestion', 'Severity_Score': 40},
            {'LID': 'L002', 'Factors': 'Storm', 'Severity_Score': 60},
        ])

        # 5. Mock TRIPS
        num_trips = 100
        trips = []
        for i in range(num_trips):
            # Random selections
            lane = self.lanes.sample(1).iloc[0]
            carrier = self.carriers.sample(1).iloc[0]
            vehicle = self.vehicles.sample(1).iloc[0]
            
            # Dates
            atd = datetime.now() - timedelta(days=np.random.randint(1, 30))
            duration = (lane['Distance'] / 30) + np.random.normal(0, 24) # Rough hours calc
            if vehicle['VType'] == 'Air': duration = (lane['Distance'] / 800) + np.random.normal(0, 2)
            
            ata = atd + timedelta(hours=duration)
            
            trips.append({
                'TID': f"T{1000+i}",
                'LIN': lane['LID'],
                'CID': carrier['CID'],
                'VID': vehicle['VID'],
                'Transport_Vehicle_ID': f"{vehicle['VNm']}-{i}",
                'ATD': atd,
                'ATA': ata,
                'Actual_Duration_Hours': duration
            })
            
        self.trips = pd.DataFrame(trips)
        print(f"DataLoader: Generated {len(self.trips)} mock trips.")

    def get_training_view(self):
        """
        Reconstructs the Flat Table required for ML Training.
        Performs in-memory joins using the loaded Snowflake data.
        """
        if self.trips is None:
            self.load_data()
            
        # Start with Trip
        df = self.trips.copy()
        
        # Join Lane (LIN -> LID)
        df = df.merge(self.lanes, left_on='LIN', right_on='LID', how='left', suffixes=('', '_lane'))
        
        # Join Carrier (CID -> CID)
        df = df.merge(self.carriers, left_on='CID', right_on='CID', how='left', suffixes=('', '_carrier'))
        
        # Join Vehicle (VID -> VID)
        df = df.merge(self.vehicles, left_on='VID', right_on='VID', how='left', suffixes=('', '_vehicle'))
        
        # Join Ext Conditions (LIN -> LID)
        df = df.merge(self.ext_conditions, left_on='LIN', right_on='LID', how='left', suffixes=('', '_ext'))
        
        # Feature Mapping
        df['PolCode'] = df['POL']
        df['PodCode'] = df['POD']
        df['ModeOfTransport'] = df.get('VType', 'Unknown') 
        df['External_Risk_Score'] = df.get('Severity_Score', 0).fillna(0) 
        
        return df

    def get_trip_details(self, trip_id):
        """Returns full details for a specific Trip ID (for API)"""
        if self.trips is None: return None
        full_view = self.get_training_view()
        match = full_view[full_view['TID'] == trip_id]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    def get_active_trips(self, limit=100):
        """Returns a list of active trips for the Dashboard"""
        if self.trips is None: self.load_data()
        full_view = self.get_training_view()
        return full_view.head(limit).to_dict('records')
