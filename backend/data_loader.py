import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

class DataLoader:
    def __init__(self):
        self.trips = None
        self.lanes = None
        self.carriers = None
        self.vehicles = None
        self.ext_conditions = None
        self.load_data()

    def load_data(self):
        """Loads all 5 Normalized Tables from CSV"""
        try:
            self.trips = pd.read_csv(os.path.join(DATA_DIR, "TRIP.csv"))
            self.lanes = pd.read_csv(os.path.join(DATA_DIR, "LANE.csv"))
            self.carriers = pd.read_csv(os.path.join(DATA_DIR, "CARRIER.csv"))
            self.vehicles = pd.read_csv(os.path.join(DATA_DIR, "VEHICLE.csv"))
            self.ext_conditions = pd.read_csv(os.path.join(DATA_DIR, "EXT_CONDITIONS.csv"))
            print("DataLoader: All tables loaded successfully.")
        except Exception as e:
            print(f"DataLoader Error: {e}")

    def get_training_view(self):
        """
        Reconstructs the Flat Table required for ML Training.
        Performs LEFT JOINs starting from TRIP (Fact Table).
        """
        if self.trips is None:
            self.load_data()
            
        # Start with Trip
        df = self.trips.copy()
        
        # Join Lane
        df = df.merge(self.lanes, left_on='LIN', right_on='LID', how='left', suffixes=('', '_lane'))
        
        # Join Carrier
        df = df.merge(self.carriers, left_on='CID', right_on='CID', how='left', suffixes=('', '_carrier'))
        
        # Join Vehicle
        df = df.merge(self.vehicles, left_on='VID', right_on='VID', how='left', suffixes=('', '_vehicle'))
        
        # Join Ext Conditions (on Lane)
        df = df.merge(self.ext_conditions, left_on='LIN', right_on='LID', how='left', suffixes=('', '_ext'))
        
        # Rename columns to match legacy ML expectations if needed
        # Mapping: POL -> PolCode, POD -> PodCode
        df['PolCode'] = df['POL']
        df['PodCode'] = df['POD']
        df['ModeOfTransport'] = df['VType'] # Inferred from Vehicle
        
        # Note: Original CSV had 'Actual_Duration_Hours', here we might be missing it if not in TRIP.
        # Since we generated TRIP from ETA_FEATURES, we should ensure we carry over target vars if they exist.
        # For now, we return the structure.
        
        return df

    def get_trip_details(self, trip_id):
        """Returns full details for a specific Trip ID (for API)"""
        full_view = self.get_training_view()
        match = full_view[full_view['TID'] == trip_id]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    def get_active_trips(self, limit=100):
        """Returns a list of active trips for the Dashboard"""
        full_view = self.get_training_view()
        return full_view.head(limit).to_dict('records')
