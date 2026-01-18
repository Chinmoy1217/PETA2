import pandas as pd
import os
import snowflake.connector

# Snowflake Config
USER = "HACKATHON_DT"
PASSWORD = "Welcome@Start123"
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

    def get_snowflake_conn(self):
        return snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT,
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA,
            role=ROLE
        )

    def load_data(self):
        """Loads all 5 Tables from SNOWFLAKE"""
        print("DataLoader: Connecting to Snowflake...")
        try:
            conn = self.get_snowflake_conn()
            
            # 1. LANE
            print("Fetching DIM_LANE...")
            self.lanes = pd.read_sql("SELECT * FROM DIM_LANE", conn)
            # Snowflake returns uppercase columns, map to internal expected names
            self.lanes.rename(columns={'LID': 'LID', 'POL': 'POL', 'POD': 'POD'}, inplace=True) 

            # 2. CARRIER
            print("Fetching DIM_CARRIER...")
            self.carriers = pd.read_sql("SELECT * FROM DIM_CARRIER", conn)
            self.carriers.rename(columns={'CNM': 'CNm'}, inplace=True) # Map CNM -> CNm

            # 3. VEHICLE
            print("Fetching DIM_VEHICLE...")
            self.vehicles = pd.read_sql("SELECT * FROM DIM_VEHICLE", conn)
            self.vehicles.rename(columns={'VNM': 'VNm', 'VTYPE': 'VType'}, inplace=True)

            # 4. EXT_CONDITIONS
            print("Fetching FACT_EXT_CONDITIONS...")
            self.ext_conditions = pd.read_sql("SELECT * FROM FACT_EXT_CONDITIONS", conn)
            self.ext_conditions.rename(columns={'FACTORS': 'Factors', 'SEVERITY_SCORE': 'Severity_Score'}, inplace=True)

            # 5. TRIP (Calculated Fields)
            print("Fetching FACT_TRIP...")
            # We need Actual_Duration_Hours for training
            query = """
            SELECT 
                *, 
                DATEDIFF(hour, ATD, ATA) as "Actual_Duration_Hours" 
            FROM FACT_TRIP
            WHERE ATD IS NOT NULL AND ATA IS NOT NULL
            """
            self.trips = pd.read_sql(query, conn)
            # Map LID -> LIN to match internal join logic
            self.trips.rename(columns={'LID': 'LIN'}, inplace=True)

            conn.close()
            print(f"DataLoader: Loaded {len(self.trips)} trips from Snowflake.")
            
        except Exception as e:
            print(f"DataLoader Error (Snowflake): {e}")
            # Fallback or Raise? User said "no more csv use", so we raise/fail.
            raise e

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
        # Note: EXT_CONDITIONS in Snowflake connects via LID
        df = df.merge(self.ext_conditions, left_on='LIN', right_on='LID', how='left', suffixes=('', '_ext'))
        
        # Feature Mapping
        df['PolCode'] = df['POL']
        df['PodCode'] = df['POD']
        df['ModeOfTransport'] = df.get('VType', 'Unknown') 
        
        return df

    def get_trip_details(self, trip_id):
        """Returns full details for a specific Trip ID (for API)"""
        if self.trips is None: return None
        # Convert to string for comparison if needed, though usually int/str consistency matters
        # Snowflake IDs might be Numbers/Ints.
        pass # Optimization: Query snowflake directly for single ID? 
        # For now, stick to cache
        full_view = self.get_training_view()
        match = full_view[full_view['TID'] == trip_id]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    def get_active_trips(self, limit=100):
        """Returns a list of active trips for the Dashboard"""
        if self.trips is None: self.load_data()
        
        # Simple filter for "Active" - example: has ETD but no ATA? 
        # Or just latest.
        full_view = self.get_training_view()
        return full_view.head(limit).to_dict('records')
