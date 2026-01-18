import pandas as pd
import numpy as np
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

class DataLoader:
    def __init__(self):
        self.conn_params = SNOWFLAKE_CONFIG

    def get_connection(self):
        return snowflake.connector.connect(**self.conn_params)

    def get_training_view(self):
        """
        Fetches the primary view for ML Training and analysis from Snowflake.
        Joins FACT_TRIP with Lane level risk features.
        """
        print("DataLoader: Fetching Training Data from Snowflake...")
        query = """
        WITH latest_lane_features AS (
            SELECT 
                LANE_NAME, 
                TOTAL_LANE_RISK_SCORE,
                BASE_ETA_DAYS,
                WEATHER_RISK_SCORE,
                GEOPOLITICAL_RISK_SCORE,
                LABOR_STRIKE_SCORE,
                CUSTOMS_DELAY_SCORE,
                PORT_CONGESTION_SCORE,
                CARRIER_DELAY_SCORE,
                PEAK_SEASON_SCORE,
                SNAPSHOT_DATE
            FROM DT_INGESTION.FACT_LANE_ETA_FEATURES
            QUALIFY ROW_NUMBER() OVER (PARTITION BY LANE_NAME ORDER BY SNAPSHOT_DATE DESC) = 1
        )
        SELECT 
            t.TRIP_ID as "TRIP_ID",
            t.POL as "PolCode",
            t.POD as "PodCode",
            'OCEAN' as "ModeOfTransport",
            t.CARRIER_SCAC_CODE as "Carrier",
            ABS(DATEDIFF('hour', t.POL_ATD, t.POD_ETD)) as "Actual_Duration_Hours",
            f.TOTAL_LANE_RISK_SCORE as "External_Risk_Score",
            f.BASE_ETA_DAYS as "BASE_ETA_DAYS",
            f.WEATHER_RISK_SCORE as "WEATHER_RISK_SCORE",
            f.GEOPOLITICAL_RISK_SCORE as "GEOPOLITICAL_RISK_SCORE",
            f.LABOR_STRIKE_SCORE as "LABOR_STRIKE_SCORE",
            f.CUSTOMS_DELAY_SCORE as "CUSTOMS_DELAY_SCORE",
            f.PORT_CONGESTION_SCORE as "PORT_CONGESTION_SCORE",
            f.CARRIER_DELAY_SCORE as "CARRIER_DELAY_SCORE",
            f.PEAK_SEASON_SCORE as "PEAK_SEASON_SCORE"
        FROM DT_INGESTION.FACT_TRIP t
        LEFT JOIN latest_lane_features f
          ON (CASE 
                WHEN t.POL LIKE 'US%' THEN 'North America'
                WHEN t.POL LIKE 'CN%' THEN 'Asia'
                WHEN t.POL LIKE 'NL%' THEN 'Europe'
                ELSE 'Asia' 
              END) || '-' || 
             (CASE 
                WHEN t.POD LIKE 'US%' THEN 'North America'
                WHEN t.POD LIKE 'CN%' THEN 'Asia'
                WHEN t.POD LIKE 'NL%' THEN 'Europe'
                ELSE 'Asia' 
              END) = f.LANE_NAME
        WHERE t.POL_ATD IS NOT NULL AND t.POD_ETD IS NOT NULL
        """
        
        conn = self.get_connection()
        try:
            df = pd.read_sql(query, conn)
            print(f"DataLoader: Loaded {len(df)} training records.")
            return df
        finally:
            conn.close()

    def get_trip_details(self, trip_id):
        """Returns details for a specific Trip ID from Snowflake"""
        conn = self.get_connection()
        try:
            query = f"SELECT * FROM DT_INGESTION.FACT_TRIP WHERE TRIP_ID = '{trip_id}'"
            df = pd.read_sql(query, conn)
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        finally:
            conn.close()

    def get_active_trips(self, limit=100):
        """Returns active shipments from Snowflake FACT_TRIP for the dashboard"""
        print(f"DataLoader: Fetching {limit} active shipments from Snowflake...")
        query = f"""
        SELECT 
            TRIP_ID as "id",
            POL as "origin",
            POD as "destination",
            'OCEAN' as "mode",
            'In Transit' as "status", -- Mock status for now as we don't have tracking status yet
            ABS(DATEDIFF('hour', POL_ATD, POD_ETD)) as "eta" -- Hours
        FROM DT_INGESTION.FACT_TRIP
        LIMIT {limit}
        """
        conn = self.get_connection()
        try:
            df = pd.read_sql(query, conn)
            # Convert eta to string with 'h'
            df['eta'] = df['eta'].apply(lambda x: f"{x}h")
            return df.to_dict('records')
        finally:
            conn.close()
