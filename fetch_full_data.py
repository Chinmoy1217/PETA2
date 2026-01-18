
import snowflake.connector
import pandas as pd
import os
from backend.config import SNOWFLAKE_CONFIG

def fetch_data():
    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    
    try:
        # Trip Data (Full History)
        sql_trip = """
            SELECT 
                TRIP_ID, 
                POL as POL_CODE, 
                POD as POD_CODE, 
                CARRIER_SCAC_CODE,
                POL_ATD,
                'OCEAN' as MODE_OF_TRANSPORT, 
                ABS(DATEDIFF('hour', POL_ATD, POD_ATD)) as ACTUAL_DURATION_HOURS
            FROM DT_INGESTION.FACT_TRIP 
            WHERE POD_ATD IS NOT NULL AND POL_ATD IS NOT NULL
        """
        print("Executing Trip Query (Full History)...")
        df_trip = pd.read_sql(sql_trip, conn)
        print(f"Fetched {len(df_trip)} trips.")

        # Lane Features
        sql_lane = """
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
                PEAK_SEASON_SCORE
            FROM DT_INGESTION.FACT_LANE_ETA_FEATURES
            QUALIFY ROW_NUMBER() OVER (PARTITION BY LANE_NAME ORDER BY SNAPSHOT_DATE DESC) = 1
        """
        print("Executing Lane Feature Query...")
        df_lane = pd.read_sql(sql_lane, conn)
        print(f"Fetched {len(df_lane)} lane features.")
        
        # Merge Logic (Replicated from main.py)
        def get_region_name(code):
            if not code or len(str(code)) < 2: return 'Unknown'
            country = str(code)[:2].upper()
            if country in ['US', 'CA', 'MX']: return 'North America'
            if country == 'IN': return 'India'
            if country in ['CN', 'JP', 'KR', 'SG', 'MY', 'VN', 'TH', 'HK', 'TW']: return 'Asia'
            if country in ['DE', 'GB', 'NL', 'FR', 'ES', 'IT', 'BE', 'PL']: return 'Europe'
            if country in ['AE', 'SA', 'OM', 'QA', 'KW']: return 'Middle East'
            return 'Asia'

        print("Merging Data...")
        df_trip['Origin_Region'] = df_trip['POL_CODE'].apply(get_region_name)
        df_trip['Dest_Region'] = df_trip['POD_CODE'].apply(get_region_name)
        df_trip['LANE_NAME'] = df_trip['Origin_Region'] + '-' + df_trip['Dest_Region']
        
        df_merged = pd.merge(df_trip, df_lane, on='LANE_NAME', how='left')
        
        # Renaissance Schema Mapping
        final_df = pd.DataFrame()
        final_df['PolCode'] = df_merged['POL_CODE']
        final_df['PodCode'] = df_merged['POD_CODE']
        final_df['ModeOfTransport'] = df_merged['MODE_OF_TRANSPORT']
        final_df['Actual_Duration_Hours'] = df_merged['ACTUAL_DURATION_HOURS']
        final_df['trip_id'] = df_merged['TRIP_ID']
        final_df['Carrier'] = df_merged.get('CARRIER_SCAC_CODE', 'UNKNOWN')
        
        final_df['External_Risk_Score'] = df_merged['TOTAL_LANE_RISK_SCORE']
        final_df['BASE_ETA_DAYS'] = df_merged['BASE_ETA_DAYS']
        final_df['WEATHER_RISK_SCORE'] = df_merged['WEATHER_RISK_SCORE']
        final_df['GEOPOLITICAL_RISK_SCORE'] = df_merged['GEOPOLITICAL_RISK_SCORE']
        final_df['LABOR_STRIKE_SCORE'] = df_merged['LABOR_STRIKE_SCORE']
        final_df['CUSTOMS_DELAY_SCORE'] = df_merged['CUSTOMS_DELAY_SCORE']
        final_df['PORT_CONGESTION_SCORE'] = df_merged['PORT_CONGESTION_SCORE']
        final_df['CARRIER_DELAY_SCORE'] = df_merged['CARRIER_DELAY_SCORE']
        final_df['PEAK_SEASON_SCORE'] = df_merged['PEAK_SEASON_SCORE']
        
        # Real Date
        final_df['trip_ATD'] = df_merged['POL_ATD']
        
        # Save
        print("Saving to local_full_data.pkl...")
        final_df.to_pickle("local_full_data.pkl")
        print("Done.")
        
    finally:
        conn.close()

if __name__ == "__main__":
    fetch_data()
