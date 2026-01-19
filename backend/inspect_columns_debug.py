from backend.data_loader import DataLoader
import pandas as pd

try:
    loader = DataLoader() # Connects
    conn = loader.get_snowflake_conn()
    
    print("--- FACT_TRIP Columns ---")
    df_trip = pd.read_sql("SELECT * FROM FACT_TRIP LIMIT 0", conn)
    print(df_trip.columns.tolist())
    
    print("\n--- DIM_CARRIER Columns ---")
    df_carrier = pd.read_sql("SELECT * FROM DIM_CARRIER LIMIT 0", conn)
    print(df_carrier.columns.tolist())
    
    conn.close()
except Exception as e:
    print(e)
