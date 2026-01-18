import snowflake.connector
import os

# Credentials (Recovered from git)
ACCOUNT = "COZENTUS-DATAPRACTICE"
USER = "HACKATHON_DT"
PASSWORD = "Welcome@Start123"
WAREHOUSE = "COMPUTE_WH" # Default trial warehouse
DATABASE = "HACAKATHON" # Note: Preserving original spelling
SCHEMA = "PUBLIC"

def check_schema():
    try:
        conn = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT,
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA
        )
        cursor = conn.cursor()
        
        print("\n--- DESCRIBE FACT_TRIP ---")
        try:
            cursor.execute("DESCRIBE TABLE FACT_TRIP")
            for row in cursor.fetchall():
                print(f"Column: {row[0]}, Type: {row[1]}")
        except Exception as e:
            print(f"Error accessing FACT_TRIP: {e}")

        print("\n--- DESCRIBE DT_PREDICTIONS ---")
        try:
            cursor.execute("DESCRIBE TABLE DT_PREDICTIONS")
            for row in cursor.fetchall():
                print(f"Column: {row[0]}, Type: {row[1]}")
        except Exception as e:
            print(f"Error accessing DT_PREDICTIONS: {e}")
            
        conn.close()
        
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    check_schema()
