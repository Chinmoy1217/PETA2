import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG
import pandas as pd

def check_tables():
    print("Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cs = conn.cursor()
        
        # 1. List Tables
        print("\n[checking tables in DT_INGESTION...]")
        cs.execute("SHOW TABLES LIKE 'BATCH_RESULTS_NODE_%' IN SCHEMA DT_INGESTION")
        tables = cs.fetchall()
        
        if not tables:
            print("❌ No 'BATCH_RESULTS_NODE_' tables found.")
            return
            
        # Tables format: (created_on, name, ...) - usually index 1 is name
        # We need to sort by name (timestamp) descending
        # Inspect columns to be sure. SHOW TABLES returns many cols.
        # Index 1 is usually name.
        
        table_names = [t[1] for t in tables]
        table_names.sort(reverse=True)
        
        latest = table_names[0]
        print(f"✅ Found {len(table_names)} result tables.")
        print(f"📌 Latest Table: {latest}")
        
        # 2. Verify Content
        print(f"\n[Verifying Data in {latest}...]")
        df = pd.read_sql(f"SELECT * FROM DT_INGESTION.{latest} LIMIT 5", conn)
        print(f"   -> Row Count (approx): {len(df)} sample rows fetched.")
        if not df.empty:
            print("   -> Sample Data:")
            print(df[['TRIP_ID', 'PETA_PREDICTED_DURATION', 'ESTIMATED_ATA']].to_string(index=False))
        else:
             print("   ⚠️  Table is empty.")
             
        # 3. Check Count
        cs.execute(f"SELECT COUNT(*) FROM DT_INGESTION.{latest}")
        total = cs.fetchone()[0]
        print(f"\n📊 Total Records in {latest}: {total}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_tables()
