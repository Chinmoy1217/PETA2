import sys
import types

# --- MONKEY PATCHES for Legacy Python Compatibility ---
import collections
import collections.abc
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence

try:
    import cgi
except ImportError:
    # Create a dummy cgi module
    dummy_cgi = types.ModuleType('cgi')
    sys.modules['cgi'] = dummy_cgi
    
import snowflake.connector
import pandas as pd

# Credentials
ACCOUNT = "COZENTUS-DATAPRACTICE"
USER = "HACKATHON_DT"
PASSWORD = "Welcome@Start123"
WAREHOUSE = "COMPUTE_WH"
DATABASE = "HACAKATHON"
SCHEMA = "PUBLIC"

def inspect_all():
    print(f"Connecting to Account: {ACCOUNT}, Database: {DATABASE}...")
    try:
        conn = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT,
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA
        )
        cur = conn.cursor()
        
        # 1. List All Tables
        print("\n--- 1. DISCOVERING TABLES ---")
        cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'PUBLIC'")
        tables = [row[0] for row in cur.fetchall()]
        print(f"Found Tables: {tables}")
        
        # 2. Inspect Each Table
        for table in tables:
            print(f"\n{'='*30}")
            print(f"TABLE: {table}")
            print(f"{'='*30}")
            
            # Columns
            try:
                cur.execute(f"DESCRIBE TABLE {table}")
                cols = cur.fetchall()
                print("COLUMNS:")
                print(f"{'Name':<25} {'Type':<15}")
                print("-" * 40)
                for c in cols:
                    print(f"{c[0]:<25} {c[1]:<15}")
            except Exception as e:
                print(f"Error describing {table}: {e}")

            # Sample Data
            try:
                print("\nSAMPLE DATA (First 3 Rows):")
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
                if not df.empty:
                    print(df.to_string(index=False))
                else:
                    print("(Table is empty)")
            except Exception as e:
                print(f"Error fetching data from {table}: {e}")

        conn.close()
        print("\n--- INSPECTION COMPLETE ---")
        
    except Exception as e:
        print(f"\nFATAL CRASH: {e}")

if __name__ == "__main__":
    inspect_all()
