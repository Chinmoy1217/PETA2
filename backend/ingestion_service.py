import os
import requests
import snowflake.connector
import xml.etree.ElementTree as ET
import time
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# ---------- AZURE ----------
AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")
AZURE_INPUT_FOLDER = os.getenv("AZURE_STORAGE_INPUT_FOLDER", "input")
AZURE_ARCHIVE_FOLDER = os.getenv("AZURE_STORAGE_ARCHIVE_FOLDER", "Archieve")
AZURE_SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN")

AZURE_ENDPOINT = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net"
# Ensure URL is constructed correctly
AZURE_CONTAINER_URL = f"{AZURE_ENDPOINT}/{AZURE_CONTAINER}"

# ---------- SNOWFLAKE ----------
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "SYSADMIN")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

# ============================================================
# STEP 1: CHECK FILES IN AZURE INPUT FOLDER (REST API)
# ============================================================

def get_files_from_azure():
    print(f"\n🔍 Checking Azure Blob '{AZURE_INPUT_FOLDER}/' folder (REST)...")

    token = AZURE_SAS_TOKEN if AZURE_SAS_TOKEN.startswith('?') else f"?{AZURE_SAS_TOKEN}"

    list_url = (
        f"{AZURE_CONTAINER_URL}"
        f"?restype=container&comp=list"
        f"&prefix={AZURE_INPUT_FOLDER}/"
        f"{token}"
    )

    try:
        resp = requests.get(list_url)
        if resp.status_code != 200:
            print(f"❌ Failed to list blobs: {resp.status_code}")
            return []

        root = ET.fromstring(resp.content)

        blobs_with_date = []
        import email.utils

        for node in root.iter():
            if node.tag.endswith('Blob'):
                name = None
                modified = None

                for child in node:
                    if child.tag.endswith('Name'):
                        name = child.text
                    if child.tag.endswith('Properties'):
                        for prop in child:
                            if prop.tag.endswith('Last-Modified'):
                                modified = email.utils.parsedate_to_datetime(prop.text)

                if name and name.lower().endswith(".csv"):
                    blobs_with_date.append((name, modified))

        if not blobs_with_date:
            print("ℹ️ No CSV files found")
            return []

        blobs_with_date.sort(key=lambda x: x[1], reverse=True)
        latest = blobs_with_date[0][0]

        print(f"⭐ Latest file selected: {latest}")
        return [latest]

    except Exception as e:
        print(f"❌ Azure List Error: {e}")
        return []

# ============================================================
# STEP 2: SNOWFLAKE CONNECTION
# ============================================================

def snowflake_connection():
    print("\n❄️ Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            autocommit=True
        )
        print("✅ Snowflake connection established")
        return conn
    except Exception as e:
        print(f"❌ Snowflake Connection Error: {e}")
        return None

# ============================================================
# STEP 3: CHECK IF TABLE EXISTS
# ============================================================

def table_exists(conn, table_name):
    print(f"\n🔎 Checking if table '{table_name}' exists...")

    sql = f"""
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{SNOWFLAKE_SCHEMA}'
        AND TABLE_NAME = '{table_name.upper()}';
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0] == 1

# ============================================================
# STEP 4: COPY DATA INTO SNOWFLAKE
# ============================================================

def copy_into_snowflake(conn, table_name, blob):
    print(f"\n📥 Loading data into table '{table_name}'...")

    azure_file_path = (
        f"azure://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/"
        f"{AZURE_CONTAINER}/{blob}"
    )

    copy_sql = f"""
        COPY INTO {table_name}
        FROM '{azure_file_path}'
        CREDENTIALS = (AZURE_SAS_TOKEN='{AZURE_SAS_TOKEN}')
        FILE_FORMAT = (
            TYPE = CSV
            SKIP_HEADER = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        )
        ON_ERROR = 'CONTINUE';
    """

    with conn.cursor() as cur:
        cur.execute(copy_sql)
        results = cur.fetchall()   # ✅ Point 3

        for row in results:
            print(f"   ➜ COPY RESULT: {row}")

# ============================================================
# STEP 5: MOVE FILE TO ARCHIEVE FOLDER (REST API)
# ============================================================

def move_blob_to_archive(blob_name):
    filename_only = os.path.basename(blob_name)

    print(f"\n📦 Archiving file: {filename_only}")

    token = AZURE_SAS_TOKEN if AZURE_SAS_TOKEN.startswith('?') else f"?{AZURE_SAS_TOKEN}"

    source_url = f"{AZURE_CONTAINER_URL}/{blob_name}{token}"
    dest_blob = f"{AZURE_ARCHIVE_FOLDER}/{filename_only}"
    dest_url = f"{AZURE_CONTAINER_URL}/{dest_blob}{token}"

    headers = {
        "x-ms-copy-source": source_url,
        "x-ms-version": "2020-04-08"
    }

    resp = requests.put(dest_url, headers=headers)
    if resp.status_code not in (201, 202):
        print(f"❌ Copy failed: {resp.status_code}")
        return

    time.sleep(1)

    delete_url = f"{AZURE_CONTAINER_URL}/{blob_name}{token}"
    resp_del = requests.delete(delete_url)

    if resp_del.status_code in (202, 204):
        print("✅ Archived successfully")
    else:
        print(f"❌ Delete failed: {resp_del.status_code}")

# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def run_ingestion():
    print("\n🚀 Pipeline started")

    blobs = get_files_from_azure()
    if not blobs:
        print("🏁 Pipeline finished (No files)")
        return

    conn = snowflake_connection()
    if not conn:
        print("🏁 Pipeline finished (DB Connection Failed)")
        return

    try:
        for blob in blobs:
            file_name = os.path.basename(blob)
            table_name = os.path.splitext(file_name)[0]

            print("\n------------------------------------------")
            print(f"📄 Processing file: {file_name}")
            print(f"➡️ Target table: {table_name}")

            if table_exists(conn, table_name):
                copy_into_snowflake(conn, table_name, blob)
                move_blob_to_archive(blob)
            else:
                print(f"⚠️ Table '{table_name}' not found")

    finally:
        conn.close()
        print("\n🔒 Snowflake connection closed")
        print("🏁 Pipeline finished")

# ============================================================
# NEW: DIRECT INGESTION (SKIP AZURE)
# ============================================================
def ingest_direct_from_file(file_path):
    print(f"\n🚀 Direct Ingestion Started for: {file_path}")
    conn = snowflake_connection()
    if not conn:
        return {"status": "error", "message": "DB Connection Failed"}
        
    try:
        file_name = os.path.basename(file_path)
        table_name = os.path.splitext(file_name)[0]
        
        # 1. Ensure Table Exists
        if not table_exists(conn, table_name):
             print(f"❌ Table '{table_name}' does not exist. Please create it first.")
             return {"status": "error", "message": f"Table '{table_name}' not found"}

        print(f"✅ Table '{table_name}' found.")
        
        # 2. Upload to Internal Stage
        print(f"📤 Uploading {file_name} to internal stage @%{table_name}...")
        
        safe_path = file_path.replace("\\", "/")
        put_sql = f"PUT file://{safe_path} @%{table_name} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        
        with conn.cursor() as cur:
            cur.execute(put_sql)
            
        print("✅ Upload to Snowflake Internal Stage complete.")
        
        # 3. Copy into Table
        print(f"📥 Loading data into {table_name}...")
        copy_sql = f"""
            COPY INTO {table_name}
            FROM @%{table_name}/{file_name}
            FILE_FORMAT = (
                TYPE = CSV
                SKIP_HEADER = 1
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            )
            ON_ERROR = 'CONTINUE';
        """
        
        with conn.cursor() as cur:
            cur.execute(copy_sql)
            results = cur.fetchall()
            for row in results:
                print(f"   ➜ COPY RESULT: {row}")
                
        print(f"✅ Data successfully loaded into '{table_name}'")
        
        # 4. Upload to Azure Archive (REST)
        # Since we skipped Azure 'input', we upload directly to 'Archieve' for audit trails
        upload_to_archive_rest(file_path)

        return {"status": "success", "message": "Direct Ingestion & Archival Complete"}
        
    except Exception as e:
        print(f"❌ Direct Ingestion Error: {e}")
        return {"status": "error", "message": str(e)}
        
    finally:
        conn.close()

def upload_to_archive_rest(file_path):
    try:
        filename = os.path.basename(file_path)
        print(f"\n📦 Archiving '{filename}' to Azure (REST)...")
        
        # Ensure latest SAS token variable from config section is used
        token = AZURE_SAS_TOKEN if AZURE_SAS_TOKEN.startswith('?') else f"?{AZURE_SAS_TOKEN}"
        
        # Archive URL
        blob_path = f"{AZURE_ARCHIVE_FOLDER}/{filename}"
        url = f"{AZURE_ENDPOINT}/{AZURE_CONTAINER}/{blob_path}{token}"
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        headers = {
            'x-ms-blob-type': 'BlockBlob',
            'Content-Type': 'application/octet-stream',
            'x-ms-version': '2020-04-08'
        }
        
        resp = requests.put(url, data=file_data, headers=headers)
        
        if resp.status_code == 201:
            print(f"✅ Successfully archived to: {blob_path}")
        else:
            print(f"❌ Archive Upload Failed: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"❌ Archive Error: {e}")


if __name__ == "__main__":
    run_ingestion()
