import os
import uuid
import json
import time
import requests
import traceback
import pandas as pd
from datetime import datetime, timedelta
from backend.ingestion_service import snowflake_connection
from backend.model_engine import PetaModelEngine # Fallback if direct call needed, but preferring API

# CONFIG
PREDICTION_API_URL = "http://127.0.0.1:8000/predict" # Local for now
BATCH_SIZE = 500

def get_snowflake_conn():
    return snowflake_connection()

def get_incremental_shipments(conn, limit=BATCH_SIZE):
    """
    Fetch shipments from DT_PREP.PETA that are newer than the latest prediction.
    """
    cursor = conn.cursor()
    try:
        # 1. Get max departure date from predictions to know where to resume
        # Using 1900-01-01 as fallback if table is empty
        cursor.execute("""
            SELECT COALESCE(MAX(DEPARTURE_DATE), DATE '1900-01-01') 
            FROM HACAKATHON.DT_INGESTION.PETA_PREDICTION_RESULTS
        """)
        last_date_row = cursor.fetchone()
        last_date = last_date_row[0] if last_date_row else datetime(1900, 1, 1).date()
        
        print(f"🔎 Incremental Scan: Looking for shipments after {last_date}")

        # 2. Query Incremental Rows
        # Notes:
        # - Using COALESCE(ACT, EST) for Departure Date
        # - Ordering by Date to process oldest first
        query = f"""
            SELECT 
                PORT_OF_DEP, 
                PORT_OF_ARR, 
                COALESCE(PORT_OF_DEP_ACT_DT, PORT_OF_DEP_EST_DT) AS DEP_DT,
                CARRIER_SCAC_CODE,
                VESSEL_IMO
            FROM HACAKATHON.DT_PREP.PETA
            WHERE COALESCE(PORT_OF_DEP_ACT_DT, PORT_OF_DEP_EST_DT) > %s
            ORDER BY DEP_DT ASC
            LIMIT {limit}
        """
        cursor.execute(query, (last_date,))
        rows = cursor.fetchall()
        return rows # List of tuples
        
    finally:
        cursor.close()

def enrich_shipment(conn, carrier, pol, pod, dep_date):
    """
    Query FACT_LANE_ETA_FEATURES for congestion/weather scores.
    Key: CARRIER_SCAC_CODE, LANE_NAME (POL-POD), SNAPSHOT_DATE (Approx match or latest)
    """
    lane_name = f"{pol}-{pod}"
    cursor = conn.cursor()
    try:
        # Simple lookup: Find latest snapshot <= departure_date
        # If specific date not found, fallback to latest avail for that lane/carrier
        query = """
            SELECT WEATHER_RISK_SCORE, PORT_CONGESTION_SCORE
            FROM HACAKATHON.DT_INGESTION.FACT_LANE_ETA_FEATURES
            WHERE CARRIER_SCAC_CODE = %s 
              AND LANE_NAME = %s
              AND SNAPSHOT_DATE <= %s
            ORDER BY SNAPSHOT_DATE DESC
            LIMIT 1
        """
        cursor.execute(query, (carrier, lane_name, dep_date))
        row = cursor.fetchone()
        
        if row:
            return {"weather": row[0] or 10, "congestion": row[1] or 10}
        else:
            return {"weather": 10, "congestion": 10} # Defaults
            
    except Exception as e:
        print(f"⚠️ Enrichment warning: {e}")
        return {"weather": 10, "congestion": 10}
    finally:
        cursor.close()

def insert_to_dlq(conn, payload, error_msg):
    """Insert failed record into Dead Letter Queue"""
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO HACAKATHON.DT_INGESTION.PETA_PREDICT_DLQ
            (DLQ_ID, PREDICTION_INPUT, ERROR_MSG, ERROR_AT)
            VALUES (%s, PARSE_JSON(%s), %s, CURRENT_TIMESTAMP())
        """
        dlq_id = str(uuid.uuid4())
        cursor.execute(query, (dlq_id, json.dumps(payload), str(error_msg)))
        conn.commit()
        print(f"📥 Saved to DLQ: {dlq_id}")
    except Exception as ie:
        print(f"❌ Critical DLQ Failure: {ie}")
    finally:
        cursor.close()

def save_prediction(conn, record):
    """Idempotent Insert into Results Table"""
    cursor = conn.cursor()
    try:
        # Check existence (Idempotency)
        check_sql = """
            SELECT 1 FROM HACAKATHON.DT_INGESTION.PETA_PREDICTION_RESULTS
            WHERE POL_CODE=%s AND POD_CODE=%s 
              AND CARRIER_SCAC_CODE=%s AND DEPARTURE_DATE=%s
            LIMIT 1
        """
        cursor.execute(check_sql, (record['pol'], record['pod'], record['carrier'], record['dep_date']))
        if cursor.fetchone():
            print(f"⏭️ Skipping Duplicate: {record['pol']}-{record['pod']} on {record['dep_date']}")
            return

        insert_sql = """
            INSERT INTO HACAKATHON.DT_INGESTION.PETA_PREDICTION_RESULTS (
              PREDICTION_ID, POL_CODE, POD_CODE, MODE_OF_TRANSPORT, CARRIER_SCAC_CODE,
              DEPARTURE_DATE, CONGESTION_LEVEL, WEATHER_SEVERITY, 
              PETA_HOURS, PETA_DAYS, PETA_DATE, PETA_DATETIME, 
              ROUTE, STATUS, CREATED_BY
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Calculate derived fields
        pred_uuid = str(uuid.uuid4())
        p_hours = float(record['peta_hours'])
        p_days = p_hours / 24.0
        
        # Calculate ETA Date/Time
        # Assuming dep_date is DATE, converting to datetime first if needed
        # Or if we have a departure_timestamp, use that. Here we only have date usually.
        # Let's assume start of day for departure date if time unavailable
        dep_dt_obj = datetime.combine(record['dep_date'], datetime.min.time())
        eta_dt_obj = dep_dt_obj + timedelta(hours=p_hours)
        
        cursor.execute(insert_sql, (
            pred_uuid,
            record['pol'],
            record['pod'],
            "OCEAN",
            record['carrier'],
            record['dep_date'],
            record['congestion'],
            record['weather'],
            p_hours,
            p_days,
            eta_dt_obj.date(),
            eta_dt_obj,
            f"{record['pol']}-{record['pod']}",
            "PREDICTED",
            "PETA_API"
        ))
        conn.commit()
        print(f"✅ Prediction Saved: {pred_uuid}")

    finally:
        cursor.close()


def run_prediction_pipeline():
    print("\n🚀 Starting Incremental Prediction Pipeline...")
    
    conn = get_snowflake_conn()
    if not conn:
        print("❌ Pipeline Failed: No DB Connection")
        return {"status": "error", "message": "No DB Connection"}

    try:
        # 1. Fetch Incremental Data
        shipments = get_incremental_shipments(conn, limit=BATCH_SIZE)
        if not shipments:
            print("ℹ️ No new shipments to process.")
            return {"status": "success", "message": "No new shipments found.", "processed": 0}

        print(f"📦 Found {len(shipments)} new shipments to process.")
        
        success_count = 0
        fail_count = 0

        # 2. Process Each Shipment
        for row in shipments:
            # Unpack row (POL, POD, DEP_DT, CARRIER, IMO)
            pol, pod, dep_dt, carrier, imo = row
            
            # Normalize Date
            # dep_dt might be datetime or date
            dep_date = dep_dt.date() if isinstance(dep_dt, datetime) else dep_dt
            
            try:
                # A. Enrich
                features = enrich_shipment(conn, carrier, pol, pod, dep_date)
                
                # B. Prepare Prediction Payload
                payload = {
                    "PolCode": pol,
                    "PodCode": pod,
                    "ModeOfTransport": "Ocean", # Hardcoded
                    "Carrier": carrier,
                    "trip_ATD": dep_date.isoformat(),
                    "congestion_level": int(features['congestion']),
                    "weather_severity": int(features['weather']),
                    "VesselIMO": str(imo) if imo else None
                }
                
                # C. Call Prediction Engine 
                # (Simulating API call roughly, or importing engine if acceptable)
                # Using requests to localhost provided API usually works best for separation
                try:
                    # Retry logic (simple)
                    resp = requests.post(PREDICTION_API_URL, json=payload, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()
                    peta_hours = result.get('prediction_hours', 240) # Default if missing
                except Exception as api_err:
                    # If API is down, maybe fail fast? or DLQ?
                    # Let's assume DLQ for resilience
                    raise Exception(f"Prediction API Failed: {str(api_err)}")

                # D. Save Result
                record = {
                    "pol": pol, "pod": pod, "carrier": carrier, "dep_date": dep_date,
                    "congestion": int(features['congestion']),
                    "weather": int(features['weather']),
                    "peta_hours": peta_hours
                }
                save_prediction(conn, record)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                print(f"❌ Row Error: {e}")
                insert_to_dlq(conn, {"pol": pol, "pod": pod, "err": str(e)}, str(e))

        return {
            "status": "success", 
            "message": f"Processed {len(shipments)} shipments. Success: {success_count}, Failed: {fail_count}",
            "processed": len(shipments),
            "errors": fail_count
        }

    except Exception as fatal_e:
        print(f"❌ Pipeline Fatal Error: {fatal_e}")
        traceback.print_exc()
        return {"status": "error", "message": str(fatal_e)}
        
    finally:
        if conn: conn.close()
        print("🏁 Prediction Pipeline Finished.")

if __name__ == "__main__":
    run_prediction_pipeline()
