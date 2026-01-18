import snowflake.connector
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
from backend.ingestion_service import snowflake_connection

def run_transformation():
    print("\n🏗️ STARTING DATA TRANSFORMATION (Fact/Dims Load)...")
    
    # Use the shared connection logic
    conn = snowflake_connection()
    if not conn:
        print("❌ Transformation Aborted: No DB Connection")
        return {"status": "error", "message": "No DB Connection"}

    try:
        cs = conn.cursor()

        # ----------------------------
        # FACT_TRIP – SQL-based INSERT (Fast & Robust)
        # ----------------------------
        print("🔄 Loading FACT_TRIP (SQL Optimized)...")
        # Direct Snowflake-to-Snowflake insertion
        # Solves "maximum number of expressions in a list exceeded"
        # Fixes Logical Mapping: POL=Dep, POD=Arr
        cs.execute("""
            INSERT INTO HACAKATHON.DT_INGESTION.FACT_TRIP
            (POL, POL_ATD, POL_ETD, POD, POD_ATD, POD_ETD, CARRIER_SCAC_CODE, VESSEL_IMO)
            SELECT 
                PORT_OF_DEP,
                PORT_OF_DEP_ACT_DT,     -- ✅ no TRY_TO_TIMESTAMP
                PORT_OF_DEP_EST_DT,
                PORT_OF_ARR,
                PORT_OF_ARRVL_ACT_DT,
                PORT_OF_ARRVL_EST_DT,
                CARRIER_SCAC_CODE,
                TRY_TO_NUMBER(VESSEL_IMO)
            FROM HACAKATHON.DT_PREP.PETA
            WHERE PORT_OF_DEP IS NOT NULL 
            AND PORT_OF_ARR IS NOT NULL
        """)
        trip_count = cs.rowcount
        print(f"   ✅ FACT_TRIP Loaded: {trip_count} rows.")

        # =====================================================
        # DIM_CARRIER – SQL-based UPSERT (Much Faster)
        # =====================================================
        print("🔄 Loading DIM_CARRIER (SQL Optimized)...")
        # Logic: Insert SCACs from CARRIER and MASTER_CARRIER if they don't exist
        cs.execute("""
            INSERT INTO HACAKATHON.DT_INGESTION.DIM_CARRIER (CRR_SCAC_CD, CRR_NAME, CRR_TYPE, IS_ACTIVE, SERVICE_START_DATE)
            SELECT DISTINCT SCAC, NAME, 'Ocean', 1, '2024-01-01'
            FROM (
                SELECT CARRIER_SCAC_CODE as SCAC, CARRIER_NAME as NAME FROM HACAKATHON.DT_PREP.PETA WHERE CARRIER_SCAC_CODE IS NOT NULL
                UNION
                SELECT MASTER_CARRIER_SCAC_CODE as SCAC, MASTER_CARRIER_NAME as NAME FROM HACAKATHON.DT_PREP.PETA WHERE MASTER_CARRIER_SCAC_CODE IS NOT NULL
            ) src
            WHERE NOT EXISTS (
                SELECT 1 FROM HACAKATHON.DT_INGESTION.DIM_CARRIER tgt WHERE tgt.CRR_SCAC_CD = src.SCAC
            )
        """)
        carrier_count = cs.rowcount
        print(f"   ✅ DIM_CARRIER Loaded: {carrier_count} rows.")

        # =====================================================
        # DIM_VEHICLE – SQL-based UPSERT (Much Faster)
        # =====================================================
        print("🔄 Loading DIM_VEHICLE (SQL Optimized)...")
        # Logic: Clean IMO (Digits only) and Insert if not exists
        # Use TRY_TO_NUMBER to handle 'UNKNOWN' or other junk safely
        cs.execute("""
            INSERT INTO HACAKATHON.DT_INGESTION.DIM_VEHICLE 
            (VEHICLE_NUMBER, VEHICLE_NM, VEHICLE_SECONDARY_NUMBER, VEHICLE_CALL_SIGN)
            SELECT DISTINCT 
                TRY_TO_NUMBER(VESSEL_IMO), 
                VESSEL_NAME, 
                VESSEL_MMSI, 
                VESSEL_CALL_SIGN
            FROM HACAKATHON.DT_PREP.PETA
            WHERE VESSEL_IMO IS NOT NULL 
              AND TRY_TO_NUMBER(VESSEL_IMO) IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM HACAKATHON.DT_INGESTION.DIM_VEHICLE tgt 
                WHERE tgt.VEHICLE_NUMBER = TRY_TO_NUMBER(HACAKATHON.DT_PREP.PETA.VESSEL_IMO)
            )
        """)
        vehicle_count = cs.rowcount
        print(f"   ✅ DIM_VEHICLE Loaded: {vehicle_count} rows.")

        # =====================================================
        # DIM_PORT – SQL-based UPSERT (Much Faster)
        # =====================================================
        print("🔄 Loading DIM_PORT (SQL Optimized)...")
        cs.execute("""
            INSERT INTO HACAKATHON.DT_INGESTION.DIM_PORT (CODE, SOURCE_NAME)
            SELECT DISTINCT CODE, 'PETA'
            FROM (
                SELECT PORT_OF_ARR as CODE FROM HACAKATHON.DT_PREP.PETA
                UNION
                SELECT PORT_OF_DEP as CODE FROM HACAKATHON.DT_PREP.PETA
            ) src
            WHERE CODE IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM HACAKATHON.DT_INGESTION.DIM_PORT tgt WHERE tgt.CODE = src.CODE
            )
        """)
        port_count = cs.rowcount
        print(f"   ✅ DIM_PORT Loaded: {port_count} rows.")

        # =====================================================
        # TRUNCATE PETA
        # =====================================================
        print("🗑️ Truncating Staging Table...")
        cs.execute("TRUNCATE TABLE HACAKATHON.DT_PREP.PETA")
        print("   ✅ Staging Truncated.")

        conn.commit()
        cs.close()
        conn.close()

        msg = (f"Transformation Success! Inserted: "
               f"Trips={trip_count}, Carriers={carrier_count}, "
               f"Vehicles={vehicle_count}, Ports={port_count}")
        print(f"✅ {msg}")
        return {"status": "success", "message": msg}

    except Exception as e:
        print(f"❌ Transformation Error: {e}")
        return {"status": "error", "message": f"Transformation Error: {e}"}
