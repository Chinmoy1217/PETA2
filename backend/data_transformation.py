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
        # Read PETA from STAGING (DT_PREP.PETA)
        # ----------------------------
        print("📊 Reading from HACAKATHON.DT_PREP.PETA...")
        peta_df = pd.read_sql(
            "SELECT * FROM HACAKATHON.DT_PREP.PETA",
            conn
        )
        print(f"   Rows fetched: {len(peta_df)}")

        if peta_df.empty:
            print("⚠️ No data in staging table. Skipping transformation.")
            return {"status": "warning", "message": "No data to transform"}

        # =====================================================
        # FACT_TRIP – insert all rows
        # =====================================================
        print("🔄 Loading FACT_TRIP...")
        
        # Ensure target table exists (Optional, assuming it does based on user script)
        # We can add explicit create if needed.
        
        
        # --------------------------------------------------------
        # FIX: Convert Timestamps to Strings explicitly
        # Snowflake Connector can struggle with pandas Timestamps in executemany
        # --------------------------------------------------------
        def clean_ts(ts):
            if pd.isna(ts) or str(ts).strip() == "":
                return None
            try:
                # If it's already a string, return it (trimming potential whitespace)
                if isinstance(ts, str):
                    return ts.strip()
                # If it's a timestamp/datetime, stringify it
                return ts.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                return str(ts)

        # Apply conversion to relevant columns in peta_df BEFORE creating fact_trip_df
        ts_cols = ["PORT_OF_ARRVL_ACT_DT", "PORT_OF_ARRVL_EST_DT", "PORT_OF_DEP_EST_DT", "PORT_OF_DEP_ACT_DT"]
        for col in ts_cols:
            if col in peta_df.columns:
                 peta_df[col] = peta_df[col].apply(clean_ts)

        # --------------------------------------------------------
        # FIX: Clean VESSEL_IMO (Handle 'UNKNOWN' and non-numerics)
        # --------------------------------------------------------
        def clean_imo(val):
            if pd.isna(val): return None
            s = str(val).strip().upper()
            if not s.isdigit():
                return None # Return None for 'UNKNOWN' or alpha-numeric
            return s # Return as string, Snowflake handles string-to-number if digits

        if "VESSEL_IMO" in peta_df.columns:
            peta_df["VESSEL_IMO"] = peta_df["VESSEL_IMO"].apply(clean_imo)
            
        fact_trip_df = pd.DataFrame({
            "POL": peta_df["PORT_OF_ARR"],
            "POL_ATD": peta_df["PORT_OF_ARRVL_ACT_DT"],
            "POL_ETD": peta_df["PORT_OF_ARRVL_EST_DT"],
            "POD": peta_df["PORT_OF_DEP"],
            "POD_ATD": peta_df["PORT_OF_DEP_EST_DT"],
            "POD_ETD": peta_df["PORT_OF_DEP_ACT_DT"],
            "CARRIER_SCAC_CODE": peta_df["CARRIER_SCAC_CODE"],
            "VESSEL_IMO": peta_df["VESSEL_IMO"]
        })
        
        # Handle Potential NaN/None for SQL
        fact_trip_df = fact_trip_df.where(pd.notnull(fact_trip_df), None)

        cs.executemany(
            """
            INSERT INTO HACAKATHON.DT_INGESTION.FACT_TRIP
            (POL, POL_ATD, POL_ETD, POD, POD_ATD, POD_ETD, CARRIER_SCAC_CODE, VESSEL_IMO)
            VALUES (%(POL)s, %(POL_ATD)s, %(POL_ETD)s, %(POD)s, %(POD_ATD)s, %(POD_ETD)s,
                    %(CARRIER_SCAC_CODE)s, %(VESSEL_IMO)s)
            """,
            fact_trip_df.to_dict("records")
        )
        print("   ✅ FACT_TRIP loaded.")

        # =====================================================
        # DIM_CARRIER – insert only NEW SCAC codes
        # =====================================================
        print("🔄 Loading DIM_CARRIER...")
        
        carrier_df = pd.concat([
            peta_df[["CARRIER_SCAC_CODE", "CARRIER_NAME"]]
                .rename(columns={"CARRIER_SCAC_CODE": "SCAC", "CARRIER_NAME": "NAME"}),
            peta_df[["MASTER_CARRIER_SCAC_CODE", "MASTER_CARRIER_NAME"]]
                .rename(columns={"MASTER_CARRIER_SCAC_CODE": "SCAC", "MASTER_CARRIER_NAME": "NAME"})
        ]).dropna().drop_duplicates()
        
        # Check existing
        try:
            existing_carriers = pd.read_sql(
                "SELECT CRR_SCAC_CD FROM HACAKATHON.DT_INGESTION.DIM_CARRIER",
                conn
            )
            existing_scacs = set(existing_carriers["CRR_SCAC_CD"].tolist()) if not existing_carriers.empty else set()
        except Exception:
            existing_scacs = set()

        new_carriers = carrier_df[
            ~carrier_df["SCAC"].isin(existing_scacs)
        ].copy()

        if not new_carriers.empty:
            new_carriers["CRR_TYPE"] = "Ocean"
            new_carriers["IS_ACTIVE"] = 1
            new_carriers["SERVICE_START_DATE"] = "2024-01-01" # Fixed date typo from '210' to valid date
            
            new_carriers = new_carriers.where(pd.notnull(new_carriers), None)

            cs.executemany(
                """
                INSERT INTO HACAKATHON.DT_INGESTION.DIM_CARRIER
                (CRR_SCAC_CD, CRR_NAME, CRR_TYPE, IS_ACTIVE, SERVICE_START_DATE)
                VALUES (%(SCAC)s, %(NAME)s, %(CRR_TYPE)s, %(IS_ACTIVE)s, %(SERVICE_START_DATE)s)
                """,
                new_carriers.to_dict("records")
            )
            print(f"   ✅ DIM_CARRIER: {len(new_carriers)} new rows.")
        else:
            print("   ℹ️ DIM_CARRIER: No new rows.")

        # =====================================================
        # DIM_VEHICLE – insert only NEW IMO
        # =====================================================
                print("🔄 Loading DIM_VEHICLE...")

        # ------------------------------------------------------------------
        # Step 1: Prepare source vessel dataframe
        # ------------------------------------------------------------------
        vehicle_df = peta_df[[
            "VESSEL_IMO",
            "VESSEL_NAME",
            "VESSEL_MMSI",
            "VESSEL_CALL_SIGN"
        ]].dropna(subset=["VESSEL_IMO"]).drop_duplicates()

        # ------------------------------------------------------------------
        # Step 2: Read existing vehicle keys from DIM_VEHICLE
        # ------------------------------------------------------------------
        try:
            dim_vehicle_df = pd.read_sql(
                "SELECT VEHICLE_NUMBER FROM HACAKATHON.DT_INGESTION.DIM_VEHICLE",
                conn
            )
        except Exception:
            dim_vehicle_df = pd.DataFrame(columns=["VEHICLE_NUMBER"])

        # ------------------------------------------------------------------
        # Step 3: LEFT JOIN to identify only new vessels
        # ------------------------------------------------------------------
        new_vehicles = (
            vehicle_df
                .merge(
                    dim_vehicle_df,
                    how="left",
                    left_on="VESSEL_IMO",
                    right_on="VEHICLE_NUMBER",
                    indicator=True
                )
                .query("_merge == 'left_only'")
                .drop(columns=["VEHICLE_NUMBER", "_merge"])
        )

        # ------------------------------------------------------------------
        # Step 4: Insert new vessels into DIM_VEHICLE
        # ------------------------------------------------------------------
        if not new_vehicles.empty:
            # Convert NaN → None for database inserts
            new_vehicles = new_vehicles.where(pd.notnull(new_vehicles), None)

            cs.executemany(
                """
                INSERT INTO HACAKATHON.DT_INGESTION.DIM_VEHICLE
                (VEHICLE_NUMBER, VEHICLE_NM, VEHICLE_SECONDARY_NUMBER, VEHICLE_CALL_SIGN)
                VALUES (%(VESSEL_IMO)s, %(VESSEL_NAME)s, %(VESSEL_MMSI)s, %(VESSEL_CALL_SIGN)s)
                """,
                new_vehicles.to_dict("records")
            )

            print(f"   ✅ DIM_VEHICLE: {len(new_vehicles)} new rows.")
        else:
            print("   ℹ️ DIM_VEHICLE: No new rows.")

        # =====================================================
        # DIM_PORT – from ARR + DEP (insert only NEW ports)
        # =====================================================
        print("🔄 Loading DIM_PORT...")
        ports_df = pd.concat([
            peta_df["PORT_OF_ARR"],
            peta_df["PORT_OF_DEP"]
        ]).dropna().drop_duplicates().to_frame("CODE")

        try:
            existing_ports = pd.read_sql(
                "SELECT CODE FROM HACAKATHON.DT_INGESTION.DIM_PORT",
                conn
            )
            existing_codes = set(existing_ports["CODE"].tolist()) if not existing_ports.empty else set()
        except:
            existing_codes = set()

        new_ports = ports_df[
            ~ports_df["CODE"].isin(existing_codes)
        ].copy()

        if not new_ports.empty:
            new_ports["SOURCE_NAME"] = "PETA"
            new_ports["NAME"] = None
            new_ports["LATITUDE"] = None
            new_ports["LONGITUDE"] = None
            new_ports["COUNTRY_CODE"] = None
            new_ports["ZONE_CODE"] = None
            
            new_ports = new_ports.where(pd.notnull(new_ports), None)

            cs.executemany(
                """
                INSERT INTO HACAKATHON.DT_INGESTION.DIM_PORT
                (CODE, NAME, LATITUDE, LONGITUDE, COUNTRY_CODE, ZONE_CODE, SOURCE_NAME)
                VALUES (%(CODE)s, %(NAME)s, %(LATITUDE)s, %(LONGITUDE)s,
                        %(COUNTRY_CODE)s, %(ZONE_CODE)s, %(SOURCE_NAME)s)
                """,
                new_ports.to_dict("records")
            )
            print(f"   ✅ DIM_PORT: {len(new_ports)} new rows.")
        else:
             print("   ℹ️ DIM_PORT: No new rows.")

        # =====================================================
        # TRUNCATE PETA
        # =====================================================
        print("🗑️ Truncating Staging Table (DT_PREP.PETA)...")
        cs.execute("TRUNCATE TABLE HACAKATHON.DT_PREP.PETA")
        print("   ✅ Truncate Complete.")
        
        return {"status": "success", "message": "Transformation & Load Complete"}

    except Exception as e:
        print(f"❌ Transformation Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

    finally:
        cs.close()
        conn.close()

if __name__ == "__main__":
    run_transformation()
