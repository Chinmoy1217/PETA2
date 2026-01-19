"""
Update ATA (Actual Time of Arrival) for Shipments
Call this when shipments actually arrive to track accuracy
"""
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG
from datetime import datetime

def update_ata(prediction_id, actual_arrival_datetime):
    """
    Update a prediction with actual arrival time
    
    Args:
        prediction_id: UUID of the prediction
        actual_arrival_datetime: Actual arrival datetime (YYYY-MM-DD HH:MM:SS)
    """
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Parse actual arrival
        ata_dt = datetime.strptime(actual_arrival_datetime, "%Y-%m-%d %H:%M:%S")
        
        # Get original prediction
        cursor.execute("""
            SELECT DEPARTURE_DATE, PETA_DATETIME, PETA_HOURS 
            FROM DT_INGESTION.PETA_PREDICTION_RESULTS 
            WHERE PREDICTION_ID = %s
        """, (prediction_id,))
        
        result = cursor.fetchone()
        if not result:
            print(f"❌ Prediction {prediction_id} not found")
            return False
        
        depart_date, peta_dt, peta_hours = result
        
        # Calculate ATA metrics
        actual_transit = (ata_dt - depart_date).total_seconds() / 3600  # hours
        ata_days = actual_transit / 24
        
        # Calculate variance
        variance_hours = actual_transit - peta_hours
        variance_days = variance_hours / 24
        
        # Calculate accuracy (closer to 0 variance = higher accuracy)
        accuracy = max(0, 100 - abs(variance_hours / peta_hours * 100))
        
        # Update record
        cursor.execute("""
            UPDATE DT_INGESTION.PETA_PREDICTION_RESULTS
            SET 
                ATA_DATETIME = %s,
                ATA_DATE = %s,
                ATA_HOURS = %s,
                ATA_DAYS = %s,
                VARIANCE_HOURS = %s,
                VARIANCE_DAYS = %s,
                ACCURACY_PCT = %s,
                STATUS = 'COMPLETED',
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHERE PREDICTION_ID = %s
        """, (
            ata_dt,
            ata_dt.date(),
            actual_transit,
            ata_days,
            variance_hours,
            variance_days,
            accuracy,
            prediction_id
        ))
        
        conn.commit()
        
        print(f"✅ Updated ATA for prediction {prediction_id}")
        print(f"   PETA: {peta_hours:.1f} hours")
        print(f"   ATA:  {actual_transit:.1f} hours")
        print(f"   Variance: {variance_hours:+.1f} hours ({variance_days:+.1f} days)")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# Example usage
if __name__ == "__main__":
    # Example: Update with actual arrival
    # update_ata("your-prediction-id-here", "2024-06-17 14:30:00")
    
    print("ATA Update Tool")
    print("================")
    print()
    print("Usage:")
    print("  update_ata(prediction_id, actual_arrival_datetime)")
    print()
    print("Example:")
    print('  update_ata("abc-123-def", "2024-06-17 14:30:00")')
