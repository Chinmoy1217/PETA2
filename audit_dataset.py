import pandas as pd
import numpy as np

FILE_PATH = 'Cleaned_Training_Data_Augmented.csv'

# Continent Map
country_to_continent = {
    'GB': 'EU', 'FR': 'EU', 'DE': 'EU', 'IT': 'EU', 'ES': 'EU', 'NL': 'EU', 'BE': 'EU', 'PL': 'EU', 'SE': 'EU', 'NO': 'EU', 
    'FI': 'EU', 'DK': 'EU', 'GR': 'EU', 'PT': 'EU', 'IE': 'EU', 'HU': 'EU', 'CZ': 'EU', 'RO': 'EU', 'CH': 'EU', 'AT': 'EU',
    'RU': 'EU', 'TR': 'EU', 
    'CN': 'AS', 'IN': 'AS', 'JP': 'AS', 'KR': 'AS', 'SG': 'AS', 'TH': 'AS', 'VN': 'AS', 'MY': 'AS', 'ID': 'AS', 
    'PH': 'AS', 'PK': 'AS', 'BD': 'AS', 'HK': 'AS', 'TW': 'AS', 'AE': 'AS', 'SA': 'AS', 'QA': 'AS', 
    'US': 'NA', 'CA': 'NA', 'MX': 'NA',
    'BR': 'SA', 'AR': 'SA', 'CL': 'SA', 'CO': 'SA', 'PE': 'SA',
    'ZA': 'AF', 'EG': 'AF', 'NG': 'AF', 'KE': 'AF', 'GH': 'AF', 'MA': 'AF',
    'AU': 'OC', 'NZ': 'OC'
}

def get_continent(code):
    if pd.isna(code) or len(str(code)) < 2: return 'UNKNOWN'
    country = str(code)[:2].upper()
    return country_to_continent.get(country, 'UNKNOWN')

print(f"Auditing {FILE_PATH}...")
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Total Rows: {len(df)}")
    
    # Add Continent Columns
    df['SrcCont'] = df['PolCode'].apply(get_continent)
    df['DstCont'] = df['PodCode'].apply(get_continent)
    df['ModeClean'] = df['ModeOfTransport'].str.upper().str.strip()
    
    df['ModeClean'] = df['ModeOfTransport'].str.upper().str.strip()
    
    # 1. Geographic Route Validation
    def is_invalid_route(row):
        if row['SrcCont'] != 'UNKNOWN' and row['DstCont'] != 'UNKNOWN':
            if row['SrcCont'] != row['DstCont']:
                if row['ModeClean'] in ['ROAD', 'RAIL', 'TRUCK']:
                    return True
        return False

    # 2. Coordinate Integrity Validation
    # Lat: -90 to 90, Lon: -180 to 180
    def is_invalid_coord(row):
        try:
            lat1, lon1 = float(row['PolLatitude']), float(row['PolLongitude'])
            lat2, lon2 = float(row['PodLatitude']), float(row['PodLongitude'])
            
            if not (-90 <= lat1 <= 90) or not (-180 <= lon1 <= 180): return True
            if not (-90 <= lat2 <= 90) or not (-180 <= lon2 <= 180): return True
            return False
        except:
            return True # Non-numeric

    print("Checking Lat/Lon Bounds...")
    coord_issues = df[df.apply(is_invalid_coord, axis=1)]
    print(f"Rows with Invalid Coordinates: {len(coord_issues)}")

    # 3. Physics Validation (Speed Check)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    print("Running Physics Engine Validation (Speed Checks)...")
    
    # Calculate Distances
    df['Dist_Ref'] = haversine(df['PolLatitude'], df['PolLongitude'], 
                               df['PodLatitude'], df['PodLongitude'])
    
    # Calculate Implied Speed (Km/h)
    # Avoid zero division
    df['Implied_Speed'] = df['Dist_Ref'] / df['Actual_Duration_Hours'].replace(0, 0.1)
    
    def is_invalid_speed(row):
        speed = row['Implied_Speed']
        mode = row['ModeClean']
        
        # AIR Validation
        if mode == 'AIR':
            # Air too slow (< 20 km/h) - accounts for 48h+ dwell on short trips
            # Supersonic > 1200 km/h
            if speed < 20 or speed > 1200: return True
            
        # OCEAN Validation
        elif mode == 'OCEAN':
            # Ocean too fast (> 80 km/h approx 43 knots)
            if speed > 80: return True 
            
        return False

    speed_issues = df[df.apply(is_invalid_speed, axis=1)]
    print(f"Rows with Physics Anomalies (Implied Speed out of bounds): {len(speed_issues)}")
    if len(speed_issues) > 0:
        print("Sample Anomalies:")
        print(speed_issues[['ModeOfTransport', 'Dist_Ref', 'Actual_Duration_Hours', 'Implied_Speed']].head())

    invalid_df = df[df.apply(is_invalid_route, axis=1)]
    count = len(invalid_df)
    
    print(f"Invalid Routes Found: {count}")
    
    total_issues = count + len(coord_issues) + len(speed_issues)
    
    if total_issues > 0:
        print("Cleaning dataset...")
        # Drop all types of errors
        valid_df = df[~df.apply(is_invalid_route, axis=1)]
        valid_df = valid_df[~valid_df.apply(is_invalid_coord, axis=1)]
        valid_df = valid_df[~valid_df.apply(is_invalid_speed, axis=1)]

        
        # Save back
        # Drop temp columns
        valid_df = valid_df.drop(columns=['SrcCont', 'DstCont', 'ModeClean', 'Dist_Ref', 'Implied_Speed'])
        valid_df.to_csv(FILE_PATH, index=False)
        print(f"Purged {total_issues} rows. Saved cleaned data.")
    else:
        print("Dataset is CLEAN. Physics & Geography Verified.")
        
except Exception as e:
    print(f"Audit failed: {e}")
