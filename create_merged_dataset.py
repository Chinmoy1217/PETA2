import pandas as pd
import numpy as np

DIM_FILE = 'Dim_Trip_Augmented.xlsx'
FACT_FILE = 'Fact_Transpoart_Details.xlsx'
OUTPUT_FILE = 'Master_Training_Data_Augmented.csv' 
REFERENCE_CLEAN_FILE = 'Cleaned_Training_Data_Final_v6.csv'

def create_master():
    print("Loading datasets...")
    # Load Dim with excel
    dim = pd.read_excel(DIM_FILE)
        
    # Load Fact
    fact = pd.read_excel(FACT_FILE)
    
    print(f"Dim Shape: {dim.shape}")
    print(f"Fact Shape: {fact.shape}")
    
    # Preprocess Keys and Columns
    dim.columns = [c.strip() for c in dim.columns]
    
    # Fix duplicate/inconsistent columns in Dim
    # 'podcode' -> 'PodCode'
    if 'podcode' in dim.columns and 'PodCode' in dim.columns:
        print("Found both 'podcode' and 'PodCode', consolidating...")
        dim['PodCode'] = dim['PodCode'].fillna(dim['podcode'])
        dim = dim.drop(columns=['podcode'])
    elif 'podcode' in dim.columns:
        dim = dim.rename(columns={'podcode': 'PodCode'})
        
    # Standardize other potential case issues if needed
    
    # Ensure correct column names for join
    # Dim has 'ATD', 'ATA'
    # Fact has 'trip_ATD', 'trip_ATA'
    
    print("Normalizing Dates...")
    dim['ATD'] = pd.to_datetime(dim['ATD'], errors='coerce')
    dim['ATA'] = pd.to_datetime(dim['ATA'], errors='coerce')
    fact['trip_ATD'] = pd.to_datetime(fact['trip_ATD'], errors='coerce')
    fact['trip_ATA'] = pd.to_datetime(fact['trip_ATA'], errors='coerce')
    
    # Merge (Inner Join)
    print("Merging...")
    master = pd.merge(fact, dim, left_on=['trip_ATD', 'trip_ATA'], right_on=['ATD', 'ATA'], how='inner')
    
    # --- GEOGRAPHIC VALIDATION & CLEANING ---
    print("Applying Geographic Logic to remove Impossible Modes (e.g. Rail across Oceans)...")
    
    # Simple Continent Mapping (Top 30 Economies + Hubs)
    # EU: GB, FR, DE, IT, ES, NL, BE, PL, SE, NO, FI, DK, GR, PT, IE, HU, CZ, RO, CH, AT
    # ASIA: CN, IN, JP, KR, SG, TH, VN, MY, ID, PH, PK, BD, HK, TW, AE, SA, QA, TR
    # NA: US, CA, MX
    # SA: BR, AR, CL, CO, PE
    # AF: ZA, EG, NG, KE, GH, MA
    # OC: AU, NZ
    
    country_to_continent = {
        'GB': 'EU', 'FR': 'EU', 'DE': 'EU', 'IT': 'EU', 'ES': 'EU', 'NL': 'EU', 'BE': 'EU', 'PL': 'EU', 'SE': 'EU', 'NO': 'EU', 
        'FI': 'EU', 'DK': 'EU', 'GR': 'EU', 'PT': 'EU', 'IE': 'EU', 'HU': 'EU', 'CZ': 'EU', 'RO': 'EU', 'CH': 'EU', 'AT': 'EU',
        'RU': 'EU', 'TR': 'EU', # Transcontinental but treated as land-connected to EU for Rail/Road plausibility
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

    master['PolContinent'] = master['PolCode'].apply(get_continent)
    master['PodContinent'] = master['PodCode'].apply(get_continent)
    
    # Validation Rule: 
    # Inter-Continental (Different Continents) cannot use Road or Rail.
    # Exception: EU <-> AS could theoretically be Rail (Silk Road) but we'll flag it as suspicious for now unless explicitly valid.
    # We will enforce STRICT separation for now to solve the user's "Colombia to Portugal" example.
    
    def is_valid_geography(row):
        mode = str(row['ModeOfTransport']).upper().strip()
        src = row['PolContinent']
        dst = row['PodContinent']
        
        if src == 'UNKNOWN' or dst == 'UNKNOWN': return True # Give benefit of doubt
        
        if src != dst:
            # Different Continents
            if mode in ['ROAD', 'RAIL', 'TRUCK']:
                return False
        return True

    initial_len = len(master)
    master = master[master.apply(is_valid_geography, axis=1)]
    dropped_geo = initial_len - len(master)
    print(f"Dropped {dropped_geo} rows with 'Impossible Modes' (Inter-continental Road/Rail).")

    # Calculate Initial Duration (Days) for Upsampling reference
    print("Calculating initial duration...")
    master['Actual_Duration_Days'] = (master['trip_ATA'] - master['trip_ATD']).dt.total_seconds() / (24 * 3600)
    
    # --- END GEOGRAPHIC VALIDATION ---


    # --- UPSAMPLING & DATE OVERWRITE (Target: 2025) ---
    TARGET_ROWS = 600000 # Enough for demo/inference
    current_rows = len(master)
    
    print(f"Upsampling/Re-sampling to ~{TARGET_ROWS} rows for Year 2025...")
    
    # Sample with replacement to reach target
    if current_rows < TARGET_ROWS:
        needed = TARGET_ROWS - current_rows
        # Append samples
        synthetic = master.sample(n=needed, replace=True).copy()
        master = pd.concat([master, synthetic], axis=0)
    else:
        # Downsample if too big (optional, but keep it manageable)
        master = master.sample(n=TARGET_ROWS, replace=False).copy()
        
    # --- FORCE 2025 DATES ---
    print("Forcing all trips to Year 2025...")
    
    # Generate random start times in 2025 (Unix Seconds)
    start_ts = int(pd.Timestamp('2025-01-01').timestamp())
    end_ts = int(pd.Timestamp('2025-12-31').timestamp())
    
    random_ts = np.random.randint(start_ts, end_ts, size=len(master))
    master['trip_ATD'] = pd.to_datetime(random_ts, unit='s')
    
    # trip_ATA will be recalculated by Physics Engine later
    print("Date override complete.")

    # --- PHYSICS ENGINE & ID ENRICHMENT ---
    print("Initializing Physics Engine (Distance * Speed) and ID Generator...")
    
    # 1. Coordinates Database (Global Transhipment Hubs)
    # Expanded to include Major Ports, Air Cargo Hubs, and Rail Terminals
    city_coords = {
        # --- ASIA ---
        'CNSHA': (31.23, 121.47), 'CNPEK': (39.90, 116.40), 'CNCAN': (23.12, 113.26), 'CNHKG': (22.31, 114.16),
        'CNTAO': (36.06, 120.38), 'CNNBG': (29.86, 121.54), 'CNXMN': (24.47, 118.08), 'CNDAL': (38.91, 121.60),
        'CNCTU': (30.57, 104.06), 'CNCKG': (29.56, 106.55), 'CNCGO': (34.74, 113.62), # Inland Rail Hubs
        'SGSIN': (1.29, 103.85), 'MYPKG': (3.00, 101.40), 'THLCH': (13.08, 100.91), 'VNCLI': (10.58, 107.03),
        'JPTYO': (35.67, 139.76), 'JPKOB': (34.69, 135.19), 'JPYOK': (35.44, 139.63), 'JPNGO': (35.18, 136.90),
        'KRICN': (37.46, 126.44), 'KRPUS': (35.10, 129.04), # Busan Transhipment
        'TWKHH': (22.62, 120.28), 'TWKEL': (25.13, 121.73),
        'INBOM': (19.07, 72.87), 'INDEL': (28.61, 77.20), 'INMAA': (13.08, 80.27), 'INMUN': (22.75, 69.76), # Mundra
        'BDCGP': (22.35, 91.78), 'LKCMB': (6.92, 79.86), # Colombo (Major Sea Hub)

        # --- MIDDLE EAST ---
        'AEJXB': (25.04, 55.12), 'AEDXB': (25.25, 55.36), 'AEAUH': (24.45, 54.37), # Jebel Ali / Dubai
        'QADOH': (25.26, 51.56), 'SAJED': (21.48, 39.19), 'SADMM': (26.42, 50.08), 'OMSLL': (17.01, 54.09),

        # --- EUROPE ---
        'NLROT': (51.92, 4.47), 'BEANR': (51.21, 4.40), 'DEHAM': (53.55, 9.99), 'DEBRE': (53.07, 8.80),
        'DEFRA': (50.03, 8.55), 'DEDUI': (51.43, 6.76), # Duisburg (Rail Hub)
        'GBLHR': (51.47, -0.45), 'GBFEL': (51.95, 1.35), 'GBSOU': (50.90, -1.40), 'GBLGP': (51.50, 0.45),
        'FRPAR': (48.85, 2.35), 'FRLEH': (49.49, 0.10), 'FRMRS': (43.29, 5.36),
        'ESBCN': (41.38, 2.17), 'ESVLC': (39.46, -0.37), 'ESALG': (36.14, -5.45), # Algeciras
        'ITGOA': (44.40, 8.94), 'ITNAP': (40.85, 14.26), 'ITTRS': (45.64, 13.78),
        'PLGDN': (54.35, 18.66), 'BEZEE': (51.31, 3.22), 'TRIST': (41.00, 28.97),

        # --- NORTH AMERICA ---
        'USNYC': (40.71, -74.00), 'USLAX': (33.74, -118.28), 'USLGB': (33.77, -118.19),
        'USOAK': (37.80, -122.27), 'USSEA': (47.60, -122.33), 'USTAC': (47.25, -122.44),
        'USSAV': (32.08, -81.09), 'USCHS': (32.77, -79.93), 'USHOU': (29.76, -95.36),
        'USMIA': (25.76, -80.19), 'USMEM': (35.14, -90.04), # FedEx Superhub
        'USSDF': (38.25, -85.75), # UPS Superhub
        'USCHI': (41.87, -87.62), 'USKCY': (39.09, -94.57), # Rail Hubs
        'USANC': (61.21, -149.90), # Anchorage (Air Cargo Crossroads)
        'CAVAN': (49.28, -123.12), 'CAPRR': (54.31, -130.32), 'CAMTR': (45.50, -73.56),

        # --- SOUTH AMERICA & AFRICA & OCEANIA ---
        'BRSSZ': (-23.96, -46.29), 'BRRIO': (-22.90, -43.17), 'COBUN': (3.88, -77.07),
        'PABLB': (8.95, -79.56), 'PAMIT': (9.37, -79.88), # Panama Canal
        'ZADUR': (-29.85, 31.02), 'ZACPT': (-33.92, 18.42), 'EGSOK': (29.96, 32.54), # Suez
        'MAMED': (35.79, -5.60), # Tanger Med
        'AUZYD': (-33.86, 151.20), 'AUMEL': (-37.81, 144.96), 'NZAKL': (-36.84, 174.76),
        
        # --- HUBS (Ensure presence) ---
        'EGSUEZ': (29.96, 32.54)
    }
    # Default for unknown
    
    def get_lat_lon(code):
        return city_coords.get(str(code)[:5], (0, 0)) # First 5 chars usually match city UN/LOCODE

    import math
    def haversine_km(coord1, coord2):
        R = 6371 # Earth radius km
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    # 2. Physics Constants (Avg Speed km/h + Processing Hours)
    # Air: 800km/h + 48h handling
    # Ocean: 30km/h + 120h Port handling
    # Road: 60km/h + 4h breaks
    # Rail: 50km/h + 12h switching
    tuning = {
        'AIR': {'speed': 800, 'processing': 48, 'circuity': 1.05},
        'OCEAN': {'speed': 35, 'processing': 168, 'circuity': 1.2}, # 7 days port buffer
        'ROAD': {'speed': 60, 'processing': 12, 'circuity': 1.3}, 
        'RAIL': {'speed': 50, 'processing': 24, 'circuity': 1.15},
        'TRUCK': {'speed': 60, 'processing': 12, 'circuity': 1.3}
    }

    # 3. ID Generator (Deterministic per trip but looks real)
    def generate_id(row):
        mode = str(row['ModeOfTransport']).upper().strip()
        seed = int(row['trip_id']) if pd.notna(row.get('trip_id')) else np.random.randint(10000,99999)
        
        if 'OCEAN' in mode or 'SEA' in mode:
            # IMO Format: IMO + 7 digits
            return f"IMO {9000000 + (seed % 999999)}"
        elif 'AIR' in mode:
            # Flight: XX + 4 digits
            carriers = ['BA', 'LH', 'CX', 'UA', 'SQ', 'EK', 'AF']
            return f"{carriers[seed % len(carriers)]} {100 + (seed % 8000)}"
        elif 'RAIL' in mode:
             return f"R-{seed:08d}"
        else:
             # Road: Plate format
             return f"TRK-{seed % 99}-{text_code(seed)}"

    def text_code(seed):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"{chars[seed % 26]}{chars[(seed // 26) % 26]}"

    print("Calculating Physics-Based Durations & IDs...")
    
    # We'll use a vectorized approach or fast apply where possible, 
    # but for 1.2M rows apply might be slow. Optimization:
    # Pre-calc distances for unique routes.
    
    unique_routes = master[['PolCode', 'PodCode']].drop_duplicates()
    route_dist_map = {}
    
    for _, row in unique_routes.iterrows():
        o = row['PolCode']
        d = row['PodCode']
        c1 = city_coords.get(str(o), None)
        c2 = city_coords.get(str(d), None)
        
        if c1 and c2:
            dist = haversine_km(c1, c2)
        else:
            # Fallback estimation based on "Continent" logic already present?
            # Or just random realistic if unknown (to avoid 0 distance)
            dist = np.random.uniform(2000, 10000) # Fallback long haul
            
        route_dist_map[(o, d)] = dist

    # Transshipment Hubs (Expanded Network)
    hubs = {
        'AS': ['SGSIN', 'CNHKG', 'KRPUS', 'MYPKG'], # Singapore, Hong Kong, Busan, Port Klang
        'EU': ['NLROT', 'BEANR', 'DEHAM', 'ESALG'], # Rotterdam, Antwerp, Hamburg, Algeciras
        'ME': ['AEDXB', 'EGSUEZ', 'SAJED', 'OMSLL'], # Dubai, Suez, Jeddah, Salalah
        'NA': ['USLAX', 'USNYC', 'PABLB', 'CAMTR'], # LA, NY, Panama, Montreal
    }
    
    # Route Logic: Region -> Region mapping
    # Key: (OriginRegion, DestRegion) -> [Possible Hub Regions]
    route_flow = {
        ('AS', 'EU'): ['ME'], # Asia -> Middle East -> Europe
        ('AS', 'NA'): ['AS', 'NA'], # Asia -> Trans-Pacific -> US West
        ('EU', 'AS'): ['ME'], # Europe -> Middle East -> Asia
        ('EU', 'NA'): ['EU'], # Europe -> US East
        ('NA', 'AS'): ['NA'], 
        ('NA', 'EU'): ['NA'],
        ('AS', 'AF'): ['AS', 'ME'],
    }

    def apply_physics(row):
        mode = str(row['ModeOfTransport']).upper().strip()
        o = row['PolCode']
        d = row['PodCode']
        
        dist = route_dist_map.get((o, d), 5000)
        params = tuning.get(mode, tuning['ROAD'])
        
        # Base Travel Time
        travel_time_hours = (dist * params['circuity']) / params['speed']
        total_duration = travel_time_hours + params['processing']
        
        # --- MULTI-STOP TRANSSHIPMENT LOGIC ---
        via_ports = []
        transshipment_delay = 0
        
        # Only for Ocean/Air Long Haul (Distance > 3000km)
        if mode in ['OCEAN', 'AIR'] and dist > 3000:
            c1 = get_continent(o)
            c2 = get_continent(d)
            
            # Chance of stops: 
            # 80% for very long ocean (>10000km)
            # 50% for medium ocean
            # 30% for air
            chance = 0.5
            if mode == 'OCEAN' and dist > 10000: chance = 0.8
            if mode == 'AIR': chance = 0.3
            
            if np.random.rand() < chance:
                # Determine how many stops (1 or 2)
                num_stops = 1
                if dist > 14000 and mode == 'OCEAN':
                    num_stops = 2 if np.random.rand() < 0.4 else 1
                
                # Pick Hubs based on Flow
                possible_hub_regions = route_flow.get((c1, c2), [c1, c2])
                
                for _ in range(num_stops):
                    # Pick a region
                    region = np.random.choice(possible_hub_regions)
                    # Pick a hub in that region
                    if region in hubs:
                        hub = np.random.choice(hubs[region])
                        # Don't pick origin or dest as hub
                        if hub != o and hub != d and hub not in via_ports:
                            via_ports.append(hub)
                            
                            # Add Dwell Time
                            if mode == 'OCEAN':
                                # 2 to 5 days per stop
                                delay = np.random.uniform(48, 120)
                            else:
                                # 6 to 24 hours
                                delay = np.random.uniform(6, 24)
                            
                            transshipment_delay += delay
                            
        # Add Dwell to Total
        total_duration += transshipment_delay
        
        # Add Standard Noise (Weather, Traffic) - +/- 10%
        noise = np.random.normal(1.0, 0.1) 
        final_hours = max(24, total_duration * noise) # Minimum 24h
        
        # Format via_ports
        via_port_str = "|".join(via_ports) if via_ports else "DIRECT"
        
        return pd.Series([final_hours, via_port_str])

    # Apply
    print("Applying Physics with Multi-Stop Logic...")
    physics_results = master.apply(apply_physics, axis=1)
    master['Calculated_Duration_Hours'] = physics_results[0]
    master['via_port'] = physics_results[1]
    
    # Generate Valid IDs
    master['Transport_Vehicle_ID'] = master.apply(generate_id, axis=1)

    # --- COORDINATE SYNC ---
    # Ensure the dataset reflects the EXACT coordinates used for physics
    def get_lat(code): return city_coords.get(str(code)[:5], (None, None))[0]
    def get_lon(code): return city_coords.get(str(code)[:5], (None, None))[1]
    
    # Only update if we have the coord (keep existing if not in our top-list)
    master['New_PolLat'] = master['PolCode'].apply(get_lat)
    master['New_PolLon'] = master['PolCode'].apply(get_lon)
    master['New_PodLat'] = master['PodCode'].apply(get_lat)
    master['New_PodLon'] = master['PodCode'].apply(get_lon)
    
    # Fill existing with new where available
    master['PolLatitude'] = master['New_PolLat'].combine_first(master['PolLatitude'])
    master['PolLongitude'] = master['New_PolLon'].combine_first(master['PolLongitude'])
    master['PodLatitude'] = master['New_PodLat'].combine_first(master['PodLatitude'])
    master['PodLongitude'] = master['New_PodLon'].combine_first(master['PodLongitude'])
    
    # Cleanup temps
    master.drop(columns=['New_PolLat', 'New_PolLon', 'New_PodLat', 'New_PodLon'], inplace=True)

    # Overwrite the Timestamp Logic with Physics Logic
    # trip_ATD is already logical random.
    # Set trip_ATA = trip_ATD + Calculated Duration
    master['Actual_Duration_Hours'] = master['Calculated_Duration_Hours']
    master['trip_ATA'] = master['trip_ATD'] + pd.to_timedelta(master['Actual_Duration_Hours'], unit='h')
    
    print("Physics Engine Application & Coordinate Sync Complete.")
    # --- END PHYSICS ENGINE ---
    
    # We need to apply this PER ROUTE to be consistent (Model detects route patterns)
    # Create a unique route key
    master['RouteKey'] = master['PolCode'].astype(str) + "_" + master['PodCode'].astype(str) + "_" + master['ModeOfTransport'].astype(str)
    
    # Generate a "Standard Duration" for each unique Route
    unique_routes = master['RouteKey'].unique()
    route_standards = {}
    
    print(f"Calibrating speeds for {len(unique_routes)} unique routes...")
    
    # Pre-calculate standards for efficiency
    for route in unique_routes:
        # Extract mode from route key (simplified)
        if 'AIR' in route.upper():
            base = np.random.uniform(72, 144) # 3-6 days
        elif 'OCEAN' in route.upper() or 'SEA' in route.upper():
            base = np.random.uniform(360, 600) # 15-25 days
        else:
            base = np.random.uniform(48, 120)
            
        route_standards[route] = base
        
    # Map back to dataframe
    base_duration_hours = master['RouteKey'].map(route_standards)
    
    # 3. Apply Mode-Specific Noise (Proportional)
    def get_noise_factor(mode):
        if str(mode).upper() == 'AIR':
            return 0.05 # +/- 5% variance (Air is reliable)
        elif str(mode).upper() == 'RAIL':
            return 0.10 
        else: # ROAD/Ocean
            return 0.15 # +/- 15% variance (Sea/Road varies more)
            
    noise_factors = master['ModeOfTransport'].apply(get_noise_factor)
    
    # Generate random variance
    random_variance_pct = np.random.uniform(-1, 1, size=len(master)) * noise_factors
    
    # Add operational delay noise (skewed positive - delays are more common than early arrivals)
    delay_noise = np.random.exponential(scale=2.0, size=len(master)) # Avg 2h delay, long tail
    
    final_duration_hours = base_duration_hours * (1 + random_variance_pct) + delay_noise
    
    # 4. Calculate ATA
    master['trip_ATA'] = master['trip_ATD'] + pd.to_timedelta(final_duration_hours, unit='h')
    
    # Calculate Final Duration in HOURS
    master['Actual_Duration_Hours'] = (master['trip_ATA'] - master['trip_ATD']).dt.total_seconds() / 3600
    
    # Filter valid
    master = master[master['Actual_Duration_Hours'] > 0]
    
    # --- Column Verification vs Cleaned_Training_Data_Final_v6.csv ---
    try:
        ref_df = pd.read_csv(REFERENCE_CLEAN_FILE, nrows=1)
        required_cols = [c for c in ref_df.columns if c in master.columns] # Check intersection
        # ideally we want ALL columns from ref to be in master, except maybe for unique IDs generated later
        # But user asked to HAVE all columns that are required.
        
        missing_cols = set(ref_df.columns) - set(master.columns)
        if missing_cols:
            print(f"WARNING: The following columns from v6 are missing in new dataset: {missing_cols}")
            # If crucial columns are missing, we might need to investigate. 
            # For now, we list them. 'trip_id' should be there if it was in Fact.
        else:
            print("Verified: All columns from v6 are present in the new dataset.")
            
    except Exception as e:
        print(f"Could not verify against v6 file: {e}")

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    master.to_csv(OUTPUT_FILE, index=False)
    print("Success! Created Master Dataset with Timestamps and Hours.")

if __name__ == "__main__":
    create_master()
