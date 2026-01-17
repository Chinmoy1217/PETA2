import pandas as pd
import numpy as np

try:
    try:
        df = pd.read_csv('Dim_Trip(in).csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('Dim_Trip(in).csv', encoding='latin1')

    # Basic cleaning
    df = df.dropna(subset=['ATD', 'ATA'])
    df['Actual_Duration_Days'] = (pd.to_datetime(df['ATA']) - pd.to_datetime(df['ATD'])).dt.total_seconds() / (24 * 3600)
    df = df[df['Actual_Duration_Days'] >= 0]
    
    print(f"Total clean records: {len(df)}")
    
    # Check Target Distribution
    print("\nTarget (Duration) Stats:")
    print(df['Actual_Duration_Days'].describe())
    
    # Check Route Consistency
    # A 'Route' can be defined by POL -> POD + Mode
    df['Route'] = df['PolCode'].astype(str) + " -> " + df['PodCode'].astype(str) + " (" + df['ModeOfTransport'].astype(str) + ")"
    
    route_stats = df.groupby('Route')['Actual_Duration_Days'].agg(['count', 'mean', 'std', 'min', 'max'])
    
    # Filter for routes with significant volume
    frequent_routes = route_stats[route_stats['count'] > 50]
    
    print(f"\nNumber of unique routes: {len(route_stats)}")
    print(f"Number of routes with > 50 trips: {len(frequent_routes)}")
    
    print("\nVariance in Top 10 Frequent Routes (High std dev means unpredictable):")
    print(frequent_routes.sort_values('count', ascending=False).head(10)[['count', 'mean', 'std']])
    
    # Calculate overall "Noise"
    # If std dev per route is high compared to the mean, we need more features (weather, carrier delays)
    avg_cv = (frequent_routes['std'] / frequent_routes['mean']).mean()
    print(f"\nAverage Coefficient of Variation (Noise/Signal ratio) on frequent routes: {avg_cv:.2f}")
    if avg_cv > 0.3:
        print(">> High variance within same routes. More features needed (not just more rows).")
    else:
        print(">> Routes are consistent. More data rows might help models generalize.")

except Exception as e:
    print(f"Error: {e}")
