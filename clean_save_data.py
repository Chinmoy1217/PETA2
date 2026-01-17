import pandas as pd

INPUT_FILE = 'Master_Training_Data_Augmented.csv'
OUTPUT_FILE = 'Cleaned_Training_Data_Augmented.csv'

def clean_and_save():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Filter Invalid Durations
    df = df[df['Actual_Duration_Hours'] > 0]
    
    # 2. Outlier Removal (IQR)
    print(f"Original Rows: {len(df)}")
    Q1 = df['Actual_Duration_Hours'].quantile(0.25)
    Q3 = df['Actual_Duration_Hours'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df_clean = df[(df['Actual_Duration_Hours'] >= lower) & (df['Actual_Duration_Hours'] <= upper)]
    
    removed = len(df) - len(df_clean)
    print(f"Removed {removed} outliers ({removed/len(df)*100:.2f}%)")
    print(f"Final Cleaned Rows: {len(df_clean)}")
    
    print(f"Saving to {OUTPUT_FILE}...")
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print("Success!")

if __name__ == "__main__":
    clean_and_save()
