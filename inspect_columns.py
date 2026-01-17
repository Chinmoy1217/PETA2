import pandas as pd

DIM_FILE = 'Dim_Trip(in).csv'
FACT_FILE = 'Fact_Transpoart_Details.xlsx'

print("--- DIM COLUMNS ---")
try:
    df = pd.read_csv(DIM_FILE, encoding='utf-8')
except:
    df = pd.read_csv(DIM_FILE, encoding='latin1')
for c in df.columns:
    print(c)

print("\n--- FACT COLUMNS ---")
df = pd.read_excel(FACT_FILE)
for c in df.columns:
    print(c)
