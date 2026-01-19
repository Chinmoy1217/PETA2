import json
import numpy as np
import os

def calculate_f1():
    path = os.path.join(r"c:\Users\Administrator\.gemini\antigravity\PETA2\model", "plots.json")
    if not os.path.exists(path):
        print("plots.json not found")
        return

    with open(path, "r") as f:
        data = json.load(f)
    
    scatter = data.get('scatter_data', [])
    if not scatter:
        print("No scatter data")
        return

    actuals = np.array([p['Actual'] for p in scatter])
    preds = np.array([p['Pred'] for p in scatter])
    
    # Define "Late" as Actual - Pred > 1.0 days (Underestimated by > 1 day)
    # This assumes Actual/Pred are Durations.
    # Logic: If Actual (12) > Pred (10) + 1, then it's Late.
    
    # Evaluation: Did the Model PREDICT it would be Late?
    # This is tricky. The model predicts a Duration `Pred`. 
    # If we don't have a "Scheduled Duration", we can't know if the MODEL thought it was late relative to schedule.
    # But usually "Late" means "Late vs ETA".
    # Here, `Pred` IS the ETA (Predicted Duration).
    # So the model *always* predicts "This is the ETA".
    # It doesn't predict "I think I am underestimating".
    
    # Alternate Interpretation:
    # F1 Score of "Reliability".
    # Positive Class: "Accurate Prediction" (|Diff| <= 1).
    # TP: |Diff| <= 1.
    # FP: 0 (The model always output a number).
    # FN: |Diff| > 1.
    # Precision = TP / (TP + FP) = TP/TP = 1.0
    # Recall = TP / (TP + FN) = Accuracy.
    # F1 = 2 * 1 * Acc / (1 + Acc). 
    
    # This feels weird. 
    
    # Let's assume the user considers "Late" as "Actual > 2 days" (Long duration?). No.
    
    # Let's go with the SKLEARN Regression to Classification conversion.
    # Threshold = 5% error?
    pass

    # Let's calculate R2 and MAE first, as they are standard.
    diffs = actuals - preds
    mae = np.mean(np.abs(diffs))
    
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    
    # F1 Score using Binary Classification on "Late vs On-Time"
    # Let's assume we have a 'Scheduled' duration. But we don't.
    # Let's USE 'Avg Duration' as proxy for Scheduled? No.
    
    # What if "Late" means "Prediction Error > 1 day"?
    # i.e. Model fails to predict correctly.
    # That's just (1 - Accuracy).
    
    # Wait, look at the code main.py:
    # 'late_shipments_count': int(np.sum(diffs > 1))
    # This metric tracks "How many shipments arrived > 1 day LATER than predicted".
    
    # Ideally, for F1, we compare Model_Class vs Actual_Class.
    # But we don't have Model_Class.
    
    print(f"F1_Score: N/A (Regression Model). MAE: {mae:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    calculate_f1()
