import pandas as pd

class QualityCheck:
    @staticmethod
    def run_checks(df: pd.DataFrame) -> dict:
        """
        Validates the Training DataFrame.
        Returns a dict: {'passed': bool, 'reason': str, 'metrics': dict}
        """
        metrics = {}
        
        # 1. Volume Check
        row_count = len(df)
        metrics['row_count'] = row_count
        if row_count < 50:
            return {
                'passed': False, 
                'reason': f"Insufficient data volume. Found {row_count} rows, required 50.", 
                'metrics': metrics
            }
            
        # 2. Critical Columns Existence
        required_cols = ['PolCode', 'PodCode', 'Actual_Duration_Hours']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
             return {
                'passed': False, 
                'reason': f"Missing critical columns: {missing_cols}", 
                'metrics': metrics
            }
            
        # 3. Null Checks (Allow extensive nulls in non-critical, but features must be decent)
        # PolCode/PodCode shouldn't be null for > 5% of data
        for col in ['PolCode', 'PodCode']:
            null_pct = df[col].isnull().mean()
            metrics[f'{col}_null_pct'] = round(null_pct * 100, 2)
            if null_pct > 0.10: # 10% tolerance
                 return {
                    'passed': False, 
                    'reason': f"Column {col} has too many nulls ({round(null_pct*100, 2)}%)", 
                    'metrics': metrics
                }

        # 4. Logical Consistency (Negative Duration)
        # Actual_Duration_Hours should be Positive
        # We don't fail the whole batch, but we should flag if bad data ratio is high
        bad_duration = df[df['Actual_Duration_Hours'] <= 0]
        bad_pct = len(bad_duration) / row_count
        metrics['negative_duration_pct'] = round(bad_pct * 100, 2)
        
        if bad_pct > 0.20: # If > 20% of data has negative duration, something is wrong with source
             return {
                'passed': False, 
                'reason': f"Data Quality Error: {round(bad_pct*100, 2)}% of rows have negative/zero duration.", 
                'metrics': metrics
            }

        return {'passed': True, 'reason': "All checks passed", 'metrics': metrics}
