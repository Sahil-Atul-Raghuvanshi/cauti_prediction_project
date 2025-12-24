"""
Preprocessing utilities for CAUTI prediction pipeline.
Replicates the preprocessing steps from bronze_to_silver notebooks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re
from pathlib import Path

# Cache for feature order
_feature_order = None

def get_feature_order():
    """Get the expected feature order from the trained model."""
    global _feature_order
    
    if _feature_order is not None:
        return _feature_order
    
    # Get project root (go up from streamlit_app to project root)
    # This file is in streamlit_app/, so go up one level
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Try to load from silver dataset to get feature order
    silver_path = PROJECT_ROOT / "data" / "silver" / "silver_dataset.csv"
    if silver_path.exists():
        try:
            df_silver = pd.read_csv(str(silver_path), nrows=1)
            # Remove leakage features and target
            leakage_features = ['subject_id', 'hadm_id']
            features = [col for col in df_silver.columns if col not in leakage_features + ['y']]
            _feature_order = features
            return features
        except Exception as e:
            pass
    
    # Fallback: return None and let the app handle it
    return None


def apply_outlier_handling(df):
    """
    Apply outlier handling based on notebook 2_b2s_outlier_analysis.ipynb
    """
    df = df.copy()
    
    # BMI: clip to 10-100
    if 'BMI' in df.columns:
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
        df.loc[(df['BMI'] < 10) | (df['BMI'] > 100) | (~np.isfinite(df['BMI'])), 'BMI'] = np.nan
    
    # anchor_age: clip to 0-120
    if 'anchor_age' in df.columns:
        df['anchor_age'] = pd.to_numeric(df['anchor_age'], errors='coerce')
        df.loc[(df['anchor_age'] < 0) | (df['anchor_age'] > 120), 'anchor_age'] = np.nan
    
    # charlson_score: clip to 0-35
    if 'charlson_score' in df.columns:
        df['charlson_score'] = pd.to_numeric(df['charlson_score'], errors='coerce')
        df.loc[(df['charlson_score'] < 0) | (df['charlson_score'] > 35), 'charlson_score'] = np.nan
    
    # length_of_stay: clip negative to 0, cap at 1000
    if 'length_of_stay' in df.columns:
        df['length_of_stay'] = pd.to_numeric(df['length_of_stay'], errors='coerce')
        df.loc[df['length_of_stay'] < 0, 'length_of_stay'] = 0
        df.loc[df['length_of_stay'] > 1000, 'length_of_stay'] = 1000
    
    # urinalysis_wbc: clip to 0-500
    if 'urinalysis_wbc' in df.columns:
        df['urinalysis_wbc'] = pd.to_numeric(df['urinalysis_wbc'], errors='coerce')
        df.loc[(df['urinalysis_wbc'] < 0) | (df['urinalysis_wbc'] > 500) | (~np.isfinite(df['urinalysis_wbc'])), 'urinalysis_wbc'] = np.nan
    
    # urinalysis_rbc: clip to 0-500
    if 'urinalysis_rbc' in df.columns:
        df['urinalysis_rbc'] = pd.to_numeric(df['urinalysis_rbc'], errors='coerce')
        df.loc[(df['urinalysis_rbc'] < 0) | (df['urinalysis_rbc'] > 500) | (~np.isfinite(df['urinalysis_rbc'])), 'urinalysis_rbc'] = np.nan
    
    # blood_wbc: clip to 0-200
    if 'blood_wbc' in df.columns:
        df['blood_wbc'] = pd.to_numeric(df['blood_wbc'], errors='coerce')
        df.loc[(df['blood_wbc'] <= 0) | (df['blood_wbc'] > 200) | (~np.isfinite(df['blood_wbc'])), 'blood_wbc'] = np.nan
    
    # creatinine: clip to 0-20
    if 'creatinine' in df.columns:
        df['creatinine'] = pd.to_numeric(df['creatinine'], errors='coerce')
        df.loc[(df['creatinine'] < 0) | (df['creatinine'] > 20) | (~np.isfinite(df['creatinine'])), 'creatinine'] = np.nan
    
    # blood_crp: clip to 0-300
    if 'blood_crp' in df.columns:
        df['blood_crp'] = pd.to_numeric(df['blood_crp'], errors='coerce')
        df.loc[(df['blood_crp'] < 0) | (df['blood_crp'] > 300) | (~np.isfinite(df['blood_crp'])), 'blood_crp'] = np.nan
    
    # temperature: clip to 80-108
    if 'temperature' in df.columns:
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df.loc[df['temperature'] < 80, 'temperature'] = np.nan
        df.loc[df['temperature'] > 108, 'temperature'] = 108
    
    # heart_rate: clip to 40-220
    if 'heart_rate' in df.columns:
        df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')
        df['heart_rate'] = df['heart_rate'].clip(lower=40, upper=220)
    
    # resp_rate: clip to 5-40
    if 'resp_rate' in df.columns:
        df['resp_rate'] = pd.to_numeric(df['resp_rate'], errors='coerce')
        df['resp_rate'] = df['resp_rate'].clip(lower=5, upper=40)
    
    # o2sat: clip to 70-100
    if 'o2sat' in df.columns:
        df['o2sat'] = pd.to_numeric(df['o2sat'], errors='coerce')
        df['o2sat'] = df['o2sat'].clip(lower=70, upper=100)
    
    # BP_systolic: clip to 60-250
    if 'BP_systolic' in df.columns:
        df['BP_systolic'] = pd.to_numeric(df['BP_systolic'], errors='coerce')
        df['BP_systolic'] = df['BP_systolic'].clip(lower=60, upper=250)
    
    # BP_diastolic: clip to 40-130
    if 'BP_diastolic' in df.columns:
        df['BP_diastolic'] = pd.to_numeric(df['BP_diastolic'], errors='coerce')
        df['BP_diastolic'] = df['BP_diastolic'].clip(lower=40, upper=130)
    
    return df


def apply_missing_value_handling(df):
    """
    Apply missing value handling based on notebook 3_b2s_handle_missing_values.ipynb
    For a single row, we'll use median/mean values from training data or simple defaults
    """
    df = df.copy()
    
    # For single row prediction, we'll use reasonable defaults
    # In production, you might want to load training medians
    
    # BMI: use median (around 25-27 is typical)
    if 'BMI' in df.columns:
        df['BMI'] = df['BMI'].fillna(26.0)
    
    # urinalysis_wbc: median around 2-5
    if 'urinalysis_wbc' in df.columns:
        df['urinalysis_wbc'] = df['urinalysis_wbc'].fillna(3.0)
    
    # urinalysis_rbc: median around 1-3
    if 'urinalysis_rbc' in df.columns:
        df['urinalysis_rbc'] = df['urinalysis_rbc'].fillna(2.0)
    
    # blood_wbc: normal range 4-11
    if 'blood_wbc' in df.columns:
        df['blood_wbc'] = df['blood_wbc'].fillna(7.0)
    
    # creatinine: normal around 0.7-1.2
    if 'creatinine' in df.columns:
        df['creatinine'] = df['creatinine'].fillna(1.0)
    
    # temperature: normal 98.6
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].fillna(98.6)
    
    # heart_rate: normal 60-100
    if 'heart_rate' in df.columns:
        df['heart_rate'] = df['heart_rate'].fillna(72.0)
    
    # resp_rate: normal 12-20
    if 'resp_rate' in df.columns:
        df['resp_rate'] = df['resp_rate'].fillna(16.0)
    
    # o2sat: normal 95-100
    if 'o2sat' in df.columns:
        df['o2sat'] = df['o2sat'].fillna(98.0)
    
    # BP_systolic: normal 120
    if 'BP_systolic' in df.columns:
        df['BP_systolic'] = df['BP_systolic'].fillna(120.0)
    
    # BP_diastolic: normal 80
    if 'BP_diastolic' in df.columns:
        df['BP_diastolic'] = df['BP_diastolic'].fillna(80.0)
    
    # discharge_location: use mode (HOME)
    if 'discharge_location' in df.columns:
        df['discharge_location'] = df['discharge_location'].fillna('HOME')
    
    # urinalysis_nitrite: create flags
    if 'urinalysis_nitrite' in df.columns:
        nitrite_val = df['urinalysis_nitrite'].iloc[0] if len(df) > 0 else None
        if nitrite_val == "Not Tested" or pd.isna(nitrite_val):
            df['nitrite_tested'] = 0
            df['nitrite_positive'] = 0
        elif nitrite_val == "Positive":
            df['nitrite_tested'] = 1
            df['nitrite_positive'] = 1
        else:  # Negative
            df['nitrite_tested'] = 1
            df['nitrite_positive'] = 0
        df = df.drop(columns=['urinalysis_nitrite'], errors='ignore')
    
    # Drop columns that are removed in preprocessing
    df = df.drop(columns=['final_removal_date', 'final_insertion_date', 'urine_bacteria'], errors='ignore')
    
    # Create measurement flags
    if 'blood_crp' in df.columns:
        df['blood_crp_measured'] = df['blood_crp'].notna().astype(int)
        df = df.drop(columns=['blood_crp'], errors='ignore')
    else:
        df['blood_crp_measured'] = 0
    
    if 'cfu_count' in df.columns:
        df['cfu_count_measured'] = df['cfu_count'].notna().astype(int)
        df = df.drop(columns=['cfu_count'], errors='ignore')
    else:
        df['cfu_count_measured'] = 0
    
    if 'urine_output_ml' in df.columns:
        df['urine_output_measured'] = df['urine_output_ml'].notna().astype(int)
        df = df.drop(columns=['urine_output_ml'], errors='ignore')
    else:
        df['urine_output_measured'] = 0
    
    if 'catheter_duration_days' in df.columns:
        df['catheter_duration_measured'] = df['catheter_duration_days'].notna().astype(int)
        df = df.drop(columns=['catheter_duration_days'], errors='ignore')
    else:
        df['catheter_duration_measured'] = 0
    
    # catheter_size handling
    # Priority: Use catheter_size_fr directly if it exists (from bronze data)
    # Otherwise, extract from catheter_size string
    # Note: -1 means no catheter size present in raw source (missing value)
    if 'catheter_size_fr' in df.columns:
        # Already have catheter_size_fr - use it directly
        df['catheter_size_fr'] = pd.to_numeric(df['catheter_size_fr'], errors='coerce')
        df['catheter_size_fr'] = df['catheter_size_fr'].fillna(-1)
        # Clip only non-missing values (6-24), preserve -1 (missing indicator)
        mask_not_missing = df['catheter_size_fr'] != -1
        df.loc[mask_not_missing, 'catheter_size_fr'] = df.loc[mask_not_missing, 'catheter_size_fr'].clip(lower=6, upper=24)
        df['catheter_size_measured'] = (df['catheter_size_fr'] != -1).astype(int)
        # Drop catheter_size string if it exists (we're using catheter_size_fr directly)
        df = df.drop(columns=['catheter_size'], errors='ignore')
    elif 'catheter_size' in df.columns:
        # Extract from catheter_size string
        df['catheter_size'] = df['catheter_size'].fillna('Unknown')
        df['catheter_size_known'] = (df['catheter_size'] != 'Unknown').astype(int)
        
        # Extract French size
        def extract_french_size(x):
            if pd.isna(x) or x == "Unknown" or x == "":
                return np.nan
            m = re.search(r'(\d+)', str(x))
            return float(m.group(1)) if m else np.nan
        
        df['catheter_size_fr'] = df['catheter_size'].apply(extract_french_size)
        df['catheter_size_fr'] = df['catheter_size_fr'].clip(lower=6, upper=24)
        df['catheter_size_measured'] = df['catheter_size_fr'].notna().astype(int)
        df = df.drop(columns=['catheter_size'], errors='ignore')
    else:
        df['catheter_size_fr'] = -1
        df['catheter_size_measured'] = 0
    
    # Fill remaining NaNs with 0 for binary/categorical flags
    df = df.fillna(0)
    
    return df


def apply_binary_encoding(df):
    """
    Apply binary encoding based on notebook 4_b2s_binary_encoding.ipynb
    """
    df = df.copy()
    
    # Boolean columns to convert to int
    bool_cols = [
        'catheter_present', 'diabetes', 'cancer', 'chronic_kidney_disease',
        'neurogenic_bladder', 'recurrent_uti', 'spinal_cord_injury',
        'neurological_disorder', 'benign_prostatic_hyperplasia',
        'surgical_admission', 'recent_urologic_abdominal_surgery',
        'icu_admission', 'urinary_obstruction_present',
        'improper_drainage_position', 'catheter_insertion', 'catheter_removal',
        'catheter_removal_replacement', 'multiple_invasive_devices',
        'catheter_indication', 'catheter_care', 'closed_system',
        'securement_device', 'antibiotics_per_admission', 'recent_antibiotic_use',
        'steroids_per_admission', 'immunosuppressives_per_admission',
        'ppi_per_admission', 'procalcitonin_measured', 'oliguria',
        'urine_culture_performed', 'gram_negative_organisms_present',
        'gram_positive_organisms_present', 'fungi_present',
        'blood_culture_performed', 'other_uti_present', 'has_cauti_history',
        'pain_documented', 'ventilator_used'
    ]
    
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df


def apply_one_hot_encoding(df):
    """
    Apply one-hot encoding based on notebook 5_b2s_one_hot_encoding.ipynb
    """
    df = df.copy()
    
    # Drop columns that are removed
    df = df.drop(columns=['catheter_size_known', 'age_group', 'catheter_size_from_notes', 
                          'race', 'admittime', 'dischtime'], errors='ignore')
    
    # Simple categorical columns - all possible values from training
    simple_cat_cols = ['admission_type', 'admission_location', 'discharge_location', 'mobility_status']
    
    # Define all possible values for each categorical column
    cat_values = {
        'admission_type': ['AMBULATORY OBSERVATION', 'DIRECT EMER', 'DIRECT OBSERVATION',
                          'ELECTIVE', 'EU OBSERVATION', 'EW EMER', 'OBSERVATION ADMIT',
                          'SURGICAL SAME DAY ADMISSION', 'URGENT'],
        'admission_location': ['AMBULATORY SURGERY TRANSFER', 'CLINIC REFERRAL', 'EMERGENCY ROOM',
                              'INFORMATION NOT AVAILABLE', 'INTERNAL TRANSFER TO OR FROM PSYCH',
                              'PACU', 'PHYSICIAN REFERRAL', 'PROCEDURE SITE',
                              'TRANSFER FROM HOSPITAL', 'TRANSFER FROM SKILLED NURSING FACILITY',
                              'WALKINSELF REFERRAL'],
        'discharge_location': ['ACUTE HOSPITAL', 'AGAINST ADVICE', 'DEAD/EXPIRED', 'HOME',
                              'HOME HEALTH CARE', 'HOSPICE', 'LONG TERM CARE HOSPITAL',
                              'REHAB', 'SKILLED NURSING FACILITY', 'UNKNOWN'],
        'mobility_status': ['Assisted Ambulatory', 'Bedbound', 'Independent Ambulatory',
                           'Non-ambulatory / Paralysis', 'Out of Bed with Assistance',
                           'Unknown', 'Wheelchair']
    }
    
    for col in simple_cat_cols:
        if col in df.columns:
            possible_values = cat_values.get(col, [])
            current_val = str(df[col].iloc[0]) if len(df) > 0 and not pd.isna(df[col].iloc[0]) else None
            
            # Create dummy columns for all possible values
            for val in possible_values:
                col_name = f"{col.lower()}_{val.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
                df[col_name] = 0  # Initialize to 0
                if current_val and current_val.upper() == val.upper():
                    df[col_name] = 1
            
            df = df.drop(columns=[col], errors='ignore')
        else:
            # Column not present, create all dummy columns with 0
            possible_values = cat_values.get(col, [])
            for val in possible_values:
                col_name = f"{col.lower()}_{val.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
                df[col_name] = 0
    
    # Multi-label: catheter_type
    possible_types = ['Foley', 'Straight', 'Suprapubic', 'Ureteral', 'Nephrostomy', 'Unknown']
    for cat_type in possible_types:
        col_name = f"catheter_type_{cat_type.lower()}"
        if 'catheter_type' in df.columns and len(df) > 0:
            cat_val = df['catheter_type'].iloc[0]
            if isinstance(cat_val, list):
                df[col_name] = int(cat_type in cat_val)
            else:
                df[col_name] = 0
        else:
            df[col_name] = 0
    
    if 'catheter_type' in df.columns:
        df = df.drop(columns=['catheter_type'], errors='ignore')
    
    # Multi-label: other_uti
    possible_uti = ['UTI Unspecified', 'Cystitis', 'Pyelonephritis', 'Urethritis']
    for uti_type in possible_uti:
        col_name = f"other_uti_{uti_type.lower().replace(' ', '_')}"
        if 'other_uti' in df.columns and len(df) > 0:
            uti_val = df['other_uti'].iloc[0]
            if isinstance(uti_val, list):
                df[col_name] = int(uti_type in uti_val)
            else:
                df[col_name] = 0
        else:
            df[col_name] = 0
    
    if 'other_uti' in df.columns:
        df = df.drop(columns=['other_uti'], errors='ignore')
    
    # Gender encoding
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.upper().map({
            'M': 1, 'MALE': 1, 'F': 0, 'FEMALE': 0
        }).fillna(0).astype(int)
    else:
        df['gender'] = 0
    
    # Make column names lowercase and snake_case
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^0-9a-zA-Z_]', '', regex=True)
    
    return df


def apply_normalization(df, scaler):
    """
    Apply normalization based on notebook 6_normalize_features.ipynb
    Note: The scaler expects features in a specific order. We'll handle this in the main app.
    """
    df = df.copy()
    
    # Handle catheter_size_fr
    if 'catheter_size_fr' in df.columns:
        df['catheter_size_fr'] = df['catheter_size_fr'].fillna(-1)
    
    # Note: Actual scaling will be done in the main app after feature alignment
    # This function just prepares the data
    
    return df

