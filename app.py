# Suppress TensorFlow oneDNN warnings before importing TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
import re

# Page config
st.set_page_config(
    page_title="CAUTI Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 CAUTI Risk Prediction System")
st.markdown("---")

# Load scaler and model
@st.cache_resource
def load_scaler():
    scaler_path = Path("scripts/models/cauti_ann_loso_scaler.pkl")
    if scaler_path.exists():
        return joblib.load(scaler_path)
    else:
        st.error(f"Scaler file not found at {scaler_path}")
        return None

@st.cache_resource
def load_model():
    """
    Load the Keras model with handling for quantization_config errors.
    Tries .keras format first (recommended), then .h5, then .pkl.
    """
    def load_keras_model_safe(model_path):
        """Load Keras model with error handling for quantization_config issues."""
        try:
            # Try standard load first (with compile)
            try:
                return tf.keras.models.load_model(model_path)
            except:
                # If that fails, try without compile
                model = tf.keras.models.load_model(model_path, compile=False)
                # Compile if needed
                if not model._is_compiled:
                    model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['AUC', 'accuracy']
                    )
                return model
        except Exception as e:
            error_str = str(e)
            # If quantization_config error, clean the config file and reload
            if 'quantization_config' in error_str or 'Unrecognized keyword arguments' in error_str:
                try:
                    import json
                    import zipfile
                    import tempfile
                    
                    # .keras files are zip archives containing config.json and weights
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Extract the .keras file
                        with zipfile.ZipFile(model_path, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)
                        
                        # Read and clean config.json
                        config_path = Path(tmpdir) / 'config.json'
                        if not config_path.exists():
                            raise FileNotFoundError("config.json not found in model file")
                        
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        
                        # Recursively remove quantization_config from all layer configs
                        def remove_quantization_config(obj):
                            if isinstance(obj, dict):
                                # Remove quantization_config key
                                obj = {k: remove_quantization_config(v) for k, v in obj.items() if k != 'quantization_config'}
                            elif isinstance(obj, list):
                                obj = [remove_quantization_config(item) for item in obj]
                            return obj
                        
                        cleaned_config = remove_quantization_config(config_data)
                        
                        # Write cleaned config back
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(cleaned_config, f, indent=2)
                        
                        # Create temporary .keras file with cleaned config
                        temp_keras = Path(tmpdir) / 'cleaned_model.keras'
                        with zipfile.ZipFile(temp_keras, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                            for file_path in Path(tmpdir).rglob('*'):
                                if file_path.is_file() and file_path.name != 'cleaned_model.keras':
                                    arcname = file_path.relative_to(tmpdir)
                                    zip_ref.write(file_path, arcname)
                        
                        # Load from cleaned file (try with compile first, then without)
                        try:
                            model = tf.keras.models.load_model(temp_keras)
                        except:
                            model = tf.keras.models.load_model(temp_keras, compile=False)
                            # Recompile with original settings if needed
                            if not model._is_compiled:
                                model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['AUC', 'accuracy']
                                )
                        return model
                except Exception as e2:
                    st.warning(f"Failed to load with config cleaning: {str(e2)}")
                    # Fallback: try loading .h5 file if it exists
                    h5_path = model_path.with_suffix('.h5')
                    if h5_path.exists() and h5_path != model_path:
                        try:
                            return tf.keras.models.load_model(h5_path, compile=False)
                        except:
                            pass
                    raise e2
            else:
                raise e
    
    # Try different possible model file names and formats
    # Prioritize .keras format (newest, avoids quantization_config issues)
    model_paths = [
        Path("scripts/models/cauti_ann_loso_model.keras"),  # Newest Keras format (preferred)
        Path("scripts/models/cauti_ann_loso_model.h5"),  # Legacy H5 format
        Path("scripts/models/cauti_ann_loso_model.pkl"),  # Fallback to .pkl
        Path("scripts/models/model.pkl"),
        Path("scripts/models/ann_loso_model.pkl"),
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            try:
                if model_path.suffix in ['.keras', '.h5']:
                    # Load as TensorFlow/Keras model
                    return load_keras_model_safe(model_path)
                elif model_path.suffix == '.pkl':
                    # Try loading as joblib (could be Keras or sklearn model)
                    # Note: This may have deserialization issues with Keras models
                    model = joblib.load(model_path)
                    # Check if it's a Keras model (has predict method and is callable)
                    if hasattr(model, 'predict') and callable(getattr(model, 'predict', None)):
                        return model
                    else:
                        st.error(f"Model loaded from {model_path} but doesn't have a predict method.")
                        return None
            except Exception as e:
                st.warning(f"Error loading model from {model_path}: {str(e)}")
                continue
    
    # If no model found, show helpful message
    st.error("❌ **Model file not found!**")
    st.markdown("### 📝 Quick Instructions:")
    st.markdown("""
    1. Open your notebook: `scripts/models/ANN_LOSO.ipynb`
    2. After model training completes, add a **NEW CELL** with this code:
    """)
    
    st.code("""
import joblib
from pathlib import Path

model_dir = Path('scripts/models')
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / 'cauti_ann_loso_model.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")
    """, language='python')
    
    st.markdown("3. Run the cell to save the model")
    st.markdown("4. Refresh this page (the model will be automatically loaded)")
    
    st.info("💡 **Tip:** See `HOW_TO_SAVE_MODEL.md` for detailed instructions.")
    return None

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'uploaded_data_raw' not in st.session_state:
    st.session_state.uploaded_data_raw = None
if 'csv_is_preprocessed' not in st.session_state:
    st.session_state.csv_is_preprocessed = False

# Load resources
scaler = load_scaler()
model = load_model()

if scaler is None:
    st.stop()

# Import preprocessing functions
from preprocessing_utils import (
    apply_outlier_handling,
    apply_missing_value_handling,
    apply_binary_encoding,
    apply_one_hot_encoding,
    apply_normalization,
    get_feature_order
)

# Helper function to retrieve data from bronze_one_hot_encoded.csv by subject_id and hadm_id
@st.cache_data
def load_bronze_data():
    """Load bronze_one_hot_encoded.csv for data retrieval."""
    bronze_path = Path("silver/bronze_one_hot_encoded.csv")
    if bronze_path.exists():
        try:
            return pd.read_csv(bronze_path)
        except Exception as e:
            st.warning(f"Could not load bronze_one_hot_encoded.csv: {str(e)}")
            return None
    return None

def retrieve_bronze_data(csv_data):
    """
    Retrieve data from bronze_one_hot_encoded.csv by matching subject_id and hadm_id.
    Returns the matched row(s) from bronze data, or None if not found.
    """
    if csv_data is None or len(csv_data) == 0:
        return None
    
    # Get subject_id and hadm_id from uploaded CSV (first row)
    try:
        # Try to get IDs from first row
        row = csv_data.iloc[0]
        
        # Try different possible column names
        subject_id = None
        hadm_id = None
        
        for col in csv_data.columns:
            col_lower = col.lower()
            if col_lower == 'subject_id':
                subject_id = row[col]
            elif col_lower == 'hadm_id':
                hadm_id = row[col]
        
        if subject_id is None or hadm_id is None or pd.isna(subject_id) or pd.isna(hadm_id):
            return None
        
        # Load bronze data
        bronze_df = load_bronze_data()
        if bronze_df is None:
            return None
        
        # Match by subject_id and hadm_id
        # Convert to same type for matching
        bronze_df['subject_id'] = pd.to_numeric(bronze_df['subject_id'], errors='coerce')
        bronze_df['hadm_id'] = pd.to_numeric(bronze_df['hadm_id'], errors='coerce')
        subject_id = pd.to_numeric(subject_id, errors='coerce')
        hadm_id = pd.to_numeric(hadm_id, errors='coerce')
        
        # Find matching row
        match = bronze_df[
            (bronze_df['subject_id'] == subject_id) & 
            (bronze_df['hadm_id'] == hadm_id)
        ]
        
        if len(match) > 0:
            return match.iloc[[0]]  # Return as DataFrame with one row
        else:
            return None
            
    except Exception as e:
        st.warning(f"Error retrieving data from bronze_one_hot_encoded.csv: {str(e)}")
        return None

# Helper function to detect if CSV is already preprocessed
def is_preprocessed_csv(csv_data):
    """
    Detect if CSV data is already preprocessed (has one-hot encoded columns, normalized values, etc.)
    """
    if csv_data is None or len(csv_data) == 0:
        return False
    
    # Check for indicators of preprocessed data:
    # 1. Has one-hot encoded columns (e.g., admission_type_ambulatory_observation)
    one_hot_indicators = [
        'admission_type_', 'admission_location_', 'discharge_location_',
        'mobility_status_', 'catheter_type_', 'other_uti_'
    ]
    has_one_hot = any(any(ind in col.lower() for col in csv_data.columns) for ind in one_hot_indicators)
    
    # 2. Has processed columns (nitrite_tested instead of urinalysis_nitrite)
    has_processed_cols = any(col in csv_data.columns for col in ['nitrite_tested', 'nitrite_positive', 'blood_crp_measured'])
    
    # 3. Check if numeric values look normalized (negative values for age, etc.)
    has_normalized_values = False
    if 'anchor_age' in csv_data.columns or 'bmi' in csv_data.columns:
        sample_col = 'anchor_age' if 'anchor_age' in csv_data.columns else 'bmi'
        sample_val = csv_data[sample_col].iloc[0] if len(csv_data) > 0 else None
        if sample_val is not None and not pd.isna(sample_val):
            # Normalized values are typically in range -5 to 5, raw age would be 0-120, BMI 10-100
            if isinstance(sample_val, (int, float)) and -10 < sample_val < 10:
                has_normalized_values = True
    
    # If 2 or more indicators are present, consider it preprocessed
    indicators_count = sum([has_one_hot, has_processed_cols, has_normalized_values])
    return indicators_count >= 2

# Helper function to get value from uploaded CSV
def get_csv_value(csv_data, column_name, default_value=None):
    """Extract value from uploaded CSV data, handling case-insensitive column names."""
    if csv_data is None or len(csv_data) == 0:
        return default_value
    
    # Get first row
    row = csv_data.iloc[0]
    
    # Try exact match first
    if column_name in csv_data.columns:
        value = row[column_name]
    else:
        # Try case-insensitive match
        col_lower = column_name.lower()
        matching_cols = [col for col in csv_data.columns if col.lower() == col_lower]
        if matching_cols:
            value = row[matching_cols[0]]
        else:
            return default_value
    
    # Handle NaN/None values
    if pd.isna(value):
        return default_value
    
    return value

# CSV File Upload Section
st.markdown("---")
st.subheader("📁 Upload CSV File (Optional)")
uploaded_file = st.file_uploader(
    "Upload a CSV file to auto-fill the form",
    type=['csv'],
    help="Upload a CSV file with patient data. The form will be automatically filled with the data from the first row."
)

if uploaded_file is not None:
    try:
        # Read CSV file
        csv_data = pd.read_csv(uploaded_file)
        
        # Remove empty index column if present (first column with no name or empty name)
        if csv_data.columns[0] == '' or csv_data.columns[0].startswith('Unnamed'):
            csv_data = csv_data.drop(columns=[csv_data.columns[0]])
        
        # Check if CSV is already preprocessed
        is_preprocessed = is_preprocessed_csv(csv_data)
        st.session_state.csv_is_preprocessed = is_preprocessed
        
        if is_preprocessed:
            st.info("🔍 **Detected preprocessed CSV** - Data appears to be already preprocessed and normalized. Will use directly for prediction.")
            # Store original preprocessed data for processing (don't denormalize)
            st.session_state.uploaded_data_raw = csv_data.copy()  # For prediction
            
            # Try to retrieve data from bronze_one_hot_encoded.csv for form filling and preview
            bronze_data = retrieve_bronze_data(csv_data)
            if bronze_data is not None:
                st.success("✅ Found matching data in bronze_one_hot_encoded.csv - Using original values for form and preview.")
                st.session_state.uploaded_data = bronze_data  # For form filling (from bronze, not normalized)
                csv_data_for_preview = bronze_data
            else:
                st.warning("⚠️ Could not find matching data in bronze_one_hot_encoded.csv. Using denormalized values instead.")
                # Fallback: try to denormalize (but this may not be perfect)
                csv_data_denormalized = csv_data.copy()
                if scaler is not None:
                    # Simple denormalization for continuous columns only
                    scaler_features = [f.lower() for f in scaler.feature_names_in_] if hasattr(scaler, 'feature_names_in_') else []
                    for col in csv_data_denormalized.columns:
                        col_lower = col.lower()
                        if col_lower in scaler_features and col_lower not in ['subject_id', 'hadm_id']:
                            feat_idx = scaler_features.index(col_lower)
                            if pd.api.types.is_numeric_dtype(csv_data_denormalized[col]):
                                csv_data_denormalized[col] = csv_data_denormalized[col] * scaler.scale_[feat_idx] + scaler.mean_[feat_idx]
                st.session_state.uploaded_data = csv_data_denormalized
                csv_data_for_preview = csv_data_denormalized
        else:
            # Not preprocessed - use as is
            st.session_state.uploaded_data = csv_data
            st.session_state.uploaded_data_raw = None  # Not preprocessed, so no separate raw version
            csv_data_for_preview = csv_data
        
        # Show success message
        st.success(f"✅ CSV file loaded successfully! Found {len(csv_data)} row(s) and {len(csv_data.columns)} columns.")
        st.info(f"📋 Columns found: {', '.join(csv_data.columns.tolist()[:10])}{'...' if len(csv_data.columns) > 10 else ''}")
        
        # Show preview with denormalized values
        with st.expander("📊 Preview CSV Data (Denormalized)"):
            st.dataframe(csv_data_for_preview.head())
            
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.session_state.uploaded_data = None
        st.session_state.uploaded_data_raw = None
        st.session_state.csv_is_preprocessed = False

st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.info("Fill in the patient information below and click 'Predict CAUTI Risk' to get the prediction.")

# Main form
st.header("Patient Information Form")

with st.form("patient_form"):
    # Get uploaded data
    csv_data = st.session_state.uploaded_data
    
    # Helper function to safely get CSV value with type conversion
    def get_value(column, default, value_type=float, min_val=None, max_val=None):
        val = get_csv_value(csv_data, column, default)
        if val is None or pd.isna(val):
            return default
        try:
            if value_type == int:
                result = int(float(val))
                # Clip to valid range if provided
                if min_val is not None and result < min_val:
                    result = min_val
                if max_val is not None and result > max_val:
                    result = max_val
                return result
            elif value_type == float:
                result = float(val)
                # Clip to valid range if provided
                if min_val is not None and result < min_val:
                    result = min_val
                if max_val is not None and result > max_val:
                    result = max_val
                return result
            elif value_type == bool:
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(val)
                if isinstance(val, str):
                    return val.lower() in ['true', '1', 'yes', 'y']
                return False
            else:
                return str(val) if val is not None else default
        except:
            return default
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        subject_id = st.number_input("Subject ID", min_value=0, value=get_value('subject_id', 10000032, int), step=1)
        hadm_id = st.number_input("Hospital Admission ID", min_value=0, value=get_value('hadm_id', 22595853, int), step=1)
        
        # Gender handling
        gender_options = ["M", "F", "MALE", "FEMALE"]
        gender_csv = get_csv_value(csv_data, 'gender', 'F')
        gender_index = 1  # default
        if gender_csv:
            gender_str = str(gender_csv).upper()
            if gender_str in ['M', 'MALE']:
                gender_index = 0
            elif gender_str in ['F', 'FEMALE']:
                gender_index = 1
            elif gender_str == 'M':
                gender_index = 0
            else:
                gender_index = 1
        gender = st.selectbox("Gender", gender_options, index=gender_index)
        
        anchor_age = st.number_input("Age", min_value=0, max_value=120, value=get_value('anchor_age', 52, int, min_val=0, max_val=120), step=1)
        
        st.subheader("Admission Information")
        admission_type_options = ["AMBULATORY OBSERVATION", "DIRECT EMER", "DIRECT OBSERVATION", 
             "ELECTIVE", "EU OBSERVATION", "EW EMER", "OBSERVATION ADMIT", 
             "SURGICAL SAME DAY ADMISSION", "URGENT"]
        admission_type_csv = get_csv_value(csv_data, 'admission_type', 'AMBULATORY OBSERVATION')
        admission_type_index = 0
        if admission_type_csv and str(admission_type_csv).upper() in [opt.upper() for opt in admission_type_options]:
            admission_type_index = [opt.upper() for opt in admission_type_options].index(str(admission_type_csv).upper())
        admission_type = st.selectbox("Admission Type", admission_type_options, index=admission_type_index)
        
        admission_location_options = ["AMBULATORY SURGERY TRANSFER", "CLINIC REFERRAL", "EMERGENCY ROOM",
             "INFORMATION NOT AVAILABLE", "INTERNAL TRANSFER TO OR FROM PSYCH",
             "PACU", "PHYSICIAN REFERRAL", "PROCEDURE SITE",
             "TRANSFER FROM HOSPITAL", "TRANSFER FROM SKILLED NURSING FACILITY",
             "WALKINSELF REFERRAL"]
        admission_location_csv = get_csv_value(csv_data, 'admission_location', 'EMERGENCY ROOM')
        admission_location_index = 2
        if admission_location_csv and str(admission_location_csv).upper() in [opt.upper() for opt in admission_location_options]:
            admission_location_index = [opt.upper() for opt in admission_location_options].index(str(admission_location_csv).upper())
        admission_location = st.selectbox("Admission Location", admission_location_options, index=admission_location_index)
        
        discharge_location_options = ["ACUTE HOSPITAL", "AGAINST ADVICE", "DEAD/EXPIRED", "HOME",
             "HOME HEALTH CARE", "HOSPICE", "LONG TERM CARE HOSPITAL",
             "REHAB", "SKILLED NURSING FACILITY", "UNKNOWN"]
        discharge_location_csv = get_csv_value(csv_data, 'discharge_location', 'HOME')
        discharge_location_index = 3
        if discharge_location_csv and str(discharge_location_csv).upper() in [opt.upper() for opt in discharge_location_options]:
            discharge_location_index = [opt.upper() for opt in discharge_location_options].index(str(discharge_location_csv).upper())
        discharge_location = st.selectbox("Discharge Location", discharge_location_options, index=discharge_location_index)
        
        st.subheader("Catheter Information")
        catheter_present = st.checkbox("Catheter Present", value=get_value('catheter_present', True, bool))
        
        # Handle catheter_type - can be a list or string
        catheter_type_options = ["Foley", "Straight", "Suprapubic", "Ureteral", "Nephrostomy", "Unknown"]
        catheter_type_csv = get_csv_value(csv_data, 'catheter_type', [])
        catheter_type_default = ["Foley"]
        if catheter_type_csv:
            if isinstance(catheter_type_csv, str):
                # Try to parse as list or use as single value
                if ',' in str(catheter_type_csv):
                    catheter_type_default = [t.strip() for t in str(catheter_type_csv).split(',') if t.strip() in catheter_type_options]
                else:
                    if str(catheter_type_csv) in catheter_type_options:
                        catheter_type_default = [str(catheter_type_csv)]
            elif isinstance(catheter_type_csv, list):
                catheter_type_default = [t for t in catheter_type_csv if t in catheter_type_options]
        catheter_type = st.multiselect("Catheter Type", catheter_type_options, default=catheter_type_default)
        
        catheter_size = st.text_input("Catheter Size (e.g., '16FR', '18FR')", value=get_value('catheter_size', '', str))
        catheter_indication = st.checkbox("Catheter Indication", value=get_value('catheter_indication', False, bool))
        catheter_care = st.checkbox("Catheter Care", value=get_value('catheter_care', False, bool))
        closed_system = st.checkbox("Closed System", value=get_value('closed_system', False, bool))
        securement_device = st.checkbox("Securement Device", value=get_value('securement_device', False, bool))
        urinary_obstruction_present = st.checkbox("Urinary Obstruction Present", value=get_value('urinary_obstruction_present', False, bool))
        improper_drainage_position = st.checkbox("Improper Drainage Position", value=get_value('improper_drainage_position', False, bool))
        catheter_insertion = st.checkbox("Catheter Insertion", value=get_value('catheter_insertion', False, bool))
        catheter_removal = st.checkbox("Catheter Removal", value=get_value('catheter_removal', False, bool))
        catheter_removal_replacement = st.checkbox("Catheter Removal/Replacement", value=get_value('catheter_removal_replacement', False, bool))
        
    with col2:
        st.subheader("Medical History & Conditions")
        diabetes = st.checkbox("Diabetes", value=get_value('diabetes', False, bool))
        cancer = st.checkbox("Cancer", value=get_value('cancer', False, bool))
        chronic_kidney_disease = st.checkbox("Chronic Kidney Disease", value=get_value('chronic_kidney_disease', False, bool))
        neurogenic_bladder = st.checkbox("Neurogenic Bladder", value=get_value('neurogenic_bladder', False, bool))
        recurrent_uti = st.checkbox("Recurrent UTI", value=get_value('recurrent_uti', False, bool))
        spinal_cord_injury = st.checkbox("Spinal Cord Injury", value=get_value('spinal_cord_injury', False, bool))
        neurological_disorder = st.checkbox("Neurological Disorder", value=get_value('neurological_disorder', False, bool))
        benign_prostatic_hyperplasia = st.checkbox("Benign Prostatic Hyperplasia", value=get_value('benign_prostatic_hyperplasia', False, bool))
        has_cauti_history = st.checkbox("Has CAUTI History", value=get_value('has_cauti_history', False, bool))
        
        st.subheader("Vital Signs & Lab Values")
        BMI = st.number_input("BMI", min_value=10.0, max_value=100.0, value=get_value('BMI', 25.0, float, min_val=10.0, max_val=100.0), step=0.1)
        temperature = st.number_input("Temperature (°F)", min_value=80.0, max_value=108.0, value=get_value('temperature', 98.6, float, min_val=80.0, max_val=108.0), step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220, value=get_value('heart_rate', 72, int, min_val=40, max_val=220), step=1)
        resp_rate = st.number_input("Respiratory Rate", min_value=5, max_value=40, value=get_value('resp_rate', 16, int, min_val=5, max_val=40), step=1)
        o2sat = st.number_input("O2 Saturation (%)", min_value=70, max_value=100, value=get_value('o2sat', 98, int, min_val=70, max_val=100), step=1)
        BP_systolic = st.number_input("BP Systolic (mmHg)", min_value=60, max_value=250, value=get_value('BP_systolic', 120, int, min_val=60, max_val=250), step=1)
        BP_diastolic = st.number_input("BP Diastolic (mmHg)", min_value=40, max_value=130, value=get_value('BP_diastolic', 80, int, min_val=40, max_val=130), step=1)
        
        st.subheader("Lab Results")
        urinalysis_wbc = st.number_input("Urinalysis WBC (HPF)", min_value=0.0, max_value=500.0, value=get_value('urinalysis_wbc', 0.0, float, min_val=0.0, max_val=500.0), step=0.1)
        urinalysis_rbc = st.number_input("Urinalysis RBC (HPF)", min_value=0.0, max_value=500.0, value=get_value('urinalysis_rbc', 0.0, float, min_val=0.0, max_val=500.0), step=0.1)
        urinalysis_nitrite_options = ["Not Tested", "Negative", "Positive"]
        urinalysis_nitrite_csv = get_csv_value(csv_data, 'urinalysis_nitrite', 'Not Tested')
        urinalysis_nitrite_index = 0
        if urinalysis_nitrite_csv and str(urinalysis_nitrite_csv).title() in urinalysis_nitrite_options:
            urinalysis_nitrite_index = urinalysis_nitrite_options.index(str(urinalysis_nitrite_csv).title())
        urinalysis_nitrite = st.selectbox("Urinalysis Nitrite", urinalysis_nitrite_options, index=urinalysis_nitrite_index)
        blood_wbc = st.number_input("Blood WBC (×10³/µL)", min_value=0.0, max_value=200.0, value=get_value('blood_wbc', 7.0, float, min_val=0.0, max_val=200.0), step=0.1)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=get_value('creatinine', 1.0, float, min_val=0.0, max_val=20.0), step=0.1)
        
        st.subheader("Other Clinical Information")
        charlson_score = st.number_input("Charlson Score", min_value=0, max_value=35, value=get_value('charlson_score', 0, int, min_val=0, max_val=35), step=1)
        length_of_stay = st.number_input("Length of Stay (days)", min_value=0.0, value=get_value('length_of_stay', 1.0, float, min_val=0.0), step=0.1)
        num_of_transfers = st.number_input("Number of Transfers", min_value=0, value=get_value('num_of_transfers', 0, int, min_val=0), step=1)
        surgical_admission = st.checkbox("Surgical Admission", value=get_value('surgical_admission', False, bool))
        recent_urologic_abdominal_surgery = st.checkbox("Recent Urologic/Abdominal Surgery", value=get_value('recent_urologic_abdominal_surgery', False, bool))
        icu_admission = st.checkbox("ICU Admission", value=get_value('icu_admission', False, bool))
        mobility_status_options = ["AMBULATORY", "BEDREST", "RESTRICTED", "UNKNOWN"]
        mobility_status_csv = get_csv_value(csv_data, 'mobility_status', 'AMBULATORY')
        mobility_status_index = 0
        if mobility_status_csv and str(mobility_status_csv).upper() in [opt.upper() for opt in mobility_status_options]:
            mobility_status_index = [opt.upper() for opt in mobility_status_options].index(str(mobility_status_csv).upper())
        mobility_status = st.selectbox("Mobility Status", mobility_status_options, index=mobility_status_index)
        n_catheter_manip_unique_types = st.number_input("Number of Catheter Manipulation Types", min_value=0, value=get_value('n_catheter_manip_unique_types', 0, int, min_val=0), step=1)
        no_of_invasive_devices = st.number_input("Number of Invasive Devices", min_value=0, value=get_value('no_of_invasive_devices', 0, int, min_val=0), step=1)
        multiple_invasive_devices = st.checkbox("Multiple Invasive Devices", value=get_value('multiple_invasive_devices', False, bool))
        
    # Additional columns for remaining features
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Medications & Treatments")
        antibiotics_per_admission = st.checkbox("Antibiotics Per Admission", value=get_value('antibiotics_per_admission', False, bool))
        recent_antibiotic_use = st.checkbox("Recent Antibiotic Use", value=get_value('recent_antibiotic_use', False, bool))
        steroids_per_admission = st.checkbox("Steroids Per Admission", value=get_value('steroids_per_admission', False, bool))
        immunosuppressives_per_admission = st.checkbox("Immunosuppressives Per Admission", value=get_value('immunosuppressives_per_admission', False, bool))
        ppi_per_admission = st.checkbox("PPI Per Admission", value=get_value('ppi_per_admission', False, bool))
        procalcitonin_measured = st.checkbox("Procalcitonin Measured", value=get_value('procalcitonin_measured', False, bool))
        oliguria = st.checkbox("Oliguria", value=get_value('oliguria', False, bool))
        urine_culture_performed = st.checkbox("Urine Culture Performed", value=get_value('urine_culture_performed', False, bool))
        blood_culture_performed = st.checkbox("Blood Culture Performed", value=get_value('blood_culture_performed', False, bool))
        pain_documented = st.checkbox("Pain Documented", value=get_value('pain_documented', False, bool))
        ventilator_used = st.checkbox("Ventilator Used", value=get_value('ventilator_used', False, bool))
        
    with col4:
        st.subheader("Culture Results & Other")
        # Handle other_uti - can be a list or string
        other_uti_options = ["UTI Unspecified", "Cystitis", "Pyelonephritis", "Urethritis"]
        other_uti_csv = get_csv_value(csv_data, 'other_uti', [])
        other_uti_default = []
        if other_uti_csv:
            if isinstance(other_uti_csv, str):
                if ',' in str(other_uti_csv):
                    other_uti_default = [t.strip() for t in str(other_uti_csv).split(',') if t.strip() in other_uti_options]
                else:
                    if str(other_uti_csv) in other_uti_options:
                        other_uti_default = [str(other_uti_csv)]
            elif isinstance(other_uti_csv, list):
                other_uti_default = [t for t in other_uti_csv if t in other_uti_options]
        other_uti = st.multiselect("Other UTI Types", other_uti_options, default=other_uti_default)
        
        other_uti_present = st.checkbox("Other UTI Present", value=get_value('other_uti_present', False, bool))
        gram_negative_organisms_present = st.checkbox("Gram Negative Organisms Present", value=get_value('gram_negative_organisms_present', False, bool))
        gram_positive_organisms_present = st.checkbox("Gram Positive Organisms Present", value=get_value('gram_positive_organisms_present', False, bool))
        fungi_present = st.checkbox("Fungi Present", value=get_value('fungi_present', False, bool))
        
        # Optional fields that may be missing
        st.subheader("Optional Measurements")
        blood_crp_val = get_csv_value(csv_data, 'blood_crp', None)
        blood_crp = st.number_input("Blood CRP (mg/L)", min_value=0.0, max_value=300.0, value=float(blood_crp_val) if blood_crp_val is not None and not pd.isna(blood_crp_val) else None, step=0.1)
        
        urine_output_val = get_csv_value(csv_data, 'urine_output_ml', None)
        urine_output_ml = st.number_input("Urine Output (mL)", min_value=0.0, value=float(urine_output_val) if urine_output_val is not None and not pd.isna(urine_output_val) else None, step=100.0)
        
        cfu_count_val = get_csv_value(csv_data, 'cfu_count', None)
        cfu_count = st.number_input("CFU Count", min_value=0.0, value=float(cfu_count_val) if cfu_count_val is not None and not pd.isna(cfu_count_val) else None, step=1000.0)
        
        catheter_duration_val = get_csv_value(csv_data, 'catheter_duration_days', None)
        catheter_duration_days = st.number_input("Catheter Duration (days)", min_value=0.0, value=float(catheter_duration_val) if catheter_duration_val is not None and not pd.isna(catheter_duration_val) else None, step=0.1)
        
        # Date fields (will be converted to datetime)
        admittime_val = get_csv_value(csv_data, 'admittime', None)
        admittime = None
        if admittime_val and not pd.isna(admittime_val):
            try:
                admittime = pd.to_datetime(admittime_val).date()
            except:
                pass
        admittime = st.date_input("Admission Time", value=admittime)
        
        dischtime_val = get_csv_value(csv_data, 'dischtime', None)
        dischtime = None
        if dischtime_val and not pd.isna(dischtime_val):
            try:
                dischtime = pd.to_datetime(dischtime_val).date()
            except:
                pass
        dischtime = st.date_input("Discharge Time", value=dischtime)
        
        final_insertion_date_val = get_csv_value(csv_data, 'final_insertion_date', None)
        final_insertion_date = None
        if final_insertion_date_val and not pd.isna(final_insertion_date_val):
            try:
                final_insertion_date = pd.to_datetime(final_insertion_date_val).date()
            except:
                pass
        final_insertion_date = st.date_input("Final Insertion Date", value=final_insertion_date)
        
        final_removal_date_val = get_csv_value(csv_data, 'final_removal_date', None)
        final_removal_date = None
        if final_removal_date_val and not pd.isna(final_removal_date_val):
            try:
                final_removal_date = pd.to_datetime(final_removal_date_val).date()
            except:
                pass
        final_removal_date = st.date_input("Final Removal Date", value=final_removal_date)
    
    # Submit button
    submitted = st.form_submit_button("🔍 Predict CAUTI Risk", use_container_width=True)

# Process form submission
if submitted:
    if model is None:
        st.error("Model not loaded. Please save the trained model first.")
        st.info("""
        To save the model, add this code to your training notebook:
        
        **Option 1: Save as .pkl (recommended)**
        ```python
        import joblib
        joblib.dump(model, 'scripts/models/cauti_ann_loso_model.pkl')
        ```
        
        **Option 2: Save as .h5**
        ```python
        model.save('scripts/models/cauti_ann_loso_model.h5')
        ```
        """)
    else:
        with st.spinner("Processing patient data and making prediction..."):
            try:
                # Create dataframe from form inputs
                data = {
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'admittime': pd.to_datetime(admittime) if admittime else None,
                    'dischtime': pd.to_datetime(dischtime) if dischtime else None,
                    'gender': gender,
                    'anchor_age': anchor_age,
                    'admission_type': admission_type,
                    'admission_location': admission_location,
                    'discharge_location': discharge_location,
                    'catheter_present': catheter_present,
                    'final_insertion_date': pd.to_datetime(final_insertion_date) if final_insertion_date else None,
                    'final_removal_date': pd.to_datetime(final_removal_date) if final_removal_date else None,
                    'catheter_duration_days': catheter_duration_days,
                    'BMI': BMI,
                    'diabetes': diabetes,
                    'cancer': cancer,
                    'chronic_kidney_disease': chronic_kidney_disease,
                    'neurogenic_bladder': neurogenic_bladder,
                    'recurrent_uti': recurrent_uti,
                    'spinal_cord_injury': spinal_cord_injury,
                    'neurological_disorder': neurological_disorder,
                    'benign_prostatic_hyperplasia': benign_prostatic_hyperplasia,
                    'charlson_score': charlson_score,
                    'length_of_stay': length_of_stay,
                    'num_of_transfers': num_of_transfers,
                    'surgical_admission': surgical_admission,
                    'recent_urologic_abdominal_surgery': recent_urologic_abdominal_surgery,
                    'icu_admission': icu_admission,
                    'mobility_status': mobility_status,
                    'catheter_type': catheter_type if catheter_type else [],
                    'catheter_size': catheter_size,
                    'urinary_obstruction_present': urinary_obstruction_present,
                    'improper_drainage_position': improper_drainage_position,
                    'n_catheter_manip_unique_types': n_catheter_manip_unique_types,
                    'catheter_insertion': catheter_insertion,
                    'catheter_removal': catheter_removal,
                    'catheter_removal_replacement': catheter_removal_replacement,
                    'no_of_invasive_devices': no_of_invasive_devices,
                    'multiple_invasive_devices': multiple_invasive_devices,
                    'catheter_indication': catheter_indication,
                    'catheter_care': catheter_care,
                    'closed_system': closed_system,
                    'securement_device': securement_device,
                    'antibiotics_per_admission': antibiotics_per_admission,
                    'recent_antibiotic_use': recent_antibiotic_use,
                    'steroids_per_admission': steroids_per_admission,
                    'immunosuppressives_per_admission': immunosuppressives_per_admission,
                    'ppi_per_admission': ppi_per_admission,
                    'urinalysis_wbc': urinalysis_wbc,
                    'urinalysis_rbc': urinalysis_rbc,
                    'urinalysis_nitrite': urinalysis_nitrite,
                    'urine_bacteria': None,  # This is dropped in preprocessing
                    'blood_wbc': blood_wbc,
                    'creatinine': creatinine,
                    'blood_crp': blood_crp,
                    'procalcitonin_measured': procalcitonin_measured,
                    'urine_output_ml': urine_output_ml,
                    'oliguria': oliguria,
                    'urine_culture_performed': urine_culture_performed,
                    'cfu_count': cfu_count,
                    'gram_negative_organisms_present': gram_negative_organisms_present,
                    'gram_positive_organisms_present': gram_positive_organisms_present,
                    'fungi_present': fungi_present,
                    'blood_culture_performed': blood_culture_performed,
                    'temperature': temperature,
                    'heart_rate': heart_rate,
                    'resp_rate': resp_rate,
                    'o2sat': o2sat,
                    'BP_systolic': BP_systolic,
                    'BP_diastolic': BP_diastolic,
                    'other_uti': other_uti if other_uti else [],
                    'other_uti_present': other_uti_present,
                    'has_cauti_history': has_cauti_history,
                    'pain_documented': pain_documented,
                    'ventilator_used': ventilator_used,
                }
                
                df = pd.DataFrame([data])
                
                # Check if we should use preprocessed CSV directly
                use_preprocessed_csv = (
                    st.session_state.get('csv_is_preprocessed', False) and
                    st.session_state.get('uploaded_data_raw') is not None and
                    len(st.session_state.uploaded_data_raw) > 0
                )
                
                if use_preprocessed_csv:
                    # Use preprocessed CSV data directly (first row) - use raw normalized data
                    df_processed = st.session_state.uploaded_data_raw.iloc[[0]].copy()
                    st.info("ℹ️ Using preprocessed CSV data directly (skipping preprocessing pipeline).")
                else:
                    # Apply preprocessing pipeline for raw data
                    df_processed = apply_outlier_handling(df.copy())
                    df_processed = apply_missing_value_handling(df_processed.copy())
                    df_processed = apply_binary_encoding(df_processed.copy())
                    df_processed = apply_one_hot_encoding(df_processed.copy())
                    df_processed = apply_normalization(df_processed.copy(), scaler)
                
                # Get feature order and ensure alignment
                feature_order = get_feature_order()
                
                if feature_order is None:
                    # Try to infer from scaler
                    if hasattr(scaler, 'feature_names_in_'):
                        feature_order = list(scaler.feature_names_in_)
                    else:
                        # Fallback: use all numeric columns except IDs
                        feature_order = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                                        if col not in ['subject_id', 'hadm_id']]
                
                # Ensure all features exist, add missing ones with 0
                for feat in feature_order:
                    if feat not in df_processed.columns:
                        df_processed[feat] = 0
                
                # Reorder columns to match feature order
                df_processed = df_processed.reindex(columns=feature_order, fill_value=0)
                
                # Extract only the features (exclude IDs)
                X_features = [col for col in feature_order if col not in ['subject_id', 'hadm_id']]
                X_data = df_processed[X_features].values
                
                # Scale the data (skip if already preprocessed/normalized)
                if use_preprocessed_csv:
                    # Data is already normalized, use as-is
                    X_scaled = X_data
                else:
                    # Apply scaling for raw data
                    X_scaled = scaler.transform(X_data)
                
                # Make prediction
                # Handle both Keras models (from .h5 or .pkl) and sklearn models
                try:
                    if hasattr(model, 'predict_proba'):
                        # sklearn model
                        prediction_prob = model.predict_proba(X_scaled)[0][1]
                    else:
                        # Keras/TensorFlow model
                        pred = model.predict(X_scaled, verbose=0)
                        if isinstance(pred, np.ndarray) and pred.ndim > 1:
                            prediction_prob = pred[0][0] if pred.shape[1] > 1 else pred[0]
                        else:
                            prediction_prob = float(pred[0])
                    
                    # Ensure prediction_prob is a Python float (not numpy scalar)
                    prediction_prob = float(prediction_prob)
                    
                    prediction_class = 1 if prediction_prob >= 0.5 else 0
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.exception(e)
                    st.stop()
                
                # Display results
                st.success("✅ Prediction Complete!")
                st.markdown("---")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric("CAUTI Risk Probability", f"{prediction_prob:.2%}")
                    if prediction_class == 1:
                        st.error("⚠️ **HIGH RISK** - Patient is at risk for CAUTI")
                    else:
                        st.success("✅ **LOW RISK** - Patient is not at high risk for CAUTI")
                
                with col_result2:
                    st.metric("Risk Threshold", "50%")
                    st.info(f"Prediction confidence: {abs(prediction_prob - 0.5) * 200:.1f}%")
                
                # Additional information
                st.markdown("---")
                st.subheader("📊 Prediction Details")
                st.write(f"**Risk Score:** {prediction_prob:.4f}")
                st.write(f"**Interpretation:** {'High Risk' if prediction_class == 1 else 'Low Risk'}")
                
                st.session_state.prediction_made = True
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("**Note:** This prediction is based on machine learning models and should be used as a decision support tool, not as a replacement for clinical judgment.")

