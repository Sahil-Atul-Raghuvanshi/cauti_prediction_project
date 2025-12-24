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

# Helper function to get project root
def get_project_root():
    """Get project root directory (parent of streamlit_app)."""
    # Method 1: Try using __file__ (most reliable when available)
    try:
        if '__file__' in globals() and __file__:
            file_path = Path(__file__).resolve()
            # If file is in streamlit_app/, go up one level to project root
            if file_path.parent.name == 'streamlit_app':
                root = file_path.parent.parent
                # Verify it's the project root by checking for key directories
                if (root / 'scripts').exists() and (root / 'data').exists():
                    return root
    except (NameError, AttributeError, TypeError):
        pass
    
    # Method 2: Use current working directory
    cwd = Path.cwd().resolve()
    # If we're in streamlit_app directory, go up one level
    if cwd.name == 'streamlit_app':
        root = cwd.parent
        if (root / 'scripts').exists() and (root / 'data').exists():
            return root
    
    # Method 3: If we're already in project root
    if (cwd / 'scripts').exists() and (cwd / 'data').exists():
        return cwd
    
    # Method 4: Try to find project root by going up directories
    current = cwd
    for _ in range(5):  # Go up max 5 levels
        if (current / 'scripts').exists() and (current / 'data').exists():
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Last resort: return current directory (might not be correct, but better than error)
    return cwd

# Load scaler and model
@st.cache_resource
def load_scaler():
    """
    Load model scaler (optional - model no longer uses scaling).
    Returns None if scaler not found (which is expected after removing scaling from training).
    """
    # Get project root
    PROJECT_ROOT = get_project_root()
    scaler_path = PROJECT_ROOT / "scripts" / "models" / "cauti_ann_loso_scaler.pkl"
    if scaler_path.exists():
        return joblib.load(str(scaler_path))
    else:
        # Scaler not found is OK - model was trained without scaling
        return None

@st.cache_resource
def load_model():
    """
    Load the Keras model with handling for quantization_config errors.
    Tries .keras format first (recommended), then .h5, then .pkl.
    """
    def load_keras_model_safe(model_path):
        """Load Keras model with error handling for quantization_config issues."""
        # Ensure model_path is a string and absolute
        if isinstance(model_path, Path):
            model_path = str(model_path.resolve())
        elif isinstance(model_path, str):
            model_path = str(Path(model_path).resolve())
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
                        temp_keras_str = str(temp_keras)
                        try:
                            model = tf.keras.models.load_model(temp_keras_str)
                        except:
                            model = tf.keras.models.load_model(temp_keras_str, compile=False)
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
                    # model_path is already a string, so convert to Path for operations
                    model_path_obj = Path(model_path)
                    h5_path = model_path_obj.with_suffix('.h5')
                    if h5_path.exists() and str(h5_path) != model_path:
                        try:
                            return tf.keras.models.load_model(str(h5_path), compile=False)
                        except:
                            pass
                    raise e2
            else:
                raise e
    
    # Get project root
    PROJECT_ROOT = get_project_root()
    
    # Try different possible model file names and formats
    # Prioritize .keras format (newest, avoids quantization_config issues)
    models_dir = PROJECT_ROOT / "scripts" / "models"
    model_paths = [
        models_dir / "cauti_ann_loso_model.keras",  # Newest Keras format (preferred)
        models_dir / "cauti_ann_loso_model.h5",  # Legacy H5 format
        models_dir / "cauti_ann_loso_model.pkl",  # Fallback to .pkl
        models_dir / "model.pkl",
        models_dir / "ann_loso_model.pkl",
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            try:
                # Convert Path to string for TensorFlow/Keras
                model_path_str = str(model_path.resolve())  # Use absolute path
                if model_path.suffix in ['.keras', '.h5']:
                    # Load as TensorFlow/Keras model
                    model = load_keras_model_safe(model_path_str)
                    if model is not None:
                        return model
                    # If load_keras_model_safe returns None, continue to next path
                elif model_path.suffix == '.pkl':
                    # Try loading as joblib (could be Keras or sklearn model)
                    # Note: This may have deserialization issues with Keras models
                    model = joblib.load(model_path_str)
                    # Check if it's a Keras model (has predict method and is callable)
                    if hasattr(model, 'predict') and callable(getattr(model, 'predict', None)):
                        return model
                    else:
                        st.warning(f"Model loaded from {model_path} but doesn't have a predict method.")
                        continue
            except Exception as e:
                st.warning(f"Error loading model from {model_path}: {str(e)}")
                import traceback
                st.exception(e)
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
scaler = load_scaler()  # Optional - model no longer uses scaling
model = load_model()

if model is None:
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
    # Get project root
    PROJECT_ROOT = get_project_root()
    bronze_path = PROJECT_ROOT / "data" / "silver" / "bronze_one_hot_encoded.csv"
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

# Helper function to reverse one-hot encoding
def reverse_one_hot_encoding(csv_data, prefix, options):
    """
    Reverse one-hot encoding to get original categorical value.
    Example: If admission_type_observation_admit=1, return "OBSERVATION ADMIT"
    """
    if csv_data is None or len(csv_data) == 0:
        return None
    
    row = csv_data.iloc[0]
    prefix_lower = prefix.lower()
    
    # Find all columns that start with the prefix
    matching_cols = [col for col in csv_data.columns if col.lower().startswith(prefix_lower + '_')]
    
    if not matching_cols:
        return None
    
    # Find the column with value = 1
    for col in matching_cols:
        val = row[col]
        if pd.notna(val) and (val == 1 or val == 1.0):
            # Extract the category name from column name
            # e.g., "admission_type_observation_admit" -> "OBSERVATION ADMIT"
            # e.g., "mobility_status_out_of_bed_with_assistance" -> "OUT OF BED WITH ASSISTANCE"
            category_part = col[len(prefix) + 1:]  # Remove prefix and underscore
            # Convert snake_case to UPPER CASE with spaces
            category_name = category_part.replace('_', ' ').upper()
            # Handle double underscores (e.g., "nonambulatory__paralysis")
            category_name = category_name.replace('  ', ' ')
            
            # Try to match with options (case-insensitive, handle various formats)
            for opt in options:
                opt_upper = opt.upper()
                # Try multiple matching strategies
                if (opt_upper == category_name or  # Exact match
                    opt_upper.replace(' ', '_') == category_part.upper() or  # Match with underscores
                    opt_upper.replace(' ', '_').replace('__', '_') == category_part.upper().replace('__', '_')):  # Handle double underscores
                    return opt
            
            # If no exact match, return the formatted name
            return category_name
    
    return None

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
        # Note: bronze_one_hot_encoded.csv has gender as 0/1 (already encoded)
        # Form needs M/F, so we need to convert: 0/1 -> F/M
        gender_options = ["M", "F", "MALE", "FEMALE"]
        gender_csv = get_csv_value(csv_data, 'gender', 'F')
        gender_index = 1  # default to F
        if gender_csv is not None:
            # Handle both string (M/F) and numeric (0/1) values
            if isinstance(gender_csv, (int, float)):
                # Already encoded: 0 = F, 1 = M
                gender_index = 0 if gender_csv == 1 else 1
            else:
                gender_str = str(gender_csv).upper()
                if gender_str in ['M', 'MALE', '1']:
                    gender_index = 0
                elif gender_str in ['F', 'FEMALE', '0']:
                    gender_index = 1
        gender = st.selectbox("Gender", gender_options, index=gender_index)
        
        anchor_age = st.number_input("Age", min_value=0, max_value=120, value=get_value('anchor_age', 52, int, min_val=0, max_val=120), step=1)
        
        st.subheader("Admission Information")
        admission_type_options = ["AMBULATORY OBSERVATION", "DIRECT EMER", "DIRECT OBSERVATION", 
             "ELECTIVE", "EU OBSERVATION", "EW EMER", "OBSERVATION ADMIT", 
             "SURGICAL SAME DAY ADMISSION", "URGENT"]
        # Try to get from original column first, then reverse one-hot encoding
        admission_type_csv = get_csv_value(csv_data, 'admission_type', None)
        if admission_type_csv is None:
            admission_type_csv = reverse_one_hot_encoding(csv_data, 'admission_type', admission_type_options)
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
        # Try to get from original column first, then reverse one-hot encoding
        discharge_location_csv = get_csv_value(csv_data, 'discharge_location', None)
        if discharge_location_csv is None:
            discharge_location_csv = reverse_one_hot_encoding(csv_data, 'discharge_location', discharge_location_options)
        discharge_location_index = 3
        if discharge_location_csv and str(discharge_location_csv).upper() in [opt.upper() for opt in discharge_location_options]:
            discharge_location_index = [opt.upper() for opt in discharge_location_options].index(str(discharge_location_csv).upper())
        discharge_location = st.selectbox("Discharge Location", discharge_location_options, index=discharge_location_index)
        
        st.subheader("Catheter Information")
        catheter_present = st.checkbox("Catheter Present", value=get_value('catheter_present', True, bool))
        
        # Handle catheter_type - can be a list or string
        catheter_type_options = ["Foley", "Straight", "Suprapubic", "Ureteral", "Nephrostomy", "Unknown"]
        catheter_type_csv = get_csv_value(csv_data, 'catheter_type', [])
        catheter_type_default = []  # No default selection
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
        
        # catheter_size handling
        # Note: bronze_one_hot_encoded.csv has catheter_size_fr (numeric), not catheter_size (string)
        # We'll use catheter_size_fr directly - if it exists in bronze data, use it; otherwise use empty string
        catheter_size_fr_val = get_csv_value(csv_data, 'catheter_size_fr', None)
        if catheter_size_fr_val is not None and not pd.isna(catheter_size_fr_val) and float(catheter_size_fr_val) != -1:
            # Convert numeric value to string format for display
            catheter_size_val = f"{int(catheter_size_fr_val)}FR"
        else:
            # Try to get from catheter_size string field, or use empty
            catheter_size_val = get_csv_value(csv_data, 'catheter_size', '')
        catheter_size = st.text_input("Catheter Size (e.g., '16FR', '18FR')", value=str(catheter_size_val) if catheter_size_val else '')
        catheter_indication = st.checkbox("Catheter Indication", value=get_value('catheter_indication', False, bool))
        catheter_care = st.checkbox("Catheter Care", value=get_value('catheter_care', False, bool))
        closed_system = st.checkbox("Closed System", value=get_value('closed_system', False, bool))
        securement_device = st.checkbox("Securement Device", value=get_value('securement_device', False, bool))
        urinary_obstruction_present = st.checkbox("Urinary Obstruction Present", value=get_value('urinary_obstruction_present', False, bool))
        improper_drainage_position = st.checkbox("Improper Drainage Position", value=get_value('improper_drainage_position', False, bool))
        catheter_insertion = st.checkbox("Catheter Insertion", value=get_value('catheter_insertion', False, bool))
        catheter_removal = st.checkbox("Catheter Removal", value=get_value('catheter_removal', False, bool))
        catheter_removal_replacement = st.checkbox("Catheter Removal/Replacement", value=get_value('catheter_removal_replacement', False, bool))
        
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
        
        st.subheader("Other Clinical Information")
        charlson_score = st.number_input("Charlson Score", min_value=0, max_value=35, value=get_value('charlson_score', 0, int, min_val=0, max_val=35), step=1)
        length_of_stay = st.number_input("Length of Stay (days)", min_value=0.0, value=get_value('length_of_stay', 1.0, float, min_val=0.0), step=0.1)
        num_of_transfers = st.number_input("Number of Transfers", min_value=0, value=get_value('num_of_transfers', 0, int, min_val=0), step=1)
        surgical_admission = st.checkbox("Surgical Admission", value=get_value('surgical_admission', False, bool))
        recent_urologic_abdominal_surgery = st.checkbox("Recent Urologic/Abdominal Surgery", value=get_value('recent_urologic_abdominal_surgery', False, bool))
        icu_admission = st.checkbox("ICU Admission", value=get_value('icu_admission', False, bool))
        # Mobility status options - must match bronze data values (before one-hot encoding)
        mobility_status_options = [
            "Assisted Ambulatory", "Bedbound", "Independent Ambulatory", 
            "Non-ambulatory / Paralysis", "Out of Bed with Assistance", 
            "Unknown", "Wheelchair"
        ]
        # Try to get from original column first, then reverse one-hot encoding
        mobility_status_csv = get_csv_value(csv_data, 'mobility_status', None)
        if mobility_status_csv is None:
            mobility_status_csv = reverse_one_hot_encoding(csv_data, 'mobility_status', mobility_status_options)
        mobility_status_index = 0
        if mobility_status_csv:
            mobility_status_str = str(mobility_status_csv).upper()
            # Try to match with options (handle various formats)
            for i, opt in enumerate(mobility_status_options):
                opt_upper = opt.upper()
                # Try exact match, or match with underscores/spaces normalized
                if (opt_upper == mobility_status_str or 
                    opt_upper.replace(' ', '_') == mobility_status_str.replace(' ', '_') or
                    opt_upper.replace('__', '_') == mobility_status_str.replace('__', '_')):
                    mobility_status_index = i
                    break
        mobility_status = st.selectbox("Mobility Status", mobility_status_options, index=mobility_status_index)
        n_catheter_manip_unique_types = st.number_input("Number of Catheter Manipulation Types", min_value=0, value=get_value('n_catheter_manip_unique_types', 0, int, min_val=0), step=1)
        no_of_invasive_devices = st.number_input("Number of Invasive Devices", min_value=0, value=get_value('no_of_invasive_devices', 0, int, min_val=0), step=1)
        multiple_invasive_devices = st.checkbox("Multiple Invasive Devices", value=get_value('multiple_invasive_devices', False, bool))
        
    with col2:
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
        # urinalysis_nitrite handling
        # Note: bronze_one_hot_encoded.csv has nitrite_tested/nitrite_positive (already processed)
        # Form needs urinalysis_nitrite, so we need to reverse: nitrite_tested/nitrite_positive -> urinalysis_nitrite
        urinalysis_nitrite_options = ["Not Tested", "Negative", "Positive"]
        urinalysis_nitrite_csv = get_csv_value(csv_data, 'urinalysis_nitrite', None)
        urinalysis_nitrite_index = 0  # default to "Not Tested"
        
        # If urinalysis_nitrite not found, try to derive from nitrite_tested/nitrite_positive
        if urinalysis_nitrite_csv is None:
            nitrite_tested = get_csv_value(csv_data, 'nitrite_tested', None)
            nitrite_positive = get_csv_value(csv_data, 'nitrite_positive', None)
            if nitrite_tested is not None:
                if nitrite_tested == 0:
                    urinalysis_nitrite_index = 0  # Not Tested
                elif nitrite_positive == 1:
                    urinalysis_nitrite_index = 2  # Positive
                else:
                    urinalysis_nitrite_index = 1  # Negative
        elif urinalysis_nitrite_csv and str(urinalysis_nitrite_csv).title() in urinalysis_nitrite_options:
            urinalysis_nitrite_index = urinalysis_nitrite_options.index(str(urinalysis_nitrite_csv).title())
        
        urinalysis_nitrite = st.selectbox("Urinalysis Nitrite", urinalysis_nitrite_options, index=urinalysis_nitrite_index)
        blood_wbc = st.number_input("Blood WBC (×10³/µL)", min_value=0.0, max_value=200.0, value=get_value('blood_wbc', 7.0, float, min_val=0.0, max_val=200.0), step=0.1)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=get_value('creatinine', 1.0, float, min_val=0.0, max_val=20.0), step=0.1)
        
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
        
        st.subheader("Culture Results & Other")
        # Handle other_uti - can be a list or string
        # Note: bronze_one_hot_encoded.csv has one-hot encoded columns like other_uti_uti_unspecified
        other_uti_options = ["UTI Unspecified", "Cystitis", "Pyelonephritis", "Urethritis"]
        other_uti_csv = get_csv_value(csv_data, 'other_uti', None)
        other_uti_default = []
        
        # If other_uti not found, try to reverse one-hot encoding
        if other_uti_csv is None:
            # Check for one-hot encoded columns
            if csv_data is not None and len(csv_data) > 0:
                row = csv_data.iloc[0]
                other_uti_cols = [col for col in csv_data.columns if col.lower().startswith('other_uti_')]
                for col in other_uti_cols:
                    val = row[col]
                    if pd.notna(val) and (val == 1 or val == 1.0):
                        # Extract category name: "other_uti_uti_unspecified" -> "UTI Unspecified"
                        category_part = col[len('other_uti_'):]
                        # Convert to title case: "uti_unspecified" -> "UTI Unspecified"
                        category_name = category_part.replace('_', ' ').title()
                        # Map to option
                        for opt in other_uti_options:
                            if opt.upper().replace(' ', '_') == category_part.upper() or opt.upper() == category_name.upper():
                                other_uti_default.append(opt)
                                break
        
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
        
    # Additional columns for remaining features
    col3, col4 = st.columns(2)
    
    with col3:
        # Optional fields that may be missing
        st.subheader("Optional Measurements")
        blood_crp_val = get_csv_value(csv_data, 'blood_crp', None)
        blood_crp = st.number_input("Blood CRP (mg/L)", min_value=0.0, max_value=300.0, value=float(blood_crp_val) if blood_crp_val is not None and not pd.isna(blood_crp_val) else None, step=0.1)
        
        urine_output_val = get_csv_value(csv_data, 'urine_output_ml', None)
        urine_output_ml = st.number_input("Urine Output (mL)", min_value=0.0, value=float(urine_output_val) if urine_output_val is not None and not pd.isna(urine_output_val) else None, step=100.0)
        
        # cfu_count handling
        # Note: bronze_one_hot_encoded.csv has cfu_count_measured (already processed)
        # Form needs cfu_count. If cfu_count_measured=1, we know it was measured but can't reverse the value.
        # We'll use a placeholder value (1.0) so that cfu_count_measured will be set to 1 in preprocessing.
        cfu_count_val = get_csv_value(csv_data, 'cfu_count', None)
        if cfu_count_val is None:
            # Check if cfu_count_measured exists and is 1
            cfu_count_measured_val = get_csv_value(csv_data, 'cfu_count_measured', None)
            if cfu_count_measured_val == 1:
                # It was measured, but we don't know the value. Use placeholder 1.0
                # This ensures cfu_count_measured will be 1 in preprocessing
                cfu_count_val = 1.0
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
                    'catheter_size': catheter_size,  # String format for extraction
                    'catheter_size_fr': get_csv_value(csv_data, 'catheter_size_fr', None),  # Use directly if available from bronze
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
                
                # Always use form data for prediction (form is the source of truth)
                # Apply preprocessing pipeline to form data
                st.info("ℹ️ Using form data for prediction (any changes made to the form will be reflected in the prediction).")
                
                # Apply preprocessing steps in correct order (matching bronze_to_silver pipeline)
                df_processed = apply_outlier_handling(df.copy())
                df_processed = apply_missing_value_handling(df_processed.copy())
                df_processed = apply_binary_encoding(df_processed.copy())
                df_processed = apply_one_hot_encoding(df_processed.copy())  # This converts column names to lowercase
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
                
                # CRITICAL: The scaler was trained on ALL 115 features
                # For the model: Pass all features to scaler (scaler scales everything)
                # For CSV display: Show binary as 0/1, continuous as scaled
                
                # Define continuous columns (from notebook 6_normalize_features.ipynb)
                continuous_cols_expected = [
                    'no_of_invasive_devices', 'heart_rate', 'creatinine', 'charlson_score',
                    'num_of_transfers', 'blood_wbc', 'bp_diastolic', 'bp_systolic',
                    'temperature', 'urinalysis_wbc', 'catheter_size_fr', 'n_catheter_manip_unique_types',
                    'o2sat', 'length_of_stay', 'bmi', 'anchor_age', 'resp_rate', 'urinalysis_rbc'
                ]
                
                # Get the exact feature order the scaler expects
                if hasattr(scaler, 'feature_names_in_'):
                    scaler_feature_order = list(scaler.feature_names_in_)
                    # Use scaler's feature order (most reliable)
                    X_features = [col for col in scaler_feature_order if col not in ['subject_id', 'hadm_id']]
                else:
                    # Fallback: use feature_order from get_feature_order()
                    X_features = [col for col in feature_order if col not in ['subject_id', 'hadm_id']]
                
                # Ensure all features exist in processed data
                for feat in X_features:
                    if feat not in df_processed.columns:
                        df_processed[feat] = 0
                
                # CRITICAL INSIGHT: The model scaler was trained on silver_dataset.csv which has:
                # - Binary columns: 0/1 (raw, not normalized)
                # - Continuous columns: Already normalized (mean≈0, std≈1 from notebook 6)
                #
                # The model scaler's statistics (mean≈0, std≈1 for continuous) confirm this.
                # So we need to normalize continuous columns FIRST using the ORIGINAL (bronze) statistics,
                # then pass all features to the model scaler.
                #
                # However, we don't have the bronze dataset statistics. The solution:
                # We can compute the "reverse" normalization: If silver has normalized data with mean≈0, std≈1,
                # and the model scaler expects that, then we should normalize our raw data to match.
                # But we need the original mean/std from the bronze dataset.
                #
                # TEMPORARY SOLUTION: Try to load bronze dataset to compute statistics
                # If not available, we'll need to use an approximation or save the bronze scaler separately
                
                # Save original binary values before any scaling (for CSV display)
                df_before_scaling = df_processed.copy()
                
                # CRITICAL: We need to normalize continuous columns FIRST using the bronze scaler
                # (like notebook 6 does), then pass to model scaler
                # The bronze scaler normalizes raw values (e.g., anchor_age=85) to normalized values (e.g., anchor_age=1.28)
                # The scaler is fitted on the FULL bronze dataset (many rows) and works with 1 row or many rows
                
                bronze_scaler = None
                try:
                    # First, try to load the pre-saved bronze scaler (faster, no need to load full dataset)
                    # Get project root
                    PROJECT_ROOT = get_project_root()
                    bronze_scaler_path = PROJECT_ROOT / "scripts" / "models" / "bronze_scaler.pkl"
                    if bronze_scaler_path.exists():
                        bronze_scaler = joblib.load(str(bronze_scaler_path))
                        st.info("✅ Loaded pre-fitted bronze scaler (works with 1 row or many rows)")
                    else:
                        # Fallback: Load bronze dataset and fit scaler (slower, but works if scaler not saved)
                        st.warning("⚠️ Bronze scaler not found. Loading dataset to compute scaler (run save_bronze_scaler.py to speed this up)")
                        # PROJECT_ROOT is already defined above
                        bronze_paths = [
                            PROJECT_ROOT / "data" / "silver" / "bronze_one_hot_encoded.csv",  # From notebook 5 (before normalization)
                            PROJECT_ROOT / "data" / "silver" / "bronze_columns_filtered.csv",
                            PROJECT_ROOT / "data" / "bronze" / "bronze_dataset.csv",
                            PROJECT_ROOT / "bronze_columns_filtered.csv"  # Fallback
                        ]
                        bronze_df = None
                        bronze_path_used = None
                        for path in bronze_paths:
                            try:
                                bronze_df = pd.read_csv(str(path))
                                bronze_path_used = str(path)
                                break
                            except:
                                continue
                        
                        if bronze_df is not None:
                            # Handle catheter_size_fr missing values (like notebook 6 does)
                            if 'catheter_size_fr' in bronze_df.columns:
                                bronze_df['catheter_size_fr'] = bronze_df['catheter_size_fr'].fillna(-1)
                            
                            # Get continuous columns that exist in bronze dataset
                            continuous_in_bronze = [col for col in continuous_cols_expected if col in bronze_df.columns]
                            if len(continuous_in_bronze) > 0:
                                from sklearn.preprocessing import StandardScaler
                                bronze_scaler = StandardScaler()
                                bronze_scaler.fit(bronze_df[continuous_in_bronze])
                                st.info(f"✅ Computed bronze scaler from dataset ({len(bronze_df)} rows, {len(continuous_in_bronze)} columns)")
                        else:
                            st.error("❌ Bronze dataset not found. Cannot normalize continuous columns correctly.")
                except Exception as e:
                    st.error(f"❌ Could not load/compute bronze scaler: {e}")
                
                # Step 1: Normalize continuous columns using bronze scaler (like notebook 6)
                # The transform() method works with 1 row or many rows - uses mean/std from full dataset
                if bronze_scaler is not None:
                    continuous_in_processed = [col for col in continuous_cols_expected if col in df_processed.columns]
                    if len(continuous_in_processed) > 0:
                        # Ensure all continuous columns exist in df_processed
                        for col in continuous_in_processed:
                            if col not in df_processed.columns:
                                df_processed[col] = 0
                        
                        # Normalize continuous columns (this is the FIRST normalization, like notebook 6)
                        # Works with 1 row or many rows - uses mean/std from full bronze dataset
                        df_processed[continuous_in_processed] = bronze_scaler.transform(df_processed[continuous_in_processed])
                        st.info(f"✅ Applied bronze scaler to {len(continuous_in_processed)} continuous columns")
                else:
                    st.error("❌ Cannot proceed without bronze scaler. Please run save_bronze_scaler.py first or ensure bronze dataset is available.")
                
                # Reorder columns to match scaler's expected order
                df_final = df_processed.reindex(columns=X_features, fill_value=0)
                
                # Extract features in the correct order
                X_data = df_final[X_features].values
                
                # Model no longer uses scaling - data is already normalized
                # After bronze scaler: continuous columns normalized (mean≈0, std≈1)
                # Binary columns: 0/1
                # This matches the format the model was trained on
                X_scaled = X_data
                
                st.info("✅ Using normalized data directly (no model scaler needed - scaling removed from training)")
                
                # Create properly formatted CSV: binary as 0/1, continuous scaled
                # CRITICAL: The expected CSV format matches silver_dataset.csv:
                # - Binary columns: 0.0 or 1.0 (unchanged, from df_before_scaling)
                # - Continuous columns: Already normalized by bronze scaler (from df_processed after bronze scaler)
                #
                # df_processed now has continuous columns normalized by bronze scaler (like notebook 6)
                # df_before_scaling has original binary values (0/1)
                
                df_for_csv_data = {}
                
                for col in X_features:
                    if col in continuous_cols_expected:
                        # Continuous column - use value from df_processed AFTER bronze scaler normalization
                        # This matches the format in silver_dataset.csv (already normalized)
                        if col in df_processed.columns:
                            # Get normalized value from df_processed (after bronze scaler)
                            normalized_val = float(df_processed[col].iloc[0]) if len(df_processed) > 0 else 0.0
                            df_for_csv_data[col] = normalized_val
                        else:
                            df_for_csv_data[col] = 0.0
                    else:
                        # Binary column - use original 0/1 value from df_before_scaling
                        # (before any scaling, to preserve binary values)
                        if col in df_before_scaling.columns:
                            original_val = df_before_scaling[col].iloc[0] if len(df_before_scaling) > 0 else 0
                            # Ensure it's 0.0 or 1.0
                            clipped_val = np.clip(float(original_val), 0, 1)
                            df_for_csv_data[col] = float(int(round(clipped_val)))
                        else:
                            df_for_csv_data[col] = 0.0
                
                # Create DataFrame with columns in exact order of X_features
                df_for_csv = pd.DataFrame([{col: df_for_csv_data[col] for col in X_features}])
                
                # Debug: Show feature count and normalized data (optional, can be expanded)
                with st.expander("🔍 Debug: Preprocessing Details", expanded=False):
                    st.write(f"**Total features expected:** {len(X_features)}")
                    st.write(f"**Features passed to model:** {X_data.shape[1]}")
                    st.write(f"**Final feature array shape:** {X_scaled.shape}")
                    st.write(f"**First 10 features:** {', '.join(X_features[:10])}")
                    st.info("ℹ️ Note: Continuous columns normalized by bronze scaler, binary columns 0/1. No model scaler needed (scaling removed from training).")
                    
                    # Show sample values for verification
                    st.write("**Sample values (for verification):**")
                    sample_continuous = ['anchor_age', 'bmi', 'charlson_score', 'heart_rate']
                    for col in sample_continuous:
                        if col in df_for_csv.columns:
                            csv_val = df_for_csv[col].iloc[0]
                            bronze_val = df_processed[col].iloc[0] if col in df_processed.columns else None
                            orig_val = df_before_scaling[col].iloc[0] if col in df_before_scaling.columns else None
                            orig_str = f"{orig_val:.2f}" if orig_val is not None else "N/A"
                            bronze_str = f"{bronze_val:.4f}" if bronze_val is not None else "N/A"
                            csv_str = f"{csv_val:.4f}"
                            st.write(f"- {col}: original={orig_str}, after_bronze_scaler={bronze_str}, in_csv={csv_str}")
                
                # Display normalized/preprocessed data as CSV
                # Format: Binary columns as 0/1, only continuous columns scaled
                with st.expander("📊 Normalized/Preprocessed Data (CSV Format)", expanded=False):
                    # Create DataFrame with correct format: binary 0/1, continuous scaled
                    normalized_df = df_for_csv[X_features].copy()
                    
                    # Add subject_id and hadm_id if available
                    if 'subject_id' in df_processed.columns:
                        normalized_df.insert(0, 'subject_id', float(df_processed['subject_id'].values[0]) if len(df_processed) > 0 else None)
                    if 'hadm_id' in df_processed.columns:
                        normalized_df.insert(1, 'hadm_id', float(df_processed['hadm_id'].values[0]) if len(df_processed) > 0 else None)
                    
                    # Display as table
                    st.dataframe(normalized_df, use_container_width=True)
                    
                    # Option to download as CSV
                    csv_string = normalized_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Normalized Data as CSV",
                        data=csv_string,
                        file_name=f"normalized_data_{subject_id}_{hadm_id}.csv" if 'subject_id' in normalized_df.columns and 'hadm_id' in normalized_df.columns else "normalized_data.csv",
                        mime="text/csv"
                    )
                    
                    st.caption("💡 Binary columns (0/1) remain unscaled. Only continuous columns are normalized/scaled. This matches the format from bronze_to_silver preprocessing pipeline.")
                
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

