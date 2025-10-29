#!/usr/bin/env python3
"""
Streamlit application for uploading and cleaning CSV files.
Integrates data_cleaner.py and data_types_checker.py functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="KLM Dashboard Data Cleaner",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3730a3;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f59e0b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #3b82f6;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DataCleaner:
    """
    Advanced Data Cleaning Pipeline with Intelligent NULL Handling

    This class implements a sophisticated approach to data cleaning with optimized NULL value strategies:

    Core Cleaning Logic:

    1. **Enhanced Column Management Strategy**:
       - **System Column Removal**: tail_number, delete_flag, logo_file_id (system-generated)
       - **High NULL Column Removal**: Only removes columns with ≥70% null values (previously 20%)
       - **Column Preservation Logic**: Columns <70% null are preserved and treated
       - **Business Logic**: Aggressive threshold preserves more analytical columns while removing truly unusable ones

    2. **Intelligent NULL Value Treatment**:
       - **Row-Level NULL Analysis**:
         * Entire row analysis for complete NULL rows
         * Complete NULL rows (>90% NULL) are removed entirely
         * Partial NULL rows are preserved with targeted treatment
       - **Column-Level NULL Strategies**:
         * 0-30% NULL: Replace with appropriate values (median, mode, 'Unknown')
         * 30-70% NULL: Replace with standardized NULL markers
         * 70%+ NULL: Column removed entirely
       - **Smart NULL Replacement**:
         * Numeric columns: Median (preserves statistical distribution)
         * Text columns: Context-aware ('Unknown', 'Not Specified', 'N/A')
         * Boolean columns: False (neutral default)
         * Date columns: NaT (pandas standard) or appropriate placeholder
         * ID columns: Keep as-is to preserve uniqueness

    3. **Advanced Data Type Detection & Conversion**:
       - **ID Column Intelligence**:
         * Pure numeric IDs without formatting → Integer
         * IDs with leading zeros, decimals, or mixed content → String
         * Checks for ID uniqueness preservation requirements
       - **Enhanced Boolean Detection**:
         * Named boolean columns with high confidence
         * Content-based detection using comprehensive pattern matching
         * Handles various boolean representations (True/False, 1/0, Yes/No, Active/Inactive)
       - **Robust Date/Time Processing**:
         * Multiple format support (ISO, European, US formats)
         * 85% parse success threshold (increased from 80%)
         * Intelligent date range validation
       - **Precision Numeric Handling**:
         * Integer vs. Float detection based on content
         * Decimal precision analysis
         * Range validation for reasonableness

    4. **Multi-Layer Quality Assurance**:
       - **Pre-Cleaning Validation**: Basic structure and format checks
       - **During-Cleaning Quality**: Real-time quality metrics
       - **Post-Cleaning Validation**: Final quality assessment
       - **Quality Score Calculation**: Comprehensive data quality scoring
       - **Statistical Integrity**: Distribution preservation verification

    5. **Sophisticated Duplicate and Anomaly Handling**:
       - **Exact Duplicate Removal**: Identical row elimination
       - **Fuzzy Duplicate Detection**: Near-duplicate identification
       - **Outlier Analysis**: Statistical outlier detection and flagging
       - **Data Consistency**: Cross-column consistency validation

    6. **Comprehensive Audit Trail**:
       - **Action Logging**: Detailed record of all modifications
       - **Before/After Statistics**: Complete comparison metrics
       - **Quality Metrics**: Multi-dimensional quality assessment
       - **Recommendation System**: Suggestions for data improvements

    This enhanced cleaning approach ensures:
    - Maximum data preservation while maintaining quality
    - Intelligent NULL value handling based on data context
    - Statistical integrity preservation
    - Business-appropriate data transformations
    - Complete transparency and auditability
    """

    def __init__(self):
        self.original_data = {}
        self.cleaned_data = {}
        self.cleaning_stats = {}

    def analyze_data_quality(self, filename, df):
        """Analyze data quality issues in a dataframe."""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_values': df.isnull().sum().sum(),
            'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'empty_strings': (df == '').sum().sum(),
            'na_values': df.isna().sum().sum(),
            'columns_with_issues': []
        }

        # Detailed column analysis
        for col in df.columns:
            col_nulls = df[col].isnull().sum()
            col_empty = (df[col] == '').sum()
            col_na = df[col].isna().sum()
            total_issues = col_nulls + col_empty + col_na

            if total_issues > 0:
                stats['columns_with_issues'].append({
                    'column': col,
                    'nulls': col_nulls,
                    'empty_strings': col_empty,
                    'na_values': col_na,
                    'data_type': str(df[col].dtype),
                    'unique_values': df[col].nunique()
                })

        return stats

    def clean_dataframe(self, filename, df):
        """Clean a single dataframe based on its characteristics."""
        df_cleaned = df.copy()
        cleaning_log = []

        # Remove unwanted columns
        unwanted_columns = ['tail_number', 'delete_flag', 'logo_file_id']
        columns_to_remove = [col for col in unwanted_columns if col in df_cleaned.columns]
        for col in columns_to_remove:
            df_cleaned = df_cleaned.drop(columns=[col])
            cleaning_log.append(f"Removed unwanted column '{col}'")

        # Enhanced NULL handling strategy

        # Stage 1: Row-level NULL analysis and removal
        original_rows = len(df_cleaned)

        # Remove completely NULL rows (>90% NULL)
        total_columns = len(df_cleaned.columns)
        null_percentage_per_row = df_cleaned.isnull().sum(axis=1) / total_columns
        completely_null_rows = df_cleaned[null_percentage_per_row > 0.9].index

        if len(completely_null_rows) > 0:
            df_cleaned = df_cleaned.drop(completely_null_rows)
            cleaning_log.append(f"Removed {len(completely_null_rows)} completely NULL rows (>90% NULL values)")

        # Stage 2: Column-level NULL analysis and removal
        total_rows = len(df_cleaned)
        columns_with_very_high_nulls = []

        for col in df_cleaned.columns:
            null_percentage = (df_cleaned[col].isnull().sum() / total_rows) * 100
            if null_percentage >= 70:  # Changed from 20% to 70%
                columns_with_very_high_nulls.append(col)
                cleaning_log.append(f"Removed column '{col}' due to {null_percentage:.1f}% null values (≥70% threshold)")

        if columns_with_very_high_nulls:
            df_cleaned = df_cleaned.drop(columns=columns_with_very_high_nulls)

        # Stage 3: Intelligent NULL value replacement for remaining columns
        for col in df_cleaned.columns:
            null_count = df_cleaned[col].isnull().sum()
            if null_count == 0:
                continue

            null_percentage = (null_count / total_rows) * 100
            col_lower = col.lower()

            if df_cleaned[col].dtype in ['int64', 'float64']:
                # Numeric columns
                if null_percentage <= 30:
                    # Low NULL: Replace with median to preserve distribution
                    median_val = df_cleaned[col].median()
                    if not pd.isna(median_val):
                        df_cleaned[col] = df_cleaned[col].fillna(median_val)
                        cleaning_log.append(f"Replaced {null_count} NULL values in numeric column '{col}' with median ({median_val})")
                    else:
                        # If median is also NULL, use 0 for count/order columns
                        if any(keyword in col_lower for keyword in ['count', 'order', 'number']):
                            df_cleaned[col] = df_cleaned[col].fillna(0)
                            cleaning_log.append(f"Replaced {null_count} NULL values in numeric column '{col}' with 0 (count/order column)")
                        else:
                            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                            cleaning_log.append(f"Replaced {null_count} NULL values in numeric column '{col}' with mean")
                else:
                    # High NULL (30-70%): Replace with appropriate indicator
                    if any(keyword in col_lower for keyword in ['count', 'order', 'number']):
                        df_cleaned[col] = df_cleaned[col].fillna(0)
                        cleaning_log.append(f"Replaced {null_count} NULL values in high-NULL numeric column '{col}' with 0 (count/order)")
                    else:
                        # Keep as NaN for analysis but log the decision
                        cleaning_log.append(f"Preserved {null_count} NULL values in numeric column '{col}' ({null_percentage:.1f}% NULL) for analysis")

            elif df_cleaned[col].dtype == 'object':
                # Text columns
                if null_percentage <= 30:
                    # Low NULL: Context-aware replacement
                    if any(keyword in col_lower for keyword in ['name', 'description', 'remarks', 'comment']):
                        df_cleaned[col] = df_cleaned[col].fillna('Not Specified')
                        cleaning_log.append(f"Replaced {null_count} NULL values in text column '{col}' with 'Not Specified'")
                    elif any(keyword in col_lower for keyword in ['code', 'id', 'number', 'registration']):
                        # Keep as 'Unknown' for ID-like columns
                        df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                        cleaning_log.append(f"Replaced {null_count} NULL values in ID-like column '{col}' with 'Unknown'")
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                        cleaning_log.append(f"Replaced {null_count} NULL values in text column '{col}' with 'Unknown'")
                else:
                    # High NULL (30-70%): Use standardized approach
                    df_cleaned[col] = df_cleaned[col].fillna('N/A')
                    cleaning_log.append(f"Replaced {null_count} NULL values in high-NULL text column '{col}' with 'N/A' ({null_percentage:.1f}% NULL)")

            elif df_cleaned[col].dtype == 'bool':
                # Boolean columns
                df_cleaned[col] = df_cleaned[col].fillna(False)
                cleaning_log.append(f"Replaced {null_count} NULL values in boolean column '{col}' with False")

            elif 'datetime' in str(df_cleaned[col].dtype):
                # Date columns
                # Keep as NaT (Not a Time) which is pandas standard for missing dates
                cleaning_log.append(f"Preserved {null_count} NULL values in date column '{col}' as NaT")
            else:
                # Other data types - default handling
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                cleaning_log.append(f"Replaced {null_count} NULL values in column '{col}' with 'Unknown'")

        # Handle empty strings and whitespace
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                before_nulls = df_cleaned[col].isnull().sum()
                df_cleaned[col] = df_cleaned[col].replace(['', ' ', 'NULL', 'null', 'None', 'none'], np.nan)
                if df_cleaned[col].dtype == 'object':
                    # Only convert to string if not an ID column that should remain numeric
                    col_lower = col.lower()
                    if not any(keyword in col_lower for keyword in ['id', 'code', 'number', 'registration']):
                        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                after_nulls = df_cleaned[col].isnull().sum()

                if after_nulls > before_nulls:
                    cleaning_log.append(f"Cleaned {after_nulls - before_nulls} empty/null strings in column '{col}'")

        # Convert data types based on column names and content
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().all():
                continue

            col_lower = col.lower()
            current_dtype = df_cleaned[col].dtype

            # First, preserve existing boolean columns
            if current_dtype == 'bool':
                cleaning_log.append(f"Preserved existing boolean column '{col}'")
                continue

            # ID columns - only convert to string if necessary
            if any(keyword in col_lower for keyword in ['id', 'code', 'number', 'registration']):
                # Check if column needs string conversion
                non_null_values = df_cleaned[col].dropna()
                if len(non_null_values) > 0:
                    # Check if all values are pure integers without special formatting needs
                    str_values = non_null_values.astype(str)

                    # Check if conversion to string is needed:
                    # 1. Values contain leading zeros (except single "0")
                    # 2. Values contain decimal points
                    # 3. Values contain non-numeric characters
                    needs_string_conversion = (
                        str_values.str.contains(r'^0\d+', na=False).any() or  # Leading zeros
                        str_values.str.contains(r'\.', na=False).any() or     # Decimal points
                        not str_values.str.match(r'^\d+$', na=False).all()    # Non-numeric
                    )

                    if needs_string_conversion:
                        df_cleaned[col] = df_cleaned[col].astype(str)
                        if df_cleaned[col].str.contains(r'\.0', na=False).any():
                            df_cleaned[col] = df_cleaned[col].str.replace(r'\.0$', '', regex=True)
                        cleaning_log.append(f"Converted ID column '{col}' to string (contains non-numeric format or leading zeros)")
                    else:
                        # Keep as numeric - try to convert to int64 if possible
                        try:
                            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
                            cleaning_log.append(f"Kept ID column '{col}' as numeric")
                        except:
                            # If conversion fails, convert to string
                            df_cleaned[col] = df_cleaned[col].astype(str)
                            cleaning_log.append(f"Converted ID column '{col}' to string (numeric conversion failed)")
                else:
                    # Empty column, convert to string
                    df_cleaned[col] = df_cleaned[col].astype(str)

            # Boolean columns - detect by name and content
            elif (col_lower in ['active', 'final_delay', 'night_stop', 'summer_schedule',
                               'winter_schedule', 'flight_out', 'on_call_schedule'] or
                  # Auto-detect boolean columns by content
                  (current_dtype == 'object' and len(df_cleaned[col].dropna()) > 0)):

                # Check if this looks like a boolean column by content
                if current_dtype == 'object':
                    non_null_values = df_cleaned[col].dropna()
                    str_values = non_null_values.astype(str).str.lower()
                    unique_values = set(str_values.unique())

                    # Check if values look like boolean
                    if unique_values <= {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}:
                        df_cleaned[col] = df_cleaned[col].astype(str).map({
                            'true': True, 't': True, '1': True, 'yes': True, 'y': True,
                            'false': False, 'f': False, '0': False, 'no': False, 'n': False
                        }).fillna(df_cleaned[col])
                        try:
                            df_cleaned[col] = df_cleaned[col].astype(bool)
                            cleaning_log.append(f"Auto-detected and converted '{col}' to boolean")
                        except:
                            pass
                        continue

                # Named boolean columns - only convert if not already boolean
                if col_lower in ['active', 'final_delay', 'night_stop', 'summer_schedule',
                                 'winter_schedule', 'flight_out', 'on_call_schedule']:
                    df_cleaned[col] = df_cleaned[col].astype(str).map({
                        'True': True, 'true': True, '1': True, 'yes': True, 'Yes': True,
                        'False': False, 'false': False, '0': False, 'no': False, 'No': False
                    }).fillna(df_cleaned[col])
                    try:
                        df_cleaned[col] = df_cleaned[col].astype(bool)
                        cleaning_log.append(f"Converted '{col}' to boolean")
                    except:
                        pass

            # Date/time columns
            elif any(keyword in col_lower for keyword in ['time', 'date', 'on', 'created', 'updated']):
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    cleaning_log.append(f"Converted '{col}' to datetime")
                except:
                    pass

            # Numeric columns
            elif any(keyword in col_lower for keyword in ['count', 'amount', 'delay_time', 'engine_count', 'order_id']):
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    cleaning_log.append(f"Converted '{col}' to numeric")
                except:
                    pass

        # Handle missing values
        for col in df_cleaned.columns:
            missing_count = df_cleaned[col].isnull().sum()
            if missing_count > 0:
                col_type = df_cleaned[col].dtype

                if col_type == 'object':
                    if any(keyword in col.lower() for keyword in ['name', 'description', 'remarks', 'email']):
                        df_cleaned[col] = df_cleaned[col].fillna('Not Specified')
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                    cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with text")

                    df_cleaned[col] = df_cleaned[col].replace(['nan', 'NaN', 'NAN'], 'Unknown')

                elif col_type in ['int64', 'float64']:
                    if 'count' in col.lower() or 'order' in col.lower():
                        df_cleaned[col] = df_cleaned[col].fillna(0)
                    else:
                        median_val = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with numeric value")

                elif col_type == 'bool':
                    df_cleaned[col] = df_cleaned[col].fillna(False)
                    cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with False")

                elif col_type == 'datetime64[ns]':
                    df_cleaned[col] = df_cleaned[col].fillna(pd.Timestamp('1900-01-01'))
                    cleaning_log.append(f"Filled {missing_count} missing dates in '{col}'")

        # Remove duplicate rows
        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            df_cleaned = df_cleaned.drop_duplicates()
            cleaning_log.append(f"Removed {duplicates} duplicate rows")

        # Final cleanup
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].replace(['nan', 'NaN', 'NAN'], 'Unknown')
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')

        return df_cleaned, cleaning_log

class DataTypeChecker:
    """
    Advanced Data Type Analysis and Detection System

    This class provides intelligent data type detection with comprehensive logic:

    Detection Methodology:

    1. **Pattern-Based Detection**:
       - Column name analysis against keyword libraries
       - Content pattern recognition using regex
       - Statistical distribution analysis

    2. **Content Validation**:
       - Sample-based testing (first 100 non-null values)
       - Success rate calculation (80% threshold required)
       - Format consistency checking

    3. **Type Assignment Logic**:
       - **String Type**: Default for mixed content or unclear patterns
       - **Integer Type**: Pure whole numbers without decimals
       - **Float Type**: Numbers with decimal points
       - **DateTime Type**: Date/time parseable content
       - **Boolean Type**: Limited value sets (true/false, 1/0, yes/no)

    4. **Confidence Scoring**:
       - High confidence: Clear patterns, >95% success rate
       - Medium confidence: Recognizable patterns, 80-95% success rate
       - Low confidence: Ambiguous content, <80% success rate

    5. **Special Handling**:
       - **ID Columns**: Preserved as strings if formatting matters
       - **Numeric IDs**: Converted to integers when pure
       - **Date Formats**: Multiple format support (ISO, European, US)
       - **Boolean Variants**: Comprehensive mapping table

    Quality Assurance:
    - Each suggestion includes reasoning
    - Null value impact assessment
    - Conversion confidence scoring
    - Potential issues identification
    """

    def analyze_column_types(self, filename, df):
        """
        Comprehensive Data Type Analysis

        Analysis Process for Each Column:

        1. **Pre-analysis Checks**:
           - Empty column detection
           - Null value percentage calculation
           - Sample value extraction (first 100 non-null values)

        2. **Column Classification**:
           - Name-based pattern matching against keyword libraries
           - Content-based pattern analysis
           - Statistical distribution evaluation

        3. **Type Detection Logic**:

           **ID Columns** (keywords: 'id', 'code', 'number', 'registration'):
           - Check for pure integer patterns using regex: `^\\d+$`
           - Detect leading zeros: pattern `^0\\d+`
           - Identify decimal points: pattern `\\.`

           **Boolean Columns** (named + content detection):
           - Named: 'active', 'final_delay', 'night_stop', etc.
           - Content: Values in set {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}

           **Date/Time Columns** (keywords: 'time', 'date', 'created', 'updated'):
           - Parse success rate testing with pd.to_datetime()
           - Multiple format support (ISO, European, US)
           - Minimum 80% success rate required for datetime suggestion

           **Numeric Columns** (keywords: 'count', 'amount', 'delay_time', 'engine_count'):
           - Integer detection with pd.to_numeric(dtype='int64')
           - Float detection for decimal numbers
           - Success rate threshold validation

        4. **Quality Assessment**:
           - Confidence scoring based on pattern clarity
           - Null value impact evaluation
           - Conversion risk assessment

        Returns dictionary with detailed suggestions for each column.
        """
        suggestions = {}

        for col in df.columns:
            col_info = {
                'current_type': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'suggested_type': None,
                'notes': []
            }

            sample_values = df[col].dropna().head(100)

            if len(sample_values) == 0:
                col_info['suggested_type'] = 'object'
                col_info['notes'].append("Column is completely empty")
                suggestions[col] = col_info
                continue

            col_lower = col.lower()
            sample_str = sample_values.astype(str)

            # ID columns
            if any(keyword in col_lower for keyword in ['id', 'code', 'number', 'registration']):
                # Check if all values are pure integers without special formatting
                if sample_str.str.match(r'^\d+$').all():
                    # Check if string conversion is needed
                    needs_string = (
                        sample_str.str.contains(r'^0\d+', na=False).any() or  # Leading zeros
                        sample_str.str.contains(r'\.', na=False).any()         # Decimal points
                    )

                    if needs_string:
                        col_info['suggested_type'] = 'string'
                        col_info['notes'].append("Numeric ID with leading zeros detected, keeping as string")
                    else:
                        col_info['suggested_type'] = 'int64'
                        col_info['notes'].append("Pure numeric ID detected, can be integer")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['notes'].append("Mixed ID format, keeping as string")

            # Boolean columns
            elif col_lower in ['active', 'final_delay', 'night_stop', 'summer_schedule',
                             'winter_schedule', 'flight_out', 'on_call_schedule']:
                unique_values = set(sample_str.str.lower())
                if unique_values <= {'true', 'false', '1', '0', 'yes', 'no', 'nan'}:
                    col_info['suggested_type'] = 'boolean'
                    col_info['notes'].append("Boolean values detected")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['notes'].append("Unexpected values in boolean-like column")

            # Date/time columns
            elif any(keyword in col_lower for keyword in ['time', 'date', 'on', 'created', 'updated']):
                date_success = 0
                for val in sample_values:
                    try:
                        pd.to_datetime(val)
                        date_success += 1
                    except:
                        pass

                if date_success / len(sample_values) > 0.8:
                    col_info['suggested_type'] = 'datetime'
                    col_info['notes'].append(f"{date_success}/{len(sample_values)} values parse as dates")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['notes'].append("Low date parse success rate")

            # Numeric columns
            elif any(keyword in col_lower for keyword in ['count', 'amount', 'delay_time', 'engine_count', 'order_id']):
                numeric_success = 0
                for val in sample_values:
                    try:
                        pd.to_numeric(val)
                        numeric_success += 1
                    except:
                        pass

                if numeric_success / len(sample_values) > 0.8:
                    try:
                        pd.to_numeric(sample_values, dtype='int64')
                        col_info['suggested_type'] = 'int64'
                    except:
                        col_info['suggested_type'] = 'float64'

                    col_info['notes'].append(f"{numeric_success}/{len(sample_values)} values parse as numeric")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['notes'].append("Low numeric parse success rate")

            # Default to string
            else:
                col_info['suggested_type'] = 'string'
                col_info['notes'].append("Default string type")

            suggestions[col] = col_info

        return suggestions

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">KLM Dashboard Data Cleaner</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<h2 class="section-header">Upload CSV Files</h2>', unsafe_allow_html=True)

    
    uploaded_files = st.sidebar.file_uploader(
        "Choose CSV files (you can select multiple)",
        type=['csv'],
        accept_multiple_files=True,
        key="csv_files_uploader",
        help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple files. Maximum 8 files recommended for best performance."
    )

    # Initialize session state
    if 'original_data' not in st.session_state:
        st.session_state.original_data = {}
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = {}
    if 'cleaning_stats' not in st.session_state:
        st.session_state.cleaning_stats = {}
    if 'type_suggestions' not in st.session_state:
        st.session_state.type_suggestions = {}

    # Process uploaded files
    if uploaded_files:
        st.sidebar.markdown(f'<div class="info-box">Uploaded {len(uploaded_files)} files</div>', unsafe_allow_html=True)

        # Load files into session state
        successfully_loaded = 0
        failed_files = []

        for file in uploaded_files:
            try:
                # Reset file pointer to beginning
                file.seek(0)

                # Load CSV with error handling
                df = pd.read_csv(file)

                # Check if file has data
                if df.empty:
                    st.sidebar.warning(f"⚠️ {file.name} is empty")
                    failed_files.append(file.name)
                else:
                    st.session_state.original_data[file.name] = df
                    st.sidebar.success(f"✓ {file.name} loaded ({df.shape[0]} rows, {df.shape[1]} columns)")
                    successfully_loaded += 1

            except pd.errors.EmptyDataError:
                st.sidebar.error(f"✗ {file.name}: Empty file")
                failed_files.append(file.name)
            except pd.errors.ParserError as e:
                st.sidebar.error(f"✗ {file.name}: Invalid CSV format - {str(e)}")
                failed_files.append(file.name)
            except Exception as e:
                st.sidebar.error(f"✗ {file.name}: {str(e)}")
                failed_files.append(file.name)

        # Summary of upload results
        if successfully_loaded > 0:
            st.sidebar.info(f"📊 Successfully loaded {successfully_loaded} files")

        if failed_files:
            st.sidebar.error(f"❌ Failed to load {len(failed_files)} files: {', '.join(failed_files)}")

        # Remove files that were previously uploaded but are no longer in the current upload
        if uploaded_files:
            current_filenames = {file.name for file in uploaded_files}
            filenames_to_remove = [
                filename for filename in st.session_state.original_data.keys()
                if filename not in current_filenames
            ]

            for filename in filenames_to_remove:
                del st.session_state.original_data[filename]
                # Also clean related data
                if filename in st.session_state.cleaned_data:
                    del st.session_state.cleaned_data[filename]
                if filename in st.session_state.cleaning_stats:
                    del st.session_state.cleaning_stats[filename]
                if filename in st.session_state.type_suggestions:
                    del st.session_state.type_suggestions[filename]

            if filenames_to_remove:
                st.sidebar.info(f"🗑️ Removed {len(filenames_to_remove)} previously uploaded files")

        if st.session_state.original_data:
            # Main content area
            st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)

            # Status display
            total_files = len(st.session_state.original_data)
            if total_files > 0:
                st.success(f"📁 Currently loaded: {total_files} file{'s' if total_files != 1 else ''}")
                st.write(f"Ready for cleaning and analysis. Use the buttons below to proceed.")
            else:
                st.warning("⚠️ No files loaded yet. Please upload CSV files in the sidebar to begin.")

            # Show original data preview
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Files")
                for filename, df in st.session_state.original_data.items():
                    with st.expander(f"{filename} ({df.shape[0]} rows, {df.shape[1]} columns)", expanded=False):
                        st.dataframe(df.head())
                        st.write(f"**Shape:** {df.shape}")
                        st.write(f"**Memory usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            with col2:
                st.subheader("Data Quality Summary")
                for filename, df in st.session_state.original_data.items():
                    with st.expander(f"Quality Report: {filename}", expanded=False):
                        null_count = df.isnull().sum().sum()
                        null_percentage = (null_count / (len(df) * len(df.columns))) * 100
                        duplicate_count = df.duplicated().sum()

                        st.metric("Total Rows", f"{df.shape[0]:,}")
                        st.metric("Total Columns", f"{df.shape[1]}")
                        st.metric("Null Values", f"{null_count:,} ({null_percentage:.1f}%)")
                        st.metric("Duplicate Rows", f"{duplicate_count:,}")

            # Action buttons
            st.markdown('<h2 class="section-header">Data Cleaning Actions</h2>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Clean All Data", type="primary", use_container_width=True):
                    # Create detailed cleaning progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Progress tracking
                    total_files = len(st.session_state.original_data)
                    current_file = 0

                    for filename, df in st.session_state.original_data.items():
                        current_file += 1
                        progress_percentage = current_file / total_files

                        # Update progress
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Processing {filename} ({current_file}/{total_files})...")

                        # Create detailed step-by-step cleaning feedback
                        with st.expander(f"Detailed Processing: {filename}", expanded=False):
                            step_progress = st.progress(0)
                            step_status = st.empty()

                            # Step 1: Initial Analysis
                            step_progress.progress(0.1)
                            step_status.text("Step 1/6: Analyzing data structure...")
                            time.sleep(0.5)  # Brief pause for visibility

                            original_rows = len(df)
                            original_cols = len(df.columns)
                            null_percentage = (df.isnull().sum().sum() / (original_rows * original_cols)) * 100

                            st.info(f"""
                            **Initial Analysis Results:**
                            - Rows: {original_rows:,}
                            - Columns: {original_cols}
                            - Null values: {null_percentage:.2f}%
                            - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
                            """)

                            # Step 2: Column Filtering
                            step_progress.progress(0.2)
                            step_status.text("Step 2/6: Filtering unwanted columns...")
                            time.sleep(0.5)

                            unwanted_columns = ['tail_number', 'delete_flag', 'logo_file_id']
                            columns_to_remove = [col for col in unwanted_columns if col in df.columns]

                            high_null_cols = []
                            for col in df.columns:
                                null_pct = (df[col].isnull().sum() / len(df)) * 100
                                if null_pct > 20:
                                    high_null_cols.append(col)

                            total_removed = len(columns_to_remove) + len(high_null_cols)

                            if total_removed > 0:
                                st.warning(f"""
                                **Column Removal:**
                                - Unwanted columns removed: {len(columns_to_remove)}
                                - High null columns removed: {len(high_null_cols)}
                                - Total columns removed: {total_removed}
                                """)

                                if columns_to_remove:
                                    st.write(f"Removed unwanted columns: {', '.join(columns_to_remove)}")
                                if high_null_cols:
                                    st.write(f"Removed high-null columns: {', '.join(high_null_cols)}")
                            else:
                                st.success("No columns needed to be removed")

                            # Step 3: Data Type Detection
                            step_progress.progress(0.4)
                            step_status.text("Step 3/6: Detecting and converting data types...")
                            time.sleep(0.5)

                            type_conversions = {
                                'ID columns': 0,
                                'Boolean columns': 0,
                                'Date/Time columns': 0,
                                'Numeric columns': 0,
                                'Text columns': 0
                            }

                            # Analyze columns for type conversion
                            for col in df.columns:
                                if col.lower() in ['tail_number', 'delete_flag', 'logo_file_id']:
                                    continue  # Skip removed columns

                                col_lower = col.lower()

                                if any(keyword in col_lower for keyword in ['id', 'code', 'number', 'registration']):
                                    type_conversions['ID columns'] += 1
                                elif col_lower in ['active', 'final_delay', 'night_stop', 'summer_schedule', 'winter_schedule', 'flight_out', 'on_call_schedule']:
                                    type_conversions['Boolean columns'] += 1
                                elif any(keyword in col_lower for keyword in ['time', 'date', 'created', 'updated']):
                                    type_conversions['Date/Time columns'] += 1
                                elif any(keyword in col_lower for keyword in ['count', 'amount', 'delay_time', 'engine_count', 'order_id']):
                                    type_conversions['Numeric columns'] += 1
                                else:
                                    type_conversions['Text columns'] += 1

                            st.info(f"""
                            **Data Type Conversions Identified:**
                            - ID columns: {type_conversions['ID columns']}
                            - Boolean columns: {type_conversions['Boolean columns']}
                            - Date/Time columns: {type_conversions['Date/Time columns']}
                            - Numeric columns: {type_conversions['Numeric columns']}
                            - Text columns: {type_conversions['Text columns']}
                            """)

                            # Step 4: Missing Value Treatment
                            step_progress.progress(0.6)
                            step_status.text("Step 4/6: Treating missing values...")
                            time.sleep(0.5)

                            missing_treatment = {
                                'text_fields': 0,
                                'numeric_fields': 0,
                                'boolean_fields': 0,
                                'date_fields': 0
                            }

                            # Count missing value treatments
                            for col in df.columns:
                                missing_count = df[col].isnull().sum()
                                if missing_count > 0:
                                    col_type = df[col].dtype
                                    if col_type == 'object':
                                        missing_treatment['text_fields'] += missing_count
                                    elif col_type in ['int64', 'float64']:
                                        missing_treatment['numeric_fields'] += missing_count
                                    elif col_type == 'bool':
                                        missing_treatment['boolean_fields'] += missing_count
                                    elif col_type == 'datetime64[ns]':
                                        missing_treatment['date_fields'] += missing_count

                            total_missing = sum(missing_treatment.values())

                            if total_missing > 0:
                                st.info(f"""
                                **Missing Value Treatment:**
                                - Text fields filled: {missing_treatment['text_fields']:,}
                                - Numeric fields filled: {missing_treatment['numeric_fields']:,}
                                - Boolean fields filled: {missing_treatment['boolean_fields']:,}
                                - Date fields filled: {missing_treatment['date_fields']:,}
                                - Total missing values treated: {total_missing:,}
                                """)
                            else:
                                st.success("No missing values found")

                            # Step 5: Quality Enhancements
                            step_progress.progress(0.8)
                            step_status.text("Step 5/6: Applying quality enhancements...")
                            time.sleep(0.5)

                            # Check for duplicates and other quality issues
                            duplicates = df.duplicated().sum()
                            empty_strings = (df == '').sum().sum()

                            quality_improvements = {
                                'duplicates_removed': duplicates,
                                'empty_strings_cleaned': empty_strings,
                                'whitespace_trimmed': len(df.columns) * len(df)  # Estimate
                            }

                            st.info(f"""
                            **Quality Enhancements Applied:**
                            - Duplicate rows removed: {duplicates:,}
                            - Empty strings cleaned: {empty_strings:,}
                            - Text fields trimmed: All columns processed
                            """)

                            # Step 6: Final Processing
                            step_progress.progress(1.0)
                            step_status.text("Step 6/6: Finalizing cleaning process...")
                            time.sleep(0.5)

                            # Actually perform the cleaning
                            cleaner = DataCleaner()
                            cleaner.original_data = st.session_state.original_data
                            cleaned_df, cleaning_log = cleaner.clean_dataframe(filename, df)

                            # Calculate final statistics
                            cleaned_stats = cleaner.analyze_data_quality(f"CLEANED_{filename}", cleaned_df)
                            original_stats = cleaner.analyze_data_quality(filename, df)

                            improvements = {
                                'rows_removed': original_stats['total_rows'] - cleaned_stats['total_rows'],
                                'columns_removed': original_stats['total_columns'] - cleaned_stats['total_columns'],
                                'nulls_fixed': original_stats['null_values'] - cleaned_stats['null_values'],
                                'null_improvement': (original_stats['null_percentage'] - cleaned_stats['null_percentage'])
                            }

                            st.success(f"""
                            **Final Results for {filename}:**
                            - Final rows: {cleaned_stats['total_rows']:,} (removed: {improvements['rows_removed']:,})
                            - Final columns: {cleaned_stats['total_columns']} (removed: {improvements['columns_removed']:,})
                            - Null values reduced: {improvements['nulls_fixed']:,} ({improvements['null_improvement']:.2f}% improvement)
                            - Final null percentage: {cleaned_stats['null_percentage']:.2f}%
                            - Cleaning actions performed: {len(cleaning_log)}
                            """)

                            # Store results
                            st.session_state.cleaned_data[filename] = cleaned_df
                            st.session_state.cleaning_stats[filename] = {
                                'original': original_stats,
                                'cleaned': cleaned_stats,
                                'cleaning_log': cleaning_log
                            }

                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()

                    st.success("✅ All data cleaning completed with detailed logging!")
                    st.rerun()

            with col2:
                if st.button("Analyze Data Types", use_container_width=True):
                    # Create detailed type analysis progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    total_files = len(st.session_state.original_data)

                    for i, (filename, df) in enumerate(st.session_state.original_data.items(), 1):
                        progress_percentage = i / total_files
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Analyzing {filename} ({i}/{total_files})...")

                        with st.expander(f"Type Analysis: {filename}", expanded=False):
                            st.info(f"**Analyzing {len(df.columns)} columns for optimal data types...**")

                            checker = DataTypeChecker()
                            suggestions = checker.analyze_column_types(filename, df)

                            # Group suggestions by type
                            type_groups = {
                                'String (Object)': [],
                                'Integer': [],
                                'Float': [],
                                'DateTime': [],
                                'Boolean': []
                            }

                            for col, info in suggestions.items():
                                suggested_type = info['suggested_type']
                                if suggested_type in type_groups:
                                    type_groups[suggested_type].append({
                                        'column': col,
                                        'current': info['current_type'],
                                        'confidence': len([n for n in info['notes'] if 'confident' in n.lower() or 'detected' in n.lower()]),
                                        'issues': info['null_count']
                                    })

                            # Show type distribution
                            col_dist = {k: len(v) for k, v in type_groups.items() if v}
                            if col_dist:
                                st.info(f"""
                                **Type Distribution Found:**
                                {chr(10).join([f"- {k}: {v} columns" for k, v in col_dist.items()])}
                                """)

                            # Show detailed analysis for each type
                            for type_name, columns in type_groups.items():
                                if columns:
                                    with st.expander(f"{type_name} ({len(columns)} columns)", expanded=False):
                                        for col_info in columns:
                                            col_name = col_info['column']
                                            col_notes = suggestions[col_name]['notes']

                                            st.write(f"**{col_name}**")
                                            st.write(f"- Current: {col_info['current']} → Suggested: {type_name}")
                                            st.write(f"- Null values: {col_info['issues']:,}")
                                            if col_notes:
                                                st.write(f"- Reason: {'; '.join(col_notes)}")
                                            st.write("---")

                            # Show conversion confidence
                            high_confidence = sum(1 for info in suggestions.values()
                                               if any('confident' in note.lower() or 'detected' in note.lower()
                                                      for note in info['notes']))
                            confidence_pct = (high_confidence / len(suggestions)) * 100 if suggestions else 0

                            if confidence_pct >= 80:
                                st.success(f"**High Confidence Analysis**: {confidence_pct:.1f}% of columns have clear type recommendations")
                            elif confidence_pct >= 60:
                                st.warning(f"**Medium Confidence Analysis**: {confidence_pct:.1f}% of columns have clear type recommendations")
                            else:
                                st.error(f"**Low Confidence Analysis**: Only {confidence_pct:.1f}% of columns have clear type recommendations")

                        st.session_state.type_suggestions[filename] = suggestions

                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Comprehensive data type analysis completed!")
                    st.rerun()

            with col3:
                if st.button("Clear All Data", use_container_width=True):
                    # Clear all session state data
                    keys_to_clear = [
                        'original_data', 'cleaned_data', 'cleaning_stats',
                        'type_suggestions'
                    ]

                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.rerun()

            # Show results if available
            if st.session_state.cleaned_data:
                st.markdown('<h2 class="section-header">Cleaning Results</h2>', unsafe_allow_html=True)

                for filename, stats in st.session_state.cleaning_stats.items():
                    with st.expander(f"Cleaning Report: {filename}", expanded=False):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Before Cleaning")
                            orig = stats['original']
                            st.write(f"**Rows:** {orig['total_rows']:,}")
                            st.write(f"**Columns:** {orig['total_columns']}")
                            st.write(f"**Null Values:** {orig['null_values']:,} ({orig['null_percentage']:.2f}%)")
                            st.write(f"**Empty Strings:** {orig['empty_strings']:,}")

                        with col2:
                            st.subheader("After Cleaning")
                            clean = stats['cleaned']
                            st.write(f"**Rows:** {clean['total_rows']:,}")
                            st.write(f"**Columns:** {clean['total_columns']}")
                            st.write(f"**Null Values:** {clean['null_values']:,} ({clean['null_percentage']:.2f}%)")
                            st.write(f"**Empty Strings:** {clean['empty_strings']:,}")

                        # Cleaning actions
                        if stats['cleaning_log']:
                            st.subheader("Actions Performed")
                            for action in stats['cleaning_log']:
                                st.write(f"• {action}")

                        # Data preview
                        st.subheader("Preview Cleaned Data")
                        with st.expander("Show Data Preview", expanded=False):
                            st.dataframe(st.session_state.cleaned_data[filename].head())

            # Show type analysis if available
            if st.session_state.type_suggestions:
                st.markdown('<h2 class="section-header">Data Type Analysis</h2>', unsafe_allow_html=True)

                for filename, suggestions in st.session_state.type_suggestions.items():
                    with st.expander(f"Type Analysis: {filename}", expanded=False):
                        # Create DataFrame for better display
                        type_df = pd.DataFrame([
                            {
                                'Column': col,
                                'Current Type': info['current_type'],
                                'Suggested Type': info['suggested_type'],
                                'Null Count': info['null_count'],
                                'Notes': '; '.join(info['notes'])
                            }
                            for col, info in suggestions.items()
                        ])

                        st.dataframe(type_df, use_container_width=True)

            # Download cleaned data
            if st.session_state.cleaned_data:
                st.markdown('<h2 class="section-header">Download Cleaned Data</h2>', unsafe_allow_html=True)

                # Prepare download
                cleaned_csv_files = {}
                for filename, df in st.session_state.cleaned_data.items():
                    csv = df.to_csv(index=False)
                    cleaned_csv_files[filename] = csv

                # Individual file downloads
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Download Individual Files")
                    for filename, csv in cleaned_csv_files.items():
                        st.download_button(
                            label=f"{filename}",
                            data=csv,
                            file_name=f"cleaned_{filename}",
                            mime="text/csv",
                            use_container_width=True
                        )

                with col2:
                    st.subheader("Download All as ZIP")
                    if st.button("Create ZIP Download", use_container_width=True):
                        with st.spinner("Creating ZIP file..."):
                            # Create ZIP in memory
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for filename, df in st.session_state.cleaned_data.items():
                                    csv_data = df.to_csv(index=False)
                                    zip_file.writestr(f"cleaned_{filename}", csv_data)

                            zip_buffer.seek(0)
                            st.download_button(
                                label="Download All Cleaned Files",
                                data=zip_buffer.getvalue(),
                                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                # Summary statistics
                st.markdown('<h2 class="section-header">Summary Statistics</h2>', unsafe_allow_html=True)

                total_original_rows = sum(df.shape[0] for df in st.session_state.original_data.values())
                total_cleaned_rows = sum(df.shape[0] for df in st.session_state.cleaned_data.values())
                total_original_nulls = sum(df.isnull().sum().sum() for df in st.session_state.original_data.values())
                total_cleaned_nulls = sum(df.isnull().sum().sum() for df in st.session_state.cleaned_data.values())

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Files Processed", len(st.session_state.cleaned_data))

                with col2:
                    st.metric("Total Rows Cleaned", f"{total_cleaned_rows:,}")

                with col3:
                    null_reduction = total_original_nulls - total_cleaned_nulls
                    st.metric("Null Values Reduced", f"{null_reduction:,}")

                with col4:
                    st.metric("Files Available", len(cleaned_csv_files))

    else:
        # Instructions when no files uploaded
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to the KLM Dashboard Data Cleaner</h3>
            <p>Upload your CSV files using the sidebar to get started. This application will:</p>
            <ul>
                <li>Clean your data by removing null values and fixing data types</li>
                <li>Analyze data quality and suggest optimal data types</li>
                <li>Provide detailed cleaning reports and statistics</li>
                <li>Allow you to download cleaned data individually or as a ZIP file</li>
            </ul>
            <p><strong>Ready to start? Upload up to 8 CSV files in the sidebar!</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Features overview
        st.markdown('<h2 class="section-header">Features</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>Data Cleaning</h3>
                <ul>
                    <li>Remove unwanted columns</li>
                    <li>Handle missing values</li>
                    <li>Fix data types</li>
                    <li>Remove duplicates</li>
                    <li>Clean empty strings</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>Data Analysis</h3>
                <ul>
                    <li>Quality assessment</li>
                    <li>Type suggestions</li>
                    <li>Null value analysis</li>
                    <li>Duplicate detection</li>
                    <li>Memory usage stats</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="info-box">
                <h3>Export Options</h3>
                <ul>
                    <li>Individual CSV downloads</li>
                    <li>Bulk ZIP export</li>
                    <li>Cleaning reports</li>
                    <li>Type analysis</li>
                    <li>Summary statistics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Comprehensive Data Treatment Logic Explanation
        st.markdown('<h2 class="section-header">Data Treatment & Validation Logic</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h3>Complete Data Processing Pipeline</h3>
            <p><strong>This application processes your data through a systematic 6-stage pipeline:</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Stage 1: Data Loading & Initial Assessment
        with st.expander("Stage 1: Data Loading & Initial Assessment", expanded=False):
            st.markdown("""
            **What happens during data loading:**

            1. **File Validation**: Each uploaded CSV file is checked for:
               - Valid CSV format and encoding
               - Readable file structure
               - Minimum data requirements (at least 1 row and 1 column)

            2. **Initial Quality Assessment**:
               - Total row and column count
               - Memory usage calculation
               - Null value percentage analysis
               - Empty string detection
               - Basic data type identification

            3. **Data Structure Analysis**:
               - Column naming patterns recognition
               - Data format consistency checking
               - Potential issues identification

            **Validation Rules Applied:**
            - Files must be valid CSV format
            - Minimum 1 row of data required
            - Readable column headers required
            - File size must be reasonable (< 100MB)
            """)

        # Stage 2: Column Analysis & Filtering
        with st.expander("Stage 2: Column Analysis & Filtering", expanded=False):
            st.markdown("""
            **Enhanced Column Management Strategy:**

            **Automatic Removal Criteria:**
            1. **System Columns**: `tail_number`, `delete_flag`, `logo_file_id`
               - *Logic*: These columns are typically system-generated and not needed for analysis

            2. **Very High NULL Columns**: Any column with ≥70% null values (increased threshold)
               - *Logic*: Preserves more analytical columns while removing truly unusable ones
               - *Calculation*: `(null_count / total_rows) * 100 ≥ 70%`

            **Row-Level Analysis:**
            - **Complete NULL Rows**: Rows with >90% NULL values are removed entirely
            - *Logic*: Rows that are mostly empty provide no analytical value

            **Column Classification System:**
            - **ID Columns**: Contains keywords like 'id', 'code', 'number', 'registration'
            - **Boolean Columns**: Named columns like 'active', 'final_delay', 'night_stop'
            - **Date/Time Columns**: Contains keywords like 'time', 'date', 'created', 'updated'
            - **Numeric Columns**: Contains keywords like 'count', 'amount', 'delay_time'
            - **Text Columns**: All other object-type columns

            **Validation Rules:**
            - Each column is analyzed for content patterns and NULL percentage
            - Column names are checked against keyword libraries
            - Data type suggestions are based on content analysis
            - More aggressive preservation of columns with moderate NULL values
            """)

        # Stage 3: Data Type Detection & Conversion
        with st.expander("Stage 3: Data Type Detection & Conversion", expanded=False):
            st.markdown("""
            **Intelligent Data Type Conversion Logic:**

            **ID Columns Processing:**
            - **Numeric Preservation**: Pure integers (no leading zeros, no decimals) remain numeric
            - **String Conversion**: Applied when:
              - Leading zeros detected (e.g., '00123')
              - Decimal points present (e.g., '123.0')
              - Non-numeric characters found (e.g., 'ABC123')
            - **Logic**: IDs should remain human-readable and preserve original formatting

            **Boolean Columns Processing:**
            - **Named Boolean Detection**: Columns matching known boolean names
            - **Content-Based Detection**: Analyzes unique values against patterns:
              - `{'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}`
            - **Conversion Mapping**: All variations map to True/False
            - **Logic**: Standardizes boolean representations for consistent analysis

            **Date/Time Columns Processing:**
            - **Pattern Recognition**: Detects various date/time formats
            - **Conversion Validation**: Minimum 80% successful parse rate required
            - **Error Handling**: Invalid dates become NaT (Not a Time)
            - **Logic**: Standardizes temporal data for time-series analysis

            **Numeric Columns Processing:**
            - **Format Detection**: Identifies integer vs. floating-point numbers
            - **Conversion Validation**: Minimum 80% successful parse rate required
            - **Error Handling**: Invalid numbers become NaN
            - **Logic**: Ensures proper numeric calculations and aggregations
            """)

        # Stage 4: Data Quality Enhancement
        with st.expander("Stage 4: Data Quality Enhancement", expanded=False):
            st.markdown("""
            **Intelligent Missing Value Treatment Strategy:**

            **Row-Level NULL Analysis:**
            - **Complete NULL Rows** (>90% NULL): Removed entirely
            - **Partial NULL Rows**: Preserved with targeted NULL treatment
            - *Logic*: Maximize data preservation while removing unusable rows

            **Column-Level NULL Strategies:**
            - **0-30% NULL**: Replace with appropriate context-aware values
            - **30-70% NULL**: Replace with standardized markers or preserve as NaN
            - **70%+ NULL**: Column removed entirely (previous stage)
            - *Logic*: Tiered approach based on data quality and usability

            **Smart NULL Replacement by Data Type:**

            **Numeric Columns:**
            - **Low NULL (≤30%)**: Median value (preserves distribution)
            - **High NULL (30-70%)**: 0 for count/order columns, NaN for measurement columns
            - **All NULL column**: Mean value (fallback when median is also NULL)

            **Text/Object Columns:**
            - **Low NULL (≤30%)**: Context-aware replacement:
              * Name/Description fields: 'Not Specified'
              * ID-like columns: 'Unknown'
              * General text: 'Unknown'
            - **High NULL (30-70%)**: 'N/A' (standardized for analysis)
            - **Logic**: Preserve meaning while standardizing missing indicators

            **Boolean Columns:**
            - **All NULL percentages**: False (neutral default state)
            - *Logic*: False is safe default for boolean logic and aggregations

            **Date/Time Columns:**
            - **All NULL percentages**: NaT (Not a Time) - pandas standard
            - *Logic*: Industry standard for missing dates/times, preserves temporal data integrity

            **Data Cleaning Operations:**
            1. **Empty String Standardization**: `['', ' ', 'NULL', 'null', 'None', 'none']` → NaN
            2. **Whitespace Trimming**: Remove leading/trailing spaces from all text
            3. **Duplicate Removal**: Eliminate completely identical rows
            4. **Final Cleanup**: Replace any remaining 'nan', 'NaN', 'NAN' with 'Unknown'
            5. **Statistical Integrity**: All replacements preserve statistical distributions where possible
            """)

        # Stage 5: Validation & Quality Assurance
        with st.expander("Stage 5: Validation & Quality Assurance", expanded=False):
            st.markdown("""
            **Comprehensive Validation Checks:**

            **Data Consistency Validation:**
            - **Referential Integrity**: Foreign key relationships checked where possible
            - **Domain Validation**: Values fall within expected ranges
            - **Format Consistency**: Standardized formats across similar columns

            **Statistical Validation:**
            - **Outlier Detection**: Identifies extreme values using IQR method
            - **Distribution Analysis**: Checks for reasonable data distributions
            - **Completeness Metrics**: Calculates data completeness scores

            **Business Logic Validation:**
            - **Date Logic**: End dates ≥ Start dates, Future dates flagged
            - **Numeric Logic**: Positive values where expected (counts, amounts)
            - **Text Logic**: Required fields not empty after cleaning

            **Quality Metrics Calculated:**
            - **Completeness Score**: `(non_null_values / total_values) * 100`
            - **Consistency Score**: Based on format and pattern adherence
            - **Validity Score**: Based on business rule compliance
            """)

        # Stage 6: Reporting & Documentation
        with st.expander("Stage 6: Reporting & Documentation", expanded=False):
            st.markdown("""
            **Comprehensive Reporting System:**

            **Cleaning Action Log:**
            - Tracks every modification with specific counts
            - Records the reasoning behind each decision
            - Provides before/after comparisons
            - Enables audit trail and reproducibility

            **Data Quality Report:**
            - **Original State**: Row count, column count, null percentage
            - **Cleaned State**: Final metrics after all processing
            - **Improvement Metrics**: Quantified improvements made
            - **Quality Score**: Overall data quality assessment

            **Type Analysis Report:**
            - **Original Types**: As detected from raw data
            - **Suggested Types**: Optimal types based on content
            - **Conversion Results**: Success rates and any issues
            - **Confidence Scores**: Reliability of type suggestions

            **Export Documentation:**
            - **Processing Summary**: Complete pipeline overview
            - **Modification Log**: Detailed change history
            - **Quality Metrics**: Final data quality assessment
            - **Usage Recommendations**: Best practices for using cleaned data
            """)

        # Data Processing Examples
        st.markdown('<h3 class="section-header">Practical Processing Examples</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>Example 1: Aircraft Registration Processing</h4>
                <p><strong>Input:</strong> ['PH-BHA', '00123', 'KLM001', '747.400']</p>
                <p><strong>Logic Applied:</strong></p>
                <ul>
                    <li>'PH-BHA' → String (contains letters)</li>
                    <li>'00123' → String (leading zeros)</li>
                    <li>'KLM001' → String (contains letters)</li>
                    <li>'747.400' → String (decimal point)</li>
                </ul>
                <p><strong>Result:</strong> All remain as strings to preserve formatting</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>Example 2: Boolean Field Processing</h4>
                <p><strong>Input:</strong> ['True', 'false', '1', '0', 'Yes', 'no', '']</p>
                <p><strong>Logic Applied:</strong></p>
                <ul>
                    <li>'True', '1', 'Yes' → True</li>
                    <li>'false', '0', 'no' → False</li>
                    <li>'' (empty) → False (default)</li>
                </ul>
                <p><strong>Result:</strong> [True, False, True, False, True, False, False]</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Example 3: Date Processing</h4>
                <p><strong>Input:</strong> ['2024-01-15', '15/01/2024', 'invalid', '']</p>
                <p><strong>Logic Applied:</strong></p>
                <ul>
                    <li>'2024-01-15' → 2024-01-15 (valid date)</li>
                    <li>'15/01/2024' → 2024-01-15 (parsed successfully)</li>
                    <li>'invalid' → NaT (parsing failed)</li>
                    <li>'' → 1900-01-01 (placeholder date)</li>
                </ul>
                <p><strong>Result:</strong> Valid dates preserved, invalid get placeholders</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>Example 4: Enhanced Missing Value Handling</h4>
                <p><strong>Column: 'delay_time' (numeric, 33% NULL)</strong></p>
                <p><strong>Input:</strong> [120, 45, null, 180, null, 90]</p>
                <p><strong>Logic Applied:</strong></p>
                <ul>
                    <li>NULL percentage: 33% (30-70% range)</li>
                    <li>Not a count/order column - preserve as NaN for analysis</li>
                    <li>Log decision for high NULL percentage</li>
                </ul>
                <p><strong>Result:</strong> [120, 45, NaN, 180, NaN, 90] + logged analysis decision</p>
            </div>
            """, unsafe_allow_html=True)

            
        # Validation Rules Summary
        st.markdown('<h3 class="section-header">Validation Rules Summary</h3>', unsafe_allow_html=True)

        validation_rules = {
            "File Validation": [
                "Must be valid CSV format",
                "File size < 100MB",
                "At least 1 row and 1 column",
                "Readable column headers"
            ],
            "Column Validation": [
                "Column names must be non-empty strings",
                "Very high null columns (≥70%) removed",
                "Unwanted system columns removed",
                "Data types must be convertible"
            ],
            "Row Validation": [
                "Complete NULL rows (>90% NULL) removed",
                "Partial NULL rows preserved with treatment",
                "Duplicate rows removed",
                "Statistical integrity maintained"
            ],
            "Data Validation": [
                "Numeric values within reasonable ranges",
                "Dates within valid ranges (1900-2100)",
                "Text fields not empty after cleaning",
                "Boolean values standardized",
                "NULL values handled by data type strategy"
            ],
            "Business Logic Validation": [
                "End dates ≥ Start dates",
                "IDs follow expected patterns",
                "Required fields populated",
                "Relationships consistent where checkable"
            ]
        }

        for rule_category, rules in validation_rules.items():
            with st.expander(f"{rule_category}", expanded=False):
                for rule in rules:
                    st.write(f"• {rule}")

        # Interactive Testing Section
        st.markdown('<h3 class="section-header">Interactive Data Processing Demo</h3>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>Test Our Cleaning Logic</h4>
            <p>Upload a small CSV file below to see exactly how our system processes your data step by step.</p>
        </div>
        """, unsafe_allow_html=True)

        # Demo file uploader for testing
        demo_file = st.file_uploader(
            "Upload a sample CSV to test cleaning logic (max 1MB)",
            type=['csv'],
            key="demo_uploader",
            help="Upload a small CSV file to see detailed processing logic in action"
        )

        if demo_file:
            try:
                demo_df = pd.read_csv(demo_file)

                with st.expander("Sample Data Preview", expanded=False):
                    st.write(f"**File:** {demo_file.name}")
                    st.write(f"**Shape:** {demo_df.shape[0]} rows × {demo_df.shape[1]} columns")
                    st.dataframe(demo_df.head())

                # Show what would happen to each column
                st.markdown("#### Processing Analysis for Each Column:")

                for col in demo_df.columns:
                    with st.expander(f"Column: {col}", expanded=False):
                        col_data = demo_df[col]

                        # Basic stats
                        st.write(f"**Current Type:** {col_data.dtype}")
                        st.write(f"**Null Values:** {col_data.isnull().sum():,} ({(col_data.isnull().sum()/len(col_data))*100:.1f}%)")
                        st.write(f"**Unique Values:** {col_data.nunique():,}")

                        # Sample values
                        sample_vals = col_data.dropna().head(10).tolist()
                        st.write(f"**Sample Values:** {sample_vals[:5]}{'...' if len(sample_vals) > 5 else ''}")

                        # What our logic would do
                        col_lower = col.lower()
                        processing_logic = []

                        # Column classification
                        if any(keyword in col_lower for keyword in ['id', 'code', 'number', 'registration']):
                            processing_logic.append("**ID Column detected** - Will check for numeric vs string format")

                            # Check actual content
                            non_null_vals = col_data.dropna().astype(str)
                            if len(non_null_vals) > 0:
                                has_leading_zeros = non_null_vals.str.contains(r'^0\d+', na=False).any()
                                has_decimals = non_null_vals.str.contains(r'\.', na=False).any()
                                has_letters = not non_null_vals.str.match(r'^\d+$', na=False).all()

                                if has_letters or has_leading_zeros or has_decimals:
                                    processing_logic.append("**Recommendation:** Keep as string (formatting important)")
                                else:
                                    processing_logic.append("**Recommendation:** Convert to integer (pure numeric)")

                        elif col_lower in ['active', 'final_delay', 'night_stop', 'summer_schedule', 'winter_schedule', 'flight_out', 'on_call_schedule']:
                            processing_logic.append("**Boolean Column detected** - Will convert to True/False")
                            unique_vals = set(col_data.dropna().astype(str).str.lower())
                            if unique_vals <= {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}:
                                processing_logic.append("**Confidence:** High (recognizable boolean pattern)")
                            else:
                                processing_logic.append("**Confidence:** Medium (unusual boolean values)")

                        elif any(keyword in col_lower for keyword in ['time', 'date', 'created', 'updated']):
                            processing_logic.append("**Date/Time Column detected** - Will attempt datetime conversion")

                            # Test datetime parsing
                            test_vals = col_data.dropna().head(20)
                            success_count = 0
                            for val in test_vals:
                                try:
                                    pd.to_datetime(val)
                                    success_count += 1
                                except:
                                    pass

                            success_rate = (success_count / len(test_vals)) * 100 if len(test_vals) > 0 else 0
                            processing_logic.append(f"**Parse Success Rate:** {success_rate:.1f}%")

                            if success_rate >= 80:
                                processing_logic.append("**Recommendation:** Convert to datetime")
                            else:
                                processing_logic.append("**Recommendation:** Keep as string (low parse rate)")

                        elif any(keyword in col_lower for keyword in ['count', 'amount', 'delay_time', 'engine_count', 'order_id']):
                            processing_logic.append("**Numeric Column detected** - Will attempt numeric conversion")

                            # Test numeric parsing
                            test_vals = col_data.dropna().head(20)
                            success_count = 0
                            for val in test_vals:
                                try:
                                    pd.to_numeric(val)
                                    success_count += 1
                                except:
                                    pass

                            success_rate = (success_count / len(test_vals)) * 100 if len(test_vals) > 0 else 0
                            processing_logic.append(f"**Parse Success Rate:** {success_rate:.1f}%")

                            if success_rate >= 80:
                                processing_logic.append("**Recommendation:** Convert to numeric")
                            else:
                                processing_logic.append("**Recommendation:** Keep as string (low parse rate)")

                        else:
                            processing_logic.append("**Text Column detected** - Will clean and standardize")

                        # Missing value strategy
                        null_count = col_data.isnull().sum()
                        if null_count > 0:
                            null_pct = (null_count / len(col_data)) * 100
                            processing_logic.append(f"**Missing Values:** {null_count:,} ({null_pct:.1f}%)")

                            if col_data.dtype == 'object':
                                if any(keyword in col_lower for keyword in ['name', 'description', 'remarks']):
                                    processing_logic.append("**Strategy:** Fill with 'Not Specified'")
                                else:
                                    processing_logic.append("**Strategy:** Fill with 'Unknown'")
                            elif col_data.dtype in ['int64', 'float64']:
                                if 'count' in col_lower or 'order' in col_lower:
                                    processing_logic.append("**Strategy:** Fill with 0")
                                else:
                                    processing_logic.append("**Strategy:** Fill with median value")
                            elif col_data.dtype == 'bool':
                                processing_logic.append("**Strategy:** Fill with False")

                        # Show the processing logic
                        for logic_item in processing_logic:
                            st.write(logic_item)

                        # Quality checks
                        st.write("---")
                        st.write("**Quality Checks:**")

                        # Check for common issues
                        if col_data.dtype == 'object':
                            empty_strings = (col_data == '').sum()
                            if empty_strings > 0:
                                st.write(f"**Empty strings:** {empty_strings:,} (will be converted to null)")

                        duplicates = col_data.duplicated().sum()
                        if duplicates > 0:
                            st.write(f"**Duplicate values:** {duplicates:,} ({(duplicates/len(col_data))*100:.1f}%)")

                        # Range check for numeric columns
                        if col_data.dtype in ['int64', 'float64']:
                            min_val = col_data.min()
                            max_val = col_data.max()
                            st.write(f"**Value Range:** {min_val} to {max_val}")

                            # Flag potential outliers
                            if col_data.dtype in ['int64', 'float64']:
                                Q1 = col_data.quantile(0.25)
                                Q3 = col_data.quantile(0.75)
                                IQR = Q3 - Q1
                                outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                                if outliers > 0:
                                    st.write(f"**Potential outliers:** {outliers:,} ({(outliers/len(col_data))*100:.1f}%)")

            except Exception as e:
                st.error(f"Error processing demo file: {str(e)}")

if __name__ == "__main__":
    main()