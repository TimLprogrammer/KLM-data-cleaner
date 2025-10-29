#!/usr/bin/env python3
"""
Data types checker for KLM dashboard CSV files.
Analyzes all CSV files and suggests optimal data types for each column.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataTypeChecker:
    def __init__(self, csv_directory=""):
        self.csv_directory = csv_directory
        self.data_info = {}
        self.type_suggestions = {}

    def load_csv_file(self, file_path, filename=None):
        """Load a single CSV file."""
        if filename is None:
            filename = os.path.basename(file_path)

        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.data_info[filename] = df
                    return df
                except UnicodeDecodeError:
                    continue
            # If all encodings fail, try with error handling
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            self.data_info[filename] = df
            return df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def load_all_csv_files(self):
        """Load all CSV files from the directory."""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            filename = os.path.basename(file)
            print(f"  - {filename}")
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file, encoding=encoding)
                        self.data_info[filename] = df
                        print(f"    Loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"    Failed to load {filename}")
            except Exception as e:
                print(f"    Error loading {filename}: {e}")

    def analyze_column_types(self, filename, df):
        """Analyze and suggest optimal data types for each column."""
        suggestions = {}

        for col in df.columns:
            col_info = {
                'current_type': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'suggested_type': None,
                'confidence': 0,
                'notes': []
            }

            # Sample non-null values for analysis
            sample_values = df[col].dropna().head(100)

            if len(sample_values) == 0:
                col_info['suggested_type'] = 'object'
                col_info['confidence'] = 100
                col_info['notes'].append("Column is completely empty")
                suggestions[col] = col_info
                continue

            # Analyze patterns and suggest types
            col_lower = col.lower()
            sample_str = sample_values.astype(str)

            # ID columns
            if any(keyword in col_lower for keyword in ['id', 'code', 'number', 'registration']):
                if sample_str.str.match(r'^\d+$').all():
                    col_info['suggested_type'] = 'string'  # Keep as string to preserve leading zeros
                    col_info['confidence'] = 90
                    col_info['notes'].append("Numeric ID detected, keeping as string to preserve leading zeros")
                elif sample_str.str.match(r'^\d+\.0$').all():
                    col_info['suggested_type'] = 'string'
                    col_info['confidence'] = 85
                    col_info['notes'].append("Float ID detected, will convert to string and remove .0")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['confidence'] = 80
                    col_info['notes'].append("Mixed ID format, keeping as string")

            # Boolean columns
            elif col_lower in ['active', 'final_delay', 'night_stop', 'summer_schedule',
                             'winter_schedule', 'flight_out', 'on_call_schedule']:
                unique_values = set(sample_str.str.lower())
                if unique_values <= {'true', 'false', '1', '0', 'yes', 'no', 'nan'}:
                    col_info['suggested_type'] = 'boolean'
                    col_info['confidence'] = 95
                    col_info['notes'].append("Boolean values detected")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['confidence'] = 60
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
                    col_info['confidence'] = int((date_success / len(sample_values)) * 100)
                    col_info['notes'].append(f"{date_success}/{len(sample_values)} values parse as dates")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['confidence'] = 70
                    col_info['notes'].append("Low date parse success rate")

            # Numeric columns (excluding IDs)
            elif any(keyword in col_lower for keyword in ['count', 'amount', 'delay_time', 'engine_count', 'order_id']):
                numeric_success = 0
                for val in sample_values:
                    try:
                        pd.to_numeric(val)
                        numeric_success += 1
                    except:
                        pass

                if numeric_success / len(sample_values) > 0.8:
                    # Check if integers or floats
                    try:
                        pd.to_numeric(sample_values, dtype='int64')
                        col_info['suggested_type'] = 'int64'
                    except:
                        col_info['suggested_type'] = 'float64'

                    col_info['confidence'] = int((numeric_success / len(sample_values)) * 100)
                    col_info['notes'].append(f"{numeric_success}/{len(sample_values)} values parse as numeric")
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['confidence'] = 60
                    col_info['notes'].append("Low numeric parse success rate")

            # Categorical/string columns
            else:
                unique_ratio = col_info['unique_count'] / len(df)

                # Check for potential categorical data
                if unique_ratio < 0.1 and col_info['unique_count'] < 100:
                    col_info['suggested_type'] = 'category'
                    col_info['confidence'] = 80
                    col_info['notes'].append(f"Low cardinality ({col_info['unique_count']} unique values)")

                # Check for text data
                elif any(sample_str.str.len() > 100):
                    col_info['suggested_type'] = 'string'
                    col_info['confidence'] = 90
                    col_info['notes'].append("Long text values detected")

                # Default to string
                else:
                    col_info['suggested_type'] = 'string'
                    col_info['confidence'] = 70
                    col_info['notes'].append("Default string type")

            suggestions[col] = col_info

        return suggestions

    def analyze_all_files(self):
        """Analyze all loaded CSV files."""
        for filename, df in self.data_info.items():
            print(f"\n{'='*60}")
            print(f"Analyzing: {filename}")
            print(f"{'='*60}")

            self.type_suggestions[filename] = self.analyze_column_types(filename, df)

    def print_type_report(self):
        """Print a comprehensive type analysis report."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE DATA TYPE ANALYSIS REPORT")
        print(f"{'='*80}")

        for filename, suggestions in self.type_suggestions.items():
            print(f"\n{'='*60}")
            print(f"FILE: {filename}")
            print(f"{'='*60}")

            df = self.data_info[filename]
            print(f"Shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print()

            for col, info in suggestions.items():
                print(f"Column: {col}")
                print(f"  Current Type: {info['current_type']}")
                print(f"  Suggested Type: {info['suggested_type']}")
                print(f"  Confidence: {info['confidence']}%")
                print(f"  Null Count: {info['null_count']:,} ({info['null_percentage']:.1f}%)")
                print(f"  Unique Values: {info['unique_count']:,}")

                if info['notes']:
                    print(f"  Notes:")
                    for note in info['notes']:
                        print(f"    - {note}")
                print()

    def generate_type_conversion_code(self):
        """Generate Python code for type conversions."""
        print(f"\n{'='*80}")
        print("GENERATED TYPE CONVERSION CODE")
        print(f"{'='*80}")
        print()

        for filename, suggestions in self.type_suggestions.items():
            print(f"# Type conversions for {filename}")
            print(f"df_{filename.replace('.csv', '').replace(' ', '_').replace('-', '_')} = df_{filename.replace('.csv', '').replace(' ', '_').replace('-', '_')}.copy()")
            print()

            for col, info in suggestions.items():
                if info['suggested_type'] != info['current_type'] and info['confidence'] > 70:
                    safe_col = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')

                    if info['suggested_type'] == 'boolean':
                        df_name = f"df_{filename.replace('.csv', '').replace(' ', '_').replace('-', '_')}"
                        print(f"# Convert {col} to boolean")
                        print(f"{df_name}['{col}'] = {df_name}['{col}'].astype(str).map(")
                        print(f"    'True': True, 'true': True, '1': True, 'yes': True, 'Yes': True,")
                        print(f"    'False': False, 'false': False, '0': False, 'no': False, 'No': False")
                        print(f").fillna({df_name}['{col}'])")
                        print(f"{df_name}['{col}'] = {df_name}['{col}'].astype(bool)")
                        print()

                    elif info['suggested_type'] == 'datetime':
                        df_name = f"df_{filename.replace('.csv', '').replace(' ', '_').replace('-', '_')}"
                        print(f"# Convert {col} to datetime")
                        print(f"{df_name}['{col}'] = pd.to_datetime({df_name}['{col}'], errors='coerce')")
                        print()

                    elif info['suggested_type'] in ['int64', 'float64']:
                        df_name = f"df_{filename.replace('.csv', '').replace(' ', '_').replace('-', '_')}"
                        print(f"# Convert {col} to {info['suggested_type']}")
                        print(f"{df_name}['{col}'] = pd.to_numeric({df_name}['{col}'], errors='coerce')")
                        print()

                    elif info['suggested_type'] == 'category':
                        df_name = f"df_{filename.replace('.csv', '').replace(' ', '_').replace('-', '_')}"
                        print(f"# Convert {col} to category")
                        print(f"{df_name}['{col}'] = {df_name}['{col}'].astype('category')")
                        print()

                    elif info['suggested_type'] == 'string':
                        df_name = f"df_{filename.replace('.csv', '').replace(' ', '_').replace('-', '_')}"
                        print(f"# Convert {col} to string")
                        print(f"{df_name}['{col}'] = {df_name}['{col}'].astype(str)")
                        if '.0' in info['notes'] and 'ID' in info['notes']:
                            print(f"# Remove .0 from numeric IDs")
                            print(f"{df_name}['{col}'] = {df_name}['{col}'].str.replace(r'\\.0$', '', regex=True)")
                        print()

            print()

def main():
    """Main function to run the type analysis process."""
    # Set up paths
    csv_directory = "/Users/timlind/Documents/Jaar 4/KLM/dashboard/aplicatie/csv"

    # Initialize checker
    checker = DataTypeChecker(csv_directory)

    print("Starting comprehensive data type analysis...")
    print(f"Processing CSV files from: {csv_directory}")

    # Load all files
    checker.load_all_csv_files()

    if not checker.data_info:
        print("No CSV files were successfully loaded!")
        return

    # Analyze all files
    checker.analyze_all_files()

    # Print report
    checker.print_type_report()

    # Generate conversion code
    checker.generate_type_conversion_code()

    print(f"\n{'='*80}")
    print("DATA TYPE ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    print(f"Files analyzed: {len(checker.data_info)}")
    print("\nNext steps:")
    print("1. Review the type suggestions above")
    print("2. Use the generated code for type conversions")
    print("3. Test conversions on a copy of your data")

if __name__ == "__main__":
    main()