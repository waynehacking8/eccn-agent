import os
import json
import pandas as pd
import requests
import argparse
import re
from urllib.parse import urlparse
from pathlib import Path
import importlib.util
import sys

def download_file(url, save_path):
    """Download a file from URL and save it to the specified path if it doesn't already exist"""
    # Check if file already exists
    if os.path.exists(save_path):
        print(f"File already exists, skipping download: {save_path}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Write content to file
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        print(f"Successfully downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def get_filename_from_url(url):
    """Extract filename from URL"""
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)

def extract_system_prompt(file_path):
    """Extract SYSTEM_PROMPT from a Python file using importlib"""
    try:
        # Get absolute path
        abs_path = os.path.abspath(file_path)
        
        # Get module name from file path (without .py extension)
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create spec
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None:
            print(f"Warning: Could not create spec from {file_path}")
            return None
            
        # Create module
        module = importlib.util.module_from_spec(spec)
        
        # Add module to sys.modules
        sys.modules[module_name] = module
        
        # Execute module
        spec.loader.exec_module(module)
        
        # Get SYSTEM_PROMPT from module
        if hasattr(module, 'SYSTEM_PROMPT'):
            return module.SYSTEM_PROMPT
        else:
            print("Warning: SYSTEM_PROMPT not found in the specified file.")
            return None
    
    except Exception as e:
        print(f"Error importing SYSTEM_PROMPT: {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download images and PDFs from Excel file and create JSON test cases')
    parser.add_argument('--input', required=True, help='Path to the Excel file')
    parser.add_argument('--output', default='test_cases.json', help='Output JSON file name')
    parser.add_argument('--prompt', default='prompt.py', help='Path to the prompt.py file containing SYSTEM_PROMPT')
    args = parser.parse_args()
    
    # Path to the Excel file from command line argument
    excel_file = args.input
    
    # Output JSON file name
    output_filename = args.output
    
    # Path to the prompt.py file
    prompt_file = args.prompt
    
    # Base directory for saving files
    data_dir = "data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Extract SYSTEM_PROMPT from prompt.py
    system_prompt = extract_system_prompt(prompt_file)
    if not system_prompt:
        system_prompt = "You are a helpful assistant specialized in export control classification. Analyze the provided product information and determine the correct US Export Control Classification Number (ECCN). {user_question}"
        print(f"Using default prompt template: {system_prompt}")
    
    # Initialize test cases list
    test_cases = []
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Process each row
        for index, row in df.iterrows():
            # Get subfolder name from column A
            subfolder_name = str(row.iloc[0])
            if not subfolder_name or pd.isna(subfolder_name):
                subfolder_name = f"unknown_{index}"
            
            # Create subfolder path
            subfolder_path = os.path.join(data_dir, subfolder_name)
            
            # Get product description from column D
            product_description = str(row.iloc[3]) if not pd.isna(row.iloc[3]) else ""
            
            # Get ECCN from column with "US Export Control Classification Number (ECCN)"
            eccn_value = None
            for col_idx, col_name in enumerate(df.columns):
                if "US Export Control Classification Number (ECCN)" in str(col_name):
                    eccn_value = str(row.iloc[col_idx]) if not pd.isna(row.iloc[col_idx]) else ""
                    break
            
            if eccn_value is None:
                print(f"Warning: ECCN column not found for row {index}")
                continue
            
            # Get PDF URL from column I
            pdf_url = row.iloc[8]  # Column I (0-indexed)
            pdf_path = None
            pdf_download_success = False
            
            if pdf_url and not pd.isna(pdf_url):
                # Get PDF filename from URL or use default
                pdf_filename = get_filename_from_url(pdf_url) or f"document_{index}.pdf"
                pdf_path = os.path.join(subfolder_path, pdf_filename)
                
                # Download file only if it doesn't exist
                pdf_download_success = download_file(pdf_url, pdf_path)
            
            # Get image URL from column J
            image_url = row.iloc[9]  # Column J (0-indexed)
            image_path = None
            image_download_success = False
            
            if image_url and not pd.isna(image_url):
                # Get image filename from URL or use default
                image_filename = get_filename_from_url(image_url) or f"image_{index}.jpg"
                image_path = os.path.join(subfolder_path, image_filename)
                
                # Download file only if it doesn't exist
                image_download_success = download_file(image_url, image_path)
            
            # Create test case if both downloads were successful
            if image_download_success and pdf_download_success:
                test_case = {
                    "product_description": product_description,
                    "product_pdf": pdf_path,
                    "product_image": image_path,
                    "ground_truth": eccn_value
                }
                test_cases.append(test_case)
        
        # Create the final JSON structure
        json_data = {
            "prompt_template": system_prompt,
            "test_cases": test_cases
        }
        
        # Write JSON to file
        json_output_path = os.path.join(data_dir, output_filename)
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2)
        
        print(f"JSON file created successfully at: {json_output_path}")
        print(f"Total test cases created: {len(test_cases)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()