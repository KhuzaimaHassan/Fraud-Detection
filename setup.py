import os
import shutil
import sys
from pathlib import Path

def setup_project():
    """Create necessary directories and verify dataset locations."""
    print("Setting up project structure...")
    
    try:
        # Create necessary directories
        directories = ["models", "results", "explanations"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created {directory}/ directory")
        
        # Verify datasets exist
        datasets_dir = Path("Datasets")
        required_files = ["Fraud.csv", "FraudData_sampled.csv"]
        
        if not datasets_dir.exists():
            print(f"\nError: Datasets directory not found at {datasets_dir.absolute()}")
            print("Please ensure your datasets are in the correct location.")
            return False
        
        missing_files = []
        for file in required_files:
            file_path = datasets_dir / file
            if not file_path.exists():
                missing_files.append(file)
            else:
                print(f"✓ Found {file}")
        
        if missing_files:
            print("\nError: The following required files are missing:")
            for file in missing_files:
                print(f"- {file}")
            print(f"\nPlease place these files in the {datasets_dir} directory.")
            return False
        
        print("\nProject setup complete!")
        print("\nYou can now run:")
        print("1. python fraud_detection.py  # to train models")
        print("2. streamlit run app.py       # to start the web application")
        return True
        
    except Exception as e:
        print(f"\nError during setup: {str(e)}")
        print("Please ensure you have the necessary permissions.")
        return False

if __name__ == "__main__":
    success = setup_project()
    if not success:
        sys.exit(1) 