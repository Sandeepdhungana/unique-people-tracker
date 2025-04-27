"""
Fix dependency issues in the environment.
Run this script if you encounter errors like 'Numpy is not available'.
"""

import sys
import subprocess

def fix_dependencies():
    """Reinstall key dependencies"""
    print("Checking and fixing dependencies...")
    
    # Packages to reinstall
    packages = [
        "numpy==1.24.3",
        "torch",
        "torchvision",
        "opencv-python",
        "supervision",
        "scikit-learn"
    ]
    
    for package in packages:
        print(f"Reinstalling {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", package])
    
    # Test imports
    try:
        print("\nTesting NumPy import...")
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        print("\nTesting PyTorch import...")
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        print("\nTesting OpenCV import...")
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        print("\nTesting Supervision import...")
        import supervision as sv
        print(f"Supervision version: {sv.__version__}")
        
        print("\nAll dependencies are working correctly!")
    except ImportError as e:
        print(f"Error: {e}")
        print("Some dependencies are still not working correctly.")
        return False
    
    return True

if __name__ == "__main__":
    success = fix_dependencies()
    if success:
        print("\nYou can now run the demo_deep_reid.py script.")
    else:
        print("\nTry manually reinstalling the problematic packages.") 