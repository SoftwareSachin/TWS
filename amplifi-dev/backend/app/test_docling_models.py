#!/usr/bin/env python3
"""
Test script for docling model downloading functionality.
Run this script to test if docling models can be downloaded and initialized.
"""

import os
import sys
import time

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.utils.docling_model_manager import (
    initialize_docling_models,
    get_docling_status,
    is_docling_ready,
    get_docling_converter,
)
from app.be_core.logger import logger


def test_docling_model_download():
    """Test docling model downloading functionality."""
    print("Testing docling model download functionality...")
    
    # Check initial status
    print("\n1. Initial status:")
    status = get_docling_status()
    print(f"   Ready: {status['ready']}")
    print(f"   Converter initialized: {status['converter_initialized']}")
    
    # Test model download
    print("\n2. Testing model download...")
    start_time = time.time()
    
    success = initialize_docling_models(force_download=False)
    download_time = time.time() - start_time
    
    print(f"   Download successful: {success}")
    print(f"   Download time: {download_time:.2f} seconds")
    
    # Check final status
    print("\n3. Final status:")
    status = get_docling_status()
    print(f"   Ready: {status['ready']}")
    print(f"   Converter initialized: {status['converter_initialized']}")
    
    # Test converter availability
    print("\n4. Testing converter availability:")
    converter = get_docling_converter()
    print(f"   Converter available: {converter is not None}")
    print(f"   Is ready: {is_docling_ready()}")
    
    if success and converter:
        print("\n✅ Docling model download test PASSED")
        return True
    else:
        print("\n❌ Docling model download test FAILED")
        return False


if __name__ == "__main__":
    try:
        success = test_docling_model_download()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        sys.exit(1) 