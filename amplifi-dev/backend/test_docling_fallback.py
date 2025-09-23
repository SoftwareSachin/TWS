#!/usr/bin/env python3
"""
Test script for docling converter fallback functionality.
This script tests the get_docling_converter function to ensure it properly
initializes models when they're not available.
"""

import os
import sys
import time

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.utils.docling_model_manager import (
    get_docling_converter,
    get_docling_status,
    is_docling_ready,
    clear_docling_cache,
)
from app.models.document_model import DocumentTypeEnum
from app.be_core.logger import logger


def test_docling_converter_fallback():
    """Test the fallback functionality of get_docling_converter"""
    print("Testing docling converter fallback functionality...")
    
    # Clear any existing converters to simulate uninitialized state
    print("\n1. Clearing existing converters...")
    clear_docling_cache()
    
    # Check initial status
    print("\n2. Checking initial status:")
    status = get_docling_status()
    print(f"   Ready: {status['ready']}")
    print(f"   PDF converter initialized: {status['pdf_converter_initialized']}")
    print(f"   Other converter initialized: {status['other_converter_initialized']}")
    
    # Test PDF converter fallback
    print("\n3. Testing PDF converter fallback...")
    try:
        start_time = time.time()
        pdf_converter = get_docling_converter(DocumentTypeEnum.PDF)
        pdf_time = time.time() - start_time
        
        if pdf_converter is not None:
            print(f"   ✅ PDF converter created successfully in {pdf_time:.2f} seconds")
        else:
            print("   ❌ PDF converter creation failed")
            return False
    except Exception as e:
        print(f"   ❌ PDF converter creation failed with error: {str(e)}")
        return False
    
    # Test other document type converter fallback
    print("\n4. Testing other document type converter fallback...")
    try:
        start_time = time.time()
        other_converter = get_docling_converter(DocumentTypeEnum.DOCX)
        other_time = time.time() - start_time
        
        if other_converter is not None:
            print(f"   ✅ Other converter created successfully in {other_time:.2f} seconds")
        else:
            print("   ❌ Other converter creation failed")
            return False
    except Exception as e:
        print(f"   ❌ Other converter creation failed with error: {str(e)}")
        return False
    
    # Check final status
    print("\n5. Checking final status:")
    status = get_docling_status()
    print(f"   Ready: {status['ready']}")
    print(f"   PDF converter initialized: {status['pdf_converter_initialized']}")
    print(f"   Other converter initialized: {status['other_converter_initialized']}")
    
    # Test that subsequent calls return the same converters (caching works)
    print("\n6. Testing converter caching...")
    try:
        pdf_converter2 = get_docling_converter(DocumentTypeEnum.PDF)
        other_converter2 = get_docling_converter(DocumentTypeEnum.DOCX)
        
        if pdf_converter is pdf_converter2 and other_converter is other_converter2:
            print("   ✅ Converter caching works correctly")
        else:
            print("   ❌ Converter caching failed")
            return False
    except Exception as e:
        print(f"   ❌ Converter caching test failed with error: {str(e)}")
        return False
    
    print("\n✅ Docling converter fallback test PASSED")
    return True


if __name__ == "__main__":
    try:
        success = test_docling_converter_fallback()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        sys.exit(1) 