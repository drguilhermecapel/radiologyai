#!/usr/bin/env python3
"""
Web Server Functionality Test
Tests the web server endpoints and AI integration
"""

import sys
import os
import requests
import json
import time
import numpy as np
from io import BytesIO
import base64

def test_web_server_endpoints():
    """Test all web server endpoints"""
    print("Testing web server functionality...")
    
    base_url = "http://localhost:49571"
    
    try:
        response = requests.get(f"{base_url}/api/status", timeout=10)
        if response.status_code == 200:
            print("✅ Server status endpoint working")
            status_data = response.json()
            print(f"  Status: {status_data.get('status', 'unknown')}")
            print(f"  Models loaded: {status_data.get('models_loaded', 0)}")
        else:
            print(f"❌ Server status endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing status endpoint: {e}")
        return False
    
    try:
        response = requests.get(f"{base_url}/api/models", timeout=10)
        if response.status_code == 200:
            print("✅ Models endpoint working")
            models_data = response.json()
            print(f"  Available models: {len(models_data.get('models', []))}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing models endpoint: {e}")
    
    try:
        response = requests.get(f"{base_url}/api/clinical_metrics", timeout=10)
        if response.status_code == 200:
            print("✅ Clinical metrics endpoint working")
            metrics_data = response.json()
            print(f"  Clinical standards defined: {bool(metrics_data.get('clinical_standards'))}")
        else:
            print(f"❌ Clinical metrics endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing clinical metrics endpoint: {e}")
    
    return True

def test_image_analysis_endpoint():
    """Test the image analysis endpoint with synthetic data"""
    print("\nTesting image analysis endpoint...")
    
    base_url = "http://localhost:49571"
    
    synthetic_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    from PIL import Image
    img = Image.fromarray(synthetic_image)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    
    try:
        files = {'image': ('test_image.png', img_data, 'image/png')}
        response = requests.post(f"{base_url}/api/analyze", files=files, timeout=30)
        
        if response.status_code == 200:
            print("✅ Image analysis endpoint working")
            result = response.json()
            print(f"  Predicted class: {result.get('predicted_class', 'unknown')}")
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            print(f"  Processing time: {result.get('processing_time', 0):.3f}s")
            
            if 'clinical_findings' in result:
                print("✅ Clinical findings generated")
                findings = result['clinical_findings']
                print(f"  Findings count: {len(findings.get('findings', []))}")
                print(f"  Recommendations count: {len(findings.get('recommendations', []))}")
            
            return True
        else:
            print(f"❌ Image analysis endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing image analysis endpoint: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("WEB SERVER FUNCTIONALITY TEST")
    print("="*60)
    
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    endpoints_ok = test_web_server_endpoints()
    
    analysis_ok = test_image_analysis_endpoint()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Endpoints: {'✅ PASS' if endpoints_ok else '❌ FAIL'}")
    print(f"Image Analysis: {'✅ PASS' if analysis_ok else '❌ FAIL'}")
    
    overall_success = endpoints_ok and analysis_ok
    print(f"Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
