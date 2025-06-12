#!/usr/bin/env python3
"""
Test TorchXRayVision integration
"""

import sys
import os
sys.path.append('src')

from torchxray_integration import TorchXRayInference
import numpy as np
import cv2

def test_torchxray_integration():
    print('Testing TorchXRayVision integration...')

    inference = TorchXRayInference()
    print('Model info:', inference.get_model_info())

    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    print('Test image shape:', test_image.shape)

    result = inference.predict(test_image)
    print('Prediction result:')
    print('- Primary diagnosis:', result['primary_diagnosis'])
    print('- Confidence:', result['confidence'])
    print('- Clinical findings:', result['clinical_findings'])
    print('- Recommendations:', result['recommendations'])
    print('- Processing time:', result['processing_time'])

    print('TorchXRayVision integration test completed successfully!')
    return result

if __name__ == '__main__':
    test_torchxray_integration()
