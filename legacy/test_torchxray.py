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
    model_info = inference.get_model_info()
    print('Model info:', model_info)
    
    assert 'model_name' in model_info
    assert 'pathologies' in model_info
    assert len(model_info['pathologies']) == 18

    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    print('Test image shape:', test_image.shape)

    result = inference.predict(test_image)
    print('Prediction result:')
    print('- Primary diagnosis:', result['primary_diagnosis'])
    print('- Confidence:', result['confidence'])
    print('- Clinical findings:', result['clinical_findings'])
    print('- Recommendations:', result['recommendations'])
    print('- Processing time:', result['processing_time'])

    assert 'primary_diagnosis' in result
    assert 'confidence' in result
    assert 'all_diagnoses' in result
    assert 'pathology_scores' in result
    assert 'clinical_findings' in result
    assert 'recommendations' in result
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1
    assert isinstance(result['all_diagnoses'], list)
    assert len(result['all_diagnoses']) > 0
    assert len(result['pathology_scores']) == 18

    print('TorchXRayVision integration test completed successfully!')
    print(f'✅ Multi-diagnosis functionality verified: {len(result["all_diagnoses"])} diagnoses returned')

if __name__ == '__main__':
    test_torchxray_integration()
