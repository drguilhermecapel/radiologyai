#!/usr/bin/env python3
"""
Optimized MedAI Radiologia web server with real TorchXRayVision AI
Replaces dummy/simulation system with actual medical AI diagnostics
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
import time
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

CORS(app, 
     origins=["*"], 
     allow_headers=["Content-Type", "Accept"],
     methods=["GET", "POST", "OPTIONS"])

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

torchxray_model = None

def initialize_torchxray_model():
    """Initialize TorchXRayVision model for real AI diagnosis"""
    global torchxray_model
    try:
        import sys
        sys.path.append('src')
        from torchxray_integration import TorchXRayInference
        torchxray_model = TorchXRayInference()
        logger.info("TorchXRayVision real AI model loaded successfully")
        logger.info(f"Model info: {torchxray_model.get_model_info()}")
        return True
    except Exception as e:
        logger.error(f"Failed to load TorchXRayVision model: {str(e)}")
        return False

@app.after_request
def after_request(response):
    """Add headers to prevent credential conflicts in public deployments"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    
    if request.path.startswith('/api/') and response.content_type and 'json' not in response.content_type:
        response.headers['Content-Type'] = 'application/json'
    
    return response

@app.route('/api/analyze', methods=['OPTIONS'])
def api_analyze_options():
    """Handle preflight requests for analyze endpoint"""
    return '', 200

@app.route('/api/status', methods=['OPTIONS'])
def api_status_options():
    """Handle preflight requests for status endpoint"""
    return '', 200

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API status endpoint with TorchXRayVision model info"""
    global torchxray_model
    
    if torchxray_model:
        model_info = torchxray_model.get_model_info()
        return jsonify({
            'status': 'online',
            'version': '3.0-torchxray',
            'service': 'MedAI Radiologia - Real AI Diagnostics',
            'ai_models': 'torchxrayvision_active',
            'model_details': {
                'model_name': model_info.get('model_name', 'densenet121'),
                'pathologies': len(model_info.get('pathologies', [])),
                'device': model_info.get('device', 'cpu'),
                'clinical_categories': model_info.get('clinical_categories', [])
            },
            'message': 'Real AI diagnostic system active - TorchXRayVision models loaded'
        })
    else:
        return jsonify({
            'status': 'offline',
            'version': '3.0-torchxray',
            'service': 'MedAI Radiologia - Real AI Diagnostics',
            'ai_models': 'loading_failed',
            'message': 'TorchXRayVision model failed to load'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Real AI image analysis using TorchXRayVision"""
    global torchxray_model
    
    if not torchxray_model:
        return jsonify({'error': 'TorchXRayVision model not loaded'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            start_time = time.time()
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    pil_image = Image.open(filepath).convert('L')
                    image = np.array(pil_image)
                
                logger.info(f"Image loaded: {image.shape}, dtype: {image.dtype}")
                
                torchxray_result = torchxray_model.predict(image)
                
                if torchxray_result.get('error'):
                    return jsonify({
                        'error': f'TorchXRayVision analysis error: {torchxray_result["error"]}'
                    }), 500
                
                all_diagnoses = torchxray_result.get('all_diagnoses', [])
                primary_diagnosis = torchxray_result.get('primary_diagnosis', 'normal')
                
                diagnosis_mapping = {
                    'pneumonia': 'Pneumonia',
                    'pleural_effusion': 'Derrame pleural', 
                    'fracture': 'Fratura óssea',
                    'tumor': 'Massa/Nódulo suspeito',
                    'normal': 'Normal'
                }
                
                predicted_class = diagnosis_mapping.get(primary_diagnosis, primary_diagnosis.title())
                confidence = torchxray_result.get('confidence', 0.0)
                pathology_scores = torchxray_result.get('pathology_scores', {})
                clinical_findings = torchxray_result.get('clinical_findings', [])
                recommendations = torchxray_result.get('recommendations', [])
                
                processing_time = time.time() - start_time
                
                response_data = {
                    'status': 'success',
                    'analysis': {
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'all_diagnoses': all_diagnoses,  # New: all significant diagnoses
                        'pathology_scores': pathology_scores,
                        'clinical_findings': clinical_findings,
                        'recommendations': recommendations,
                        'processing_time': processing_time,
                        'model_info': {
                            'model_type': 'torchxrayvision',
                            'model_name': torchxray_result.get('model_info', {}).get('model_name', 'densenet121'),
                            'pathologies_detected': torchxray_result.get('model_info', {}).get('pathologies_detected', 0),
                            'total_pathologies': torchxray_result.get('model_info', {}).get('total_pathologies', 18),
                            'ai_version': '3.0-torchxray-multi-diagnosis'
                        }
                    },
                    'metadata': {
                        'filename': filename,
                        'file_size': os.path.getsize(filepath),
                        'timestamp': time.time(),
                        'image_shape': image.shape,
                        'primary_category': primary_diagnosis,
                        'total_diagnoses': len(all_diagnoses)
                    }
                }
                
                logger.info(f"Real AI diagnosis completed: {predicted_class} (confidence: {confidence:.3f})")
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({
                    'error': f'Image processing failed: {str(e)}'
                }), 500
    
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    global torchxray_model
    return jsonify({
        'status': 'healthy', 
        'mode': 'torchxray_real_ai',
        'model_loaded': torchxray_model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting MedAI Radiologia Real AI Server on {host}:{port}")
    logger.info("Initializing TorchXRayVision real AI diagnostic system...")
    
    if initialize_torchxray_model():
        logger.info("TorchXRayVision model loaded successfully - Real AI diagnostics active")
    else:
        logger.error("Failed to load TorchXRayVision model - Server will return errors")
    
    app.run(host=host, port=port, debug=False)
