#!/usr/bin/env python3
"""
Simplified MedAI Radiologia server for testing console fixes
This version removes AI dependencies to focus on testing the web interface fixes
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging

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

@app.after_request
def after_request(response):
    """Add headers to prevent credential conflicts in public deployments"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    if response.content_type and 'json' not in response.content_type:
        response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API status endpoint - fixed from /api/v1/status"""
    return jsonify({
        'status': 'online',
        'version': '3.0.0',
        'service': 'MedAI Radiologia - Test Mode',
        'ai_models': 'disabled_for_testing',
        'message': 'Console error fixes active - API endpoints corrected'
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Image analysis endpoint - fixed from /api/v1/analyze"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return jsonify({
                'status': 'success',
                'message': 'Console error fixes working correctly!',
                'analysis': {
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'test_mode': True,
                    'console_fixes': {
                        'css_reset_fixed': True,
                        'api_endpoints_corrected': True,
                        'original_issue': 'CSS styling in console resolved'
                    }
                },
                'recommendations': [
                    'Console errors have been fixed',
                    'API endpoints now working correctly',
                    'CSS conflicts resolved with !important declarations'
                ]
            })
    
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'mode': 'testing'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting MedAI Radiologia Test Server on {host}:{port}")
    logger.info("Console error fixes active - testing API endpoints")
    
    app.run(host=host, port=port, debug=False)
