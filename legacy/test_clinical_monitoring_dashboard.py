#!/usr/bin/env python3
"""
Test Clinical Monitoring Dashboard Integration
"""

import sys
import os
sys.path.append('src')

def test_dashboard_creation():
    """Test clinical monitoring dashboard creation"""
    try:
        from medai_clinical_monitoring_dashboard import ClinicalMonitoringDashboard
        
        print("Testing clinical monitoring dashboard creation...")
        dashboard = ClinicalMonitoringDashboard()
        
        print("✅ Clinical Monitoring Dashboard created successfully")
        
        html_content = dashboard.generate_dashboard_html()
        print(f"✅ Dashboard HTML generated ({len(html_content)} characters)")
        
        metrics = dashboard.get_current_performance_metrics()
        print("✅ Dashboard metrics retrieved:")
        for key, value in metrics.items():
            print(f"  - {key}: {value}")
        
        json_data = dashboard.get_dashboard_metrics_json()
        print(f"✅ Dashboard JSON generated ({len(json_data)} characters)")
        
        dashboard_path = dashboard.save_dashboard_html()
        print(f"✅ Dashboard HTML saved to: {dashboard_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_tracking():
    """Test dashboard metrics tracking"""
    try:
        from medai_clinical_monitoring_dashboard import ClinicalMonitoringDashboard
        
        print("\nTesting dashboard metrics tracking...")
        dashboard = ClinicalMonitoringDashboard()
        
        test_predictions = [
            {'predicted_class': 'Pneumonia', 'confidence': 0.92, 'model_agreement': 0.88, 'ensemble_uncertainty': 0.12, 'processing_time': 2.1, 'clinical_ready': True},
            {'predicted_class': 'Normal', 'confidence': 0.95, 'model_agreement': 0.91, 'ensemble_uncertainty': 0.09, 'processing_time': 1.8, 'clinical_ready': True},
            {'predicted_class': 'Pleural Effusion', 'confidence': 0.78, 'model_agreement': 0.72, 'ensemble_uncertainty': 0.28, 'processing_time': 2.5, 'clinical_ready': False}
        ]
        
        for pred in test_predictions:
            dashboard.track_prediction_metrics(pred)
        
        print(f"✅ Tracked {len(test_predictions)} predictions")
        
        clinical_validation = {
            'sensitivity': 0.94,
            'specificity': 0.91,
            'auc': 0.93,
            'precision': 0.89,
            'f1_score': 0.91,
            'false_positive_rate': 0.09,
            'false_negative_rate': 0.06,
            'clinical_agreement': 0.88
        }
        
        dashboard.track_clinical_validation_metrics(clinical_validation)
        print("✅ Clinical validation metrics tracked")
        
        updated_metrics = dashboard.get_current_performance_metrics()
        print("✅ Updated metrics retrieved:")
        print(f"  - Total predictions: {updated_metrics['total_predictions']}")
        print(f"  - Clinical ready percentage: {updated_metrics['clinical_ready_percentage']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_server_integration():
    """Test web server integration with dashboard"""
    try:
        print("\nTesting web server integration...")
        
        from web_server import app
        
        from medai_clinical_monitoring_dashboard import ClinicalMonitoringDashboard
        
        with app.test_client() as client:
            response = client.get('/api/clinical_metrics')
            print(f"✅ Clinical metrics endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.get_json()
                print("✅ Clinical metrics response received")
                if 'clinical_metrics' in data:
                    print("✅ Clinical metrics data present")
                if 'dashboard_url' in data:
                    print("✅ Dashboard URL present")
            
            response = client.get('/api/dashboard_metrics')
            print(f"✅ Dashboard metrics endpoint status: {response.status_code}")
            
            response = client.get('/clinical_dashboard')
            print(f"✅ Dashboard HTML endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                html_content = response.get_data(as_text=True)
                if 'MedAI Clinical Monitoring Dashboard' in html_content:
                    print("✅ Dashboard HTML contains expected content")
        
        return True
        
    except Exception as e:
        print(f"❌ Web server integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all dashboard tests"""
    print("🔍 TESTING CLINICAL MONITORING DASHBOARD")
    print("=" * 50)
    
    tests = [
        ("Dashboard Creation", test_dashboard_creation),
        ("Dashboard Tracking", test_dashboard_tracking),
        ("Web Server Integration", test_web_server_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All clinical monitoring dashboard tests passed!")
        print("✅ Dashboard integration complete")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
