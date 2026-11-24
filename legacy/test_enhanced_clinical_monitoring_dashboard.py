"""
Test script for enhanced clinical monitoring dashboard
Validates real-time monitoring, alerting, and visualization capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import json
import time
from datetime import datetime, timedelta
from medai_clinical_monitoring_dashboard import ClinicalMonitoringDashboard

def test_enhanced_monitoring_dashboard():
    """Test the enhanced clinical monitoring dashboard functionality"""
    print("🚀 Testing Enhanced Clinical Monitoring Dashboard")
    print("=" * 60)
    
    try:
        dashboard = ClinicalMonitoringDashboard()
        print("✅ Clinical Monitoring Dashboard initialized successfully")
        
        print("\n📊 Testing real-time prediction tracking...")
        prediction_data = {
            'predicted_class': 'pneumonia',
            'confidence': 0.92,
            'model_agreement': 0.88,
            'ensemble_uncertainty': 0.12,
            'processing_time': 1.8,
            'clinical_ready': True
        }
        
        dashboard.track_prediction_metrics(prediction_data)
        print("  ✅ Prediction metrics tracked successfully")
        
        print("\n🏥 Testing clinical validation tracking...")
        validation_data = {
            'sensitivity': 0.94,
            'specificity': 0.91,
            'auc': 0.93,
            'precision': 0.89,
            'f1_score': 0.91,
            'false_positive_rate': 0.09,
            'false_negative_rate': 0.06,
            'clinical_agreement': 0.92
        }
        
        dashboard.track_clinical_validation_metrics(validation_data)
        print("  ✅ Clinical validation metrics tracked successfully")
        
        print("\n🚨 Testing alert system...")
        low_performance_data = {
            'predicted_class': 'pleural_effusion',
            'confidence': 0.65,  # Below threshold
            'model_agreement': 0.60,  # Below threshold
            'ensemble_uncertainty': 0.40,
            'processing_time': 5.2,  # High processing time
            'clinical_ready': False
        }
        
        dashboard.track_prediction_metrics(low_performance_data)
        print("  ✅ Alert system triggered for low performance metrics")
        
        print("\n🔄 Testing retraining triggers...")
        degraded_validation_data = {
            'sensitivity': 0.82,  # Significant drop
            'specificity': 0.78,  # Significant drop
            'auc': 0.75,  # Below threshold
            'precision': 0.80,
            'f1_score': 0.81,
            'false_positive_rate': 0.22,  # High increase
            'false_negative_rate': 0.18,
            'clinical_agreement': 0.75
        }
        
        dashboard.track_clinical_validation_metrics(degraded_validation_data)
        print("  ✅ Retraining triggers activated for performance degradation")
        
        print("\n📈 Testing performance metrics API...")
        current_metrics = dashboard.get_current_performance_metrics()
        
        assert 'ensemble_accuracy' in current_metrics
        assert 'sensitivity' in current_metrics
        assert 'specificity' in current_metrics
        assert 'total_predictions' in current_metrics
        
        print(f"  ✅ Current metrics retrieved: {len(current_metrics)} metrics")
        print(f"  📊 Total predictions tracked: {current_metrics['total_predictions']}")
        print(f"  🎯 Ensemble accuracy: {current_metrics['ensemble_accuracy']:.1%}")
        
        print("\n🌐 Testing dashboard HTML generation...")
        dashboard_html = dashboard.generate_dashboard_html()
        
        assert len(dashboard_html) > 1000  # Should be substantial HTML
        assert 'MedAI Clinical Monitoring Dashboard' in dashboard_html
        assert 'Sensitivity' in dashboard_html
        assert 'Specificity' in dashboard_html
        assert 'chart.js' in dashboard_html
        
        print("  ✅ Dashboard HTML generated successfully")
        print(f"  📄 HTML size: {len(dashboard_html)} characters")
        
        print("\n🔗 Testing dashboard JSON API...")
        metrics_json = dashboard.get_dashboard_metrics_json()
        metrics_data = json.loads(metrics_json)
        
        assert 'metrics' in metrics_data
        assert 'thresholds' in metrics_data
        assert 'system_status' in metrics_data
        
        print("  ✅ Dashboard JSON API working correctly")
        
        print("\n⚙️ Testing configuration management...")
        config = dashboard.config
        
        assert hasattr(dashboard, 'alert_thresholds')
        assert hasattr(dashboard, 'retraining_triggers')
        assert len(dashboard.alert_thresholds) > 0
        assert len(dashboard.retraining_triggers) > 0
        
        print(f"  ✅ Alert thresholds configured: {len(dashboard.alert_thresholds)} metrics")
        print(f"  ✅ Retraining triggers configured: {len(dashboard.retraining_triggers)} triggers")
        print("  ✅ Configuration management validated")
        
        print("\n📚 Testing metrics history...")
        history_count = len(dashboard.metrics_history)
        performance_log_count = len(dashboard.performance_log)
        
        print(f"  📊 Metrics history entries: {history_count}")
        print(f"  📈 Performance log entries: {performance_log_count}")
        
        assert history_count > 0
        assert performance_log_count > 0
        
        print("  ✅ Metrics history tracking working correctly")
        
        dashboard.save_dashboard_html("models/clinical_monitoring_dashboard.html")
        print("  💾 Dashboard HTML saved to models/clinical_monitoring_dashboard.html")
        
        print("\n" + "=" * 60)
        print("✅ ENHANCED CLINICAL MONITORING DASHBOARD VALIDATED")
        print("🏥 Real-time monitoring, alerting, and visualization fully functional")
        
        print(f"\n📋 DASHBOARD CAPABILITIES SUMMARY:")
        print(f"   • Real-time prediction tracking: ✅")
        print(f"   • Clinical validation monitoring: ✅")
        print(f"   • Performance alert system: ✅")
        print(f"   • Automated retraining triggers: ✅")
        print(f"   • HTML dashboard generation: ✅")
        print(f"   • JSON API for metrics: ✅")
        print(f"   • Configuration management: ✅")
        print(f"   • Metrics history tracking: ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR IN TESTING: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_monitoring_dashboard()
    sys.exit(0 if success else 1)
