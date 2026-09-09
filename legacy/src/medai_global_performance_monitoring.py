#!/usr/bin/env python3
"""
Global Performance Monitoring System for RadiologyAI - Phase 9
Real-time monitoring and analytics for global deployment with multi-institutional performance tracking
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import asyncio
from collections import defaultdict, deque
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class PerformanceMetric(Enum):
    """Performance metrics to monitor"""
    ACCURACY = "accuracy"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    AUC_ROC = "auc_roc"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_id: str
    timestamp: str
    severity: AlertSeverity
    metric: PerformanceMetric
    institution_id: str
    region: str
    current_value: float
    threshold_value: float
    message: str
    resolved: bool = False
    resolution_timestamp: Optional[str] = None

@dataclass
class InstitutionMetrics:
    """Real-time metrics for an institution"""
    institution_id: str
    institution_name: str
    region: str
    timestamp: str
    accuracy: float
    sensitivity: float
    specificity: float
    auc_roc: float
    response_time_ms: float
    throughput_per_hour: int
    error_rate: float
    availability_percentage: float
    cases_processed_today: int
    alerts_active: int

@dataclass
class RegionalSummary:
    """Regional performance summary"""
    region: str
    institution_count: int
    total_cases_processed: int
    avg_accuracy: float
    avg_response_time: float
    total_alerts: int
    critical_alerts: int
    availability_percentage: float
    timestamp: str

@dataclass
class GlobalDashboard:
    """Global performance dashboard data"""
    dashboard_timestamp: str
    global_metrics: Dict[str, float]
    regional_summaries: List[RegionalSummary]
    institution_metrics: List[InstitutionMetrics]
    active_alerts: List[PerformanceAlert]
    performance_trends: Dict[str, List[float]]
    system_health_score: float

class RealTimeMetricsCollector:
    """Collects real-time performance metrics from institutions"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minute data
        self.is_collecting = False
        self.collection_thread = None
        
        logger.info("Real-time metrics collector initialized")
    
    def start_collection(self):
        """Start real-time metrics collection"""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Real-time metrics collection started")
    
    def stop_collection(self):
        """Stop real-time metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info("Real-time metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                self._collect_metrics_snapshot()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics_snapshot(self):
        """Collect metrics snapshot from all institutions"""
        timestamp = datetime.now().isoformat()
        
        institutions = [
            ("INST_USA_001", "Mayo Clinic", "north_america"),
            ("INST_GER_001", "Charité Berlin", "europe"),
            ("INST_JPN_001", "University of Tokyo Hospital", "asia_pacific"),
            ("INST_CAN_001", "Toronto General Hospital", "north_america"),
            ("INST_UK_001", "Imperial College London", "europe"),
            ("INST_AUS_001", "Royal Melbourne Hospital", "asia_pacific")
        ]
        
        for inst_id, inst_name, region in institutions:
            metrics = self._simulate_institution_metrics(inst_id, inst_name, region, timestamp)
            self.metrics_buffer[inst_id].append(metrics)
        
        logger.debug(f"Collected metrics snapshot at {timestamp}")
    
    def _simulate_institution_metrics(self, inst_id: str, inst_name: str, 
                                    region: str, timestamp: str) -> InstitutionMetrics:
        """Simulate real-time metrics for an institution"""
        
        base_accuracy = 0.89 + np.random.normal(0, 0.02)
        base_sensitivity = 0.87 + np.random.normal(0, 0.02)
        base_specificity = 0.91 + np.random.normal(0, 0.02)
        
        regional_factors = {
            "north_america": 1.0,
            "europe": 0.98,
            "asia_pacific": 0.96
        }
        
        factor = regional_factors.get(region, 0.95)
        
        hour = datetime.now().hour
        time_factor = 1.0 + 0.1 * np.sin(2 * np.pi * hour / 24)
        
        metrics = InstitutionMetrics(
            institution_id=inst_id,
            institution_name=inst_name,
            region=region,
            timestamp=timestamp,
            accuracy=min(base_accuracy * factor, 1.0),
            sensitivity=min(base_sensitivity * factor, 1.0),
            specificity=min(base_specificity * factor, 1.0),
            auc_roc=min(0.92 * factor + np.random.normal(0, 0.01), 1.0),
            response_time_ms=max(150 + np.random.normal(0, 30) / time_factor, 50),
            throughput_per_hour=int(max(100 * time_factor + np.random.normal(0, 20), 20)),
            error_rate=max(0.02 + np.random.normal(0, 0.01), 0.0),
            availability_percentage=min(99.5 + np.random.normal(0, 0.5), 100.0),
            cases_processed_today=int(max(500 * time_factor + np.random.normal(0, 100), 50)),
            alerts_active=max(int(np.random.poisson(1)), 0)
        )
        
        return metrics
    
    def get_latest_metrics(self, institution_id: str) -> Optional[InstitutionMetrics]:
        """Get latest metrics for an institution"""
        if institution_id in self.metrics_buffer and self.metrics_buffer[institution_id]:
            return self.metrics_buffer[institution_id][-1]
        return None
    
    def get_metrics_history(self, institution_id: str, hours: int = 24) -> List[InstitutionMetrics]:
        """Get metrics history for an institution"""
        if institution_id not in self.metrics_buffer:
            return []
        
        entries_needed = min(hours * 60, len(self.metrics_buffer[institution_id]))
        return list(self.metrics_buffer[institution_id])[-entries_needed:]

class PerformanceAlertSystem:
    """Manages performance alerts and notifications"""
    
    def __init__(self):
        self.alert_thresholds = {
            PerformanceMetric.ACCURACY: 0.85,
            PerformanceMetric.SENSITIVITY: 0.85,
            PerformanceMetric.SPECIFICITY: 0.85,
            PerformanceMetric.AUC_ROC: 0.85,
            PerformanceMetric.RESPONSE_TIME: 500.0,  # ms
            PerformanceMetric.ERROR_RATE: 0.05,
            PerformanceMetric.AVAILABILITY: 99.0
        }
        
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_history: List[PerformanceAlert] = []
        
        logger.info("Performance alert system initialized")
    
    def check_metrics_for_alerts(self, metrics: InstitutionMetrics) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts"""
        
        new_alerts = []
        timestamp = datetime.now().isoformat()
        
        if metrics.accuracy < self.alert_thresholds[PerformanceMetric.ACCURACY]:
            alert = PerformanceAlert(
                alert_id=f"ACC_{metrics.institution_id}_{int(time.time())}",
                timestamp=timestamp,
                severity=AlertSeverity.HIGH if metrics.accuracy < 0.80 else AlertSeverity.MEDIUM,
                metric=PerformanceMetric.ACCURACY,
                institution_id=metrics.institution_id,
                region=metrics.region,
                current_value=metrics.accuracy,
                threshold_value=self.alert_thresholds[PerformanceMetric.ACCURACY],
                message=f"Accuracy below threshold: {metrics.accuracy:.3f} < {self.alert_thresholds[PerformanceMetric.ACCURACY]}"
            )
            new_alerts.append(alert)
        
        if metrics.sensitivity < self.alert_thresholds[PerformanceMetric.SENSITIVITY]:
            alert = PerformanceAlert(
                alert_id=f"SEN_{metrics.institution_id}_{int(time.time())}",
                timestamp=timestamp,
                severity=AlertSeverity.HIGH if metrics.sensitivity < 0.80 else AlertSeverity.MEDIUM,
                metric=PerformanceMetric.SENSITIVITY,
                institution_id=metrics.institution_id,
                region=metrics.region,
                current_value=metrics.sensitivity,
                threshold_value=self.alert_thresholds[PerformanceMetric.SENSITIVITY],
                message=f"Sensitivity below threshold: {metrics.sensitivity:.3f} < {self.alert_thresholds[PerformanceMetric.SENSITIVITY]}"
            )
            new_alerts.append(alert)
        
        if metrics.response_time_ms > self.alert_thresholds[PerformanceMetric.RESPONSE_TIME]:
            alert = PerformanceAlert(
                alert_id=f"RT_{metrics.institution_id}_{int(time.time())}",
                timestamp=timestamp,
                severity=AlertSeverity.MEDIUM if metrics.response_time_ms < 1000 else AlertSeverity.HIGH,
                metric=PerformanceMetric.RESPONSE_TIME,
                institution_id=metrics.institution_id,
                region=metrics.region,
                current_value=metrics.response_time_ms,
                threshold_value=self.alert_thresholds[PerformanceMetric.RESPONSE_TIME],
                message=f"Response time above threshold: {metrics.response_time_ms:.1f}ms > {self.alert_thresholds[PerformanceMetric.RESPONSE_TIME]}ms"
            )
            new_alerts.append(alert)
        
        if metrics.availability_percentage < self.alert_thresholds[PerformanceMetric.AVAILABILITY]:
            alert = PerformanceAlert(
                alert_id=f"AVL_{metrics.institution_id}_{int(time.time())}",
                timestamp=timestamp,
                severity=AlertSeverity.CRITICAL if metrics.availability_percentage < 95 else AlertSeverity.HIGH,
                metric=PerformanceMetric.AVAILABILITY,
                institution_id=metrics.institution_id,
                region=metrics.region,
                current_value=metrics.availability_percentage,
                threshold_value=self.alert_thresholds[PerformanceMetric.AVAILABILITY],
                message=f"Availability below threshold: {metrics.availability_percentage:.1f}% < {self.alert_thresholds[PerformanceMetric.AVAILABILITY]}%"
            )
            new_alerts.append(alert)
        
        for alert in new_alerts:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
        
        if new_alerts:
            logger.warning(f"Generated {len(new_alerts)} alerts for {metrics.institution_id}")
        
        return new_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now().isoformat()
                self.active_alerts.remove(alert)
                logger.info(f"Alert resolved: {alert_id}")
                return True
        
        logger.warning(f"Alert not found for resolution: {alert_id}")
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[PerformanceAlert]:
        """Get active alerts, optionally filtered by severity"""
        if severity:
            return [alert for alert in self.active_alerts if alert.severity == severity]
        return self.active_alerts.copy()
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of active alerts by severity"""
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.active_alerts:
            summary[alert.severity.value] += 1
        
        return summary

class GlobalPerformanceAnalytics:
    """Advanced analytics for global performance monitoring"""
    
    def __init__(self):
        self.trend_window = 24  # hours
        logger.info("Global performance analytics initialized")
    
    def calculate_regional_summary(self, institution_metrics: List[InstitutionMetrics]) -> List[RegionalSummary]:
        """Calculate regional performance summaries"""
        
        regional_data = defaultdict(list)
        
        for metrics in institution_metrics:
            regional_data[metrics.region].append(metrics)
        
        regional_summaries = []
        
        for region, metrics_list in regional_data.items():
            if not metrics_list:
                continue
            
            summary = RegionalSummary(
                region=region,
                institution_count=len(metrics_list),
                total_cases_processed=sum(m.cases_processed_today for m in metrics_list),
                avg_accuracy=np.mean([m.accuracy for m in metrics_list]),
                avg_response_time=np.mean([m.response_time_ms for m in metrics_list]),
                total_alerts=sum(m.alerts_active for m in metrics_list),
                critical_alerts=0,  # Would be calculated from alert system
                availability_percentage=np.mean([m.availability_percentage for m in metrics_list]),
                timestamp=datetime.now().isoformat()
            )
            
            regional_summaries.append(summary)
        
        return regional_summaries
    
    def calculate_global_metrics(self, institution_metrics: List[InstitutionMetrics]) -> Dict[str, float]:
        """Calculate global performance metrics"""
        
        if not institution_metrics:
            return {}
        
        global_metrics = {
            "global_accuracy": np.mean([m.accuracy for m in institution_metrics]),
            "global_sensitivity": np.mean([m.sensitivity for m in institution_metrics]),
            "global_specificity": np.mean([m.specificity for m in institution_metrics]),
            "global_auc_roc": np.mean([m.auc_roc for m in institution_metrics]),
            "global_response_time": np.mean([m.response_time_ms for m in institution_metrics]),
            "global_throughput": sum(m.throughput_per_hour for m in institution_metrics),
            "global_availability": np.mean([m.availability_percentage for m in institution_metrics]),
            "total_institutions": len(institution_metrics),
            "total_cases_today": sum(m.cases_processed_today for m in institution_metrics)
        }
        
        return global_metrics
    
    def calculate_performance_trends(self, metrics_history: Dict[str, List[InstitutionMetrics]]) -> Dict[str, List[float]]:
        """Calculate performance trends over time"""
        
        trends = {}
        
        if not metrics_history:
            return trends
        
        all_metrics_by_time = defaultdict(list)
        
        for institution_id, metrics_list in metrics_history.items():
            for metrics in metrics_list:
                all_metrics_by_time[metrics.timestamp].append(metrics)
        
        timestamps = sorted(all_metrics_by_time.keys())
        
        accuracy_trend = []
        response_time_trend = []
        availability_trend = []
        
        for timestamp in timestamps:
            metrics_at_time = all_metrics_by_time[timestamp]
            
            if metrics_at_time:
                accuracy_trend.append(np.mean([m.accuracy for m in metrics_at_time]))
                response_time_trend.append(np.mean([m.response_time_ms for m in metrics_at_time]))
                availability_trend.append(np.mean([m.availability_percentage for m in metrics_at_time]))
        
        trends = {
            "accuracy_trend": accuracy_trend[-24:],  # Last 24 data points
            "response_time_trend": response_time_trend[-24:],
            "availability_trend": availability_trend[-24:]
        }
        
        return trends
    
    def calculate_system_health_score(self, global_metrics: Dict[str, float], 
                                    active_alerts: List[PerformanceAlert]) -> float:
        """Calculate overall system health score (0-100)"""
        
        if not global_metrics:
            return 0.0
        
        performance_score = 0.0
        
        if "global_accuracy" in global_metrics:
            performance_score += min(global_metrics["global_accuracy"] / 0.90, 1.0) * 30
        
        if "global_availability" in global_metrics:
            performance_score += min(global_metrics["global_availability"] / 99.0, 1.0) * 25
        
        if "global_response_time" in global_metrics:
            response_score = max(1.0 - (global_metrics["global_response_time"] - 200) / 300, 0.0)
            performance_score += response_score * 20
        
        alert_penalty = 0
        for alert in active_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                alert_penalty += 10
            elif alert.severity == AlertSeverity.HIGH:
                alert_penalty += 5
            elif alert.severity == AlertSeverity.MEDIUM:
                alert_penalty += 2
        
        reliability_score = 25
        
        final_score = max(performance_score + reliability_score - alert_penalty, 0.0)
        return min(final_score, 100.0)

class GlobalPerformanceMonitoringSystem:
    """Main global performance monitoring system"""
    
    def __init__(self, collection_interval: int = 60):
        self.metrics_collector = RealTimeMetricsCollector(collection_interval)
        self.alert_system = PerformanceAlertSystem()
        self.analytics = GlobalPerformanceAnalytics()
        self.is_monitoring = False
        
        logger.info("Global performance monitoring system initialized")
    
    def start_monitoring(self):
        """Start global performance monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.metrics_collector.start_collection()
        
        threading.Thread(target=self._alert_monitoring_loop, daemon=True).start()
        
        logger.info("🌍 Global performance monitoring started")
    
    def stop_monitoring(self):
        """Stop global performance monitoring"""
        self.is_monitoring = False
        self.metrics_collector.stop_collection()
        
        logger.info("Global performance monitoring stopped")
    
    def _alert_monitoring_loop(self):
        """Monitor metrics and generate alerts"""
        while self.is_monitoring:
            try:
                for inst_id in ["INST_USA_001", "INST_GER_001", "INST_JPN_001", 
                              "INST_CAN_001", "INST_UK_001", "INST_AUS_001"]:
                    latest_metrics = self.metrics_collector.get_latest_metrics(inst_id)
                    if latest_metrics:
                        self.alert_system.check_metrics_for_alerts(latest_metrics)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                time.sleep(60)
    
    def get_global_dashboard(self) -> GlobalDashboard:
        """Get comprehensive global dashboard data"""
        
        current_metrics = []
        metrics_history = {}
        
        for inst_id in ["INST_USA_001", "INST_GER_001", "INST_JPN_001", 
                       "INST_CAN_001", "INST_UK_001", "INST_AUS_001"]:
            latest = self.metrics_collector.get_latest_metrics(inst_id)
            if latest:
                current_metrics.append(latest)
            
            history = self.metrics_collector.get_metrics_history(inst_id, 24)
            if history:
                metrics_history[inst_id] = history
        
        global_metrics = self.analytics.calculate_global_metrics(current_metrics)
        regional_summaries = self.analytics.calculate_regional_summary(current_metrics)
        performance_trends = self.analytics.calculate_performance_trends(metrics_history)
        
        active_alerts = self.alert_system.get_active_alerts()
        
        system_health_score = self.analytics.calculate_system_health_score(
            global_metrics, active_alerts
        )
        
        dashboard = GlobalDashboard(
            dashboard_timestamp=datetime.now().isoformat(),
            global_metrics=global_metrics,
            regional_summaries=regional_summaries,
            institution_metrics=current_metrics,
            active_alerts=active_alerts,
            performance_trends=performance_trends,
            system_health_score=system_health_score
        )
        
        return dashboard
    
    def export_dashboard_data(self, output_path: str) -> bool:
        """Export dashboard data to file"""
        
        try:
            dashboard = self.get_global_dashboard()
            dashboard_data = asdict(dashboard)
            
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"✅ Dashboard data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export dashboard data: {e}")
            return False

def main():
    """Example usage of global performance monitoring system"""
    
    logger.info("🌍 Global Performance Monitoring System - Phase 9 Example")
    
    monitoring_system = GlobalPerformanceMonitoringSystem(collection_interval=10)
    
    monitoring_system.start_monitoring()
    
    logger.info("Collecting performance data...")
    time.sleep(30)
    
    dashboard = monitoring_system.get_global_dashboard()
    
    logger.info(f"Global System Health Score: {dashboard.system_health_score:.1f}/100")
    logger.info(f"Active Alerts: {len(dashboard.active_alerts)}")
    logger.info(f"Institutions Monitored: {len(dashboard.institution_metrics)}")
    logger.info(f"Regional Summaries: {len(dashboard.regional_summaries)}")
    
    monitoring_system.export_dashboard_data("/tmp/global_performance_dashboard.json")
    
    monitoring_system.stop_monitoring()
    
    logger.info("🎉 Global performance monitoring example completed")

if __name__ == "__main__":
    main()
