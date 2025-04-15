import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
from dataclasses import dataclass, field

class MetricType(Enum):
    SAFETY = "safety"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    USER = "user"
    SYSTEM = "system"

@dataclass
class VisualizationConfig:
    """Configuration for visualization components"""
    title: str
    metric_type: MetricType
    update_interval: int = 5  # seconds
    max_points: int = 100
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#3498db',
        'secondary': '#2ecc71',
        'warning': '#f1c40f',
        'danger': '#e74c3c',
        'neutral': '#95a5a6'
    })

class MonitoringVisualization:
    """Advanced visualization system for monitoring data"""
    
    def __init__(self):
        self.configs: Dict[str, VisualizationConfig] = {}
        self.data_buffers: Dict[str, List[Dict]] = {}
        self.figures: Dict[str, go.Figure] = {}
        self._initialize_default_configs()
        
    def _initialize_default_configs(self):
        """Initialize default visualization configurations"""
        self.configs['safety_metrics'] = VisualizationConfig(
            title="Safety Metrics Overview",
            metric_type=MetricType.SAFETY
        )
        
        self.configs['system_performance'] = VisualizationConfig(
            title="System Performance",
            metric_type=MetricType.PERFORMANCE
        )
        
        self.configs['resource_usage'] = VisualizationConfig(
            title="Resource Utilization",
            metric_type=MetricType.RESOURCE
        )
        
        self.configs['user_activity'] = VisualizationConfig(
            title="User Activity",
            metric_type=MetricType.USER
        )
        
    def create_dashboard(self) -> go.Figure:
        """Create main monitoring dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Safety Metrics',
                'System Performance',
                'Resource Usage',
                'User Activity'
            )
        )
        
        # Add safety metrics subplot
        safety_data = self._get_safety_metrics()
        fig.add_trace(
            go.Scatter(
                x=safety_data['timestamp'],
                y=safety_data['safety_score'],
                name='Safety Score',
                line=dict(color=self.configs['safety_metrics'].color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        # Add performance subplot
        perf_data = self._get_performance_metrics()
        fig.add_trace(
            go.Scatter(
                x=perf_data['timestamp'],
                y=perf_data['response_time'],
                name='Response Time',
                line=dict(color=self.configs['system_performance'].color_scheme['primary'])
            ),
            row=1, col=2
        )
        
        # Add resource usage subplot
        resource_data = self._get_resource_metrics()
        fig.add_trace(
            go.Bar(
                x=resource_data['resource'],
                y=resource_data['usage'],
                name='Resource Usage',
                marker_color=self.configs['resource_usage'].color_scheme['primary']
            ),
            row=2, col=1
        )
        
        # Add user activity subplot
        user_data = self._get_user_metrics()
        fig.add_trace(
            go.Scatter(
                x=user_data['timestamp'],
                y=user_data['active_users'],
                name='Active Users',
                line=dict(color=self.configs['user_activity'].color_scheme['primary'])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        return fig
        
    def create_safety_dashboard(self) -> go.Figure:
        """Create detailed safety monitoring dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Safety Violations',
                'Risk Assessment',
                'Compliance Metrics',
                'Safety Trends'
            )
        )
        
        # Add safety violations heatmap
        violations_data = self._get_safety_violations()
        fig.add_trace(
            go.Heatmap(
                z=violations_data['values'],
                x=violations_data['x'],
                y=violations_data['y'],
                colorscale='RdYlGn_r'
            ),
            row=1, col=1
        )
        
        # Add risk assessment gauge
        risk_data = self._get_risk_assessment()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data['risk_score'],
                gauge={'axis': {'range': [0, 100]}},
                title={'text': "Risk Level"}
            ),
            row=1, col=2
        )
        
        # Add compliance metrics
        compliance_data = self._get_compliance_metrics()
        fig.add_trace(
            go.Bar(
                x=compliance_data['metric'],
                y=compliance_data['value'],
                name='Compliance'
            ),
            row=2, col=1
        )
        
        # Add safety trends
        trend_data = self._get_safety_trends()
        fig.add_trace(
            go.Scatter(
                x=trend_data['timestamp'],
                y=trend_data['trend'],
                name='Safety Trend'
            ),
            row=2, col=2
        )
        
        return fig
        
    def create_performance_dashboard(self) -> go.Figure:
        """Create detailed performance monitoring dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Response Time Distribution',
                'Error Rates',
                'Resource Utilization',
                'Throughput'
            )
        )
        
        # Add response time histogram
        response_data = self._get_response_times()
        fig.add_trace(
            go.Histogram(
                x=response_data,
                name='Response Times'
            ),
            row=1, col=1
        )
        
        # Add error rates line plot
        error_data = self._get_error_rates()
        fig.add_trace(
            go.Scatter(
                x=error_data['timestamp'],
                y=error_data['error_rate'],
                name='Error Rate'
            ),
            row=1, col=2
        )
        
        # Add resource utilization stacked area
        resource_data = self._get_resource_utilization()
        fig.add_trace(
            go.Scatter(
                x=resource_data['timestamp'],
                y=resource_data['cpu'],
                name='CPU',
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Add throughput bar plot
        throughput_data = self._get_throughput_metrics()
        fig.add_trace(
            go.Bar(
                x=throughput_data['timestamp'],
                y=throughput_data['requests'],
                name='Throughput'
            ),
            row=2, col=2
        )
        
        return fig
        
    def update_visualization(self, metric_type: MetricType, data: Dict):
        """Update visualization with new data"""
        buffer_key = metric_type.value
        
        if buffer_key not in self.data_buffers:
            self.data_buffers[buffer_key] = []
            
        self.data_buffers[buffer_key].append({
            'timestamp': datetime.now(),
            **data
        })
        
        # Maintain buffer size
        config = self._get_config_for_metric(metric_type)
        if len(self.data_buffers[buffer_key]) > config.max_points:
            self.data_buffers[buffer_key].pop(0)
            
        # Update corresponding figure
        self._update_figure(metric_type)
        
    def _get_config_for_metric(self, metric_type: MetricType) -> VisualizationConfig:
        """Get configuration for metric type"""
        for config in self.configs.values():
            if config.metric_type == metric_type:
                return config
        return VisualizationConfig(
            title=f"{metric_type.value} Metrics",
            metric_type=metric_type
        )
        
    def _update_figure(self, metric_type: MetricType):
        """Update specific figure with new data"""
        if metric_type == MetricType.SAFETY:
            self.figures['safety'] = self.create_safety_dashboard()
        elif metric_type == MetricType.PERFORMANCE:
            self.figures['performance'] = self.create_performance_dashboard()
        elif metric_type == MetricType.RESOURCE:
            self._update_resource_figure()
        elif metric_type == MetricType.USER:
            self._update_user_figure()
            
    def export_dashboard(self, format: str = 'html') -> str:
        """Export dashboard in specified format"""
        fig = self.create_dashboard()
        
        if format == 'html':
            return fig.to_html()
        elif format == 'json':
            return fig.to_json()
        elif format == 'image':
            return fig.to_image(format='png')
        else:
            raise ValueError(f"Unsupported export format: {format}")
