"""
Research models for participant data and metrics.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class ExperimentStatus(str, Enum):
    """Status of an experiment/research hypothesis."""
    DRAFT = "draft"
    RECRUITING = "recruiting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class PeerReviewStatus(str, Enum):
    """Status of peer review for research results."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVISIONS_NEEDED = "revisions_needed"

class ParticipantStatus(str, Enum):
    """Status of a participant in a study."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"

class ParticipantMetrics(BaseModel):
    """Daily metrics recorded for a participant."""
    date: datetime
    metrics: Dict[str, float]

class ResearchResult(BaseModel):
    """Research results model."""
    id: str
    hypothesis_id: str
    completion_date: datetime
    peer_review_status: PeerReviewStatus = PeerReviewStatus.PENDING
    findings: Dict[str, Any]
    statistical_analysis: Optional[Dict[str, Any]] = None
    conclusions: List[str]
    limitations: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    reviewer_comments: Optional[List[Dict[str, str]]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "result_meditation_001",
                "hypothesis_id": "meditation_001",
                "completion_date": "2024-02-01T00:00:00",
                "peer_review_status": "pending",
                "findings": {
                    "focus_improvement": {
                        "experimental_group": 27.5,
                        "control_group": 12.3,
                        "p_value": 0.023
                    },
                    "stress_reduction": {
                        "experimental_group": -31.2,
                        "control_group": -8.7,
                        "p_value": 0.015
                    }
                },
                "statistical_analysis": {
                    "method": "t_test",
                    "confidence_interval": 0.95
                },
                "conclusions": [
                    "Daily meditation significantly improved focus metrics",
                    "Stress levels showed marked reduction in experimental group"
                ],
                "limitations": [
                    "Small sample size",
                    "Limited study duration"
                ],
                "recommendations": [
                    "Increase sample size in future studies",
                    "Consider longer study duration"
                ]
            }
        }
    )

class ResearchHypothesis(BaseModel):
    """Research hypothesis model."""
    id: str
    title: str
    description: str
    researcher_id: str
    status: ExperimentStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    target_participants: int
    duration_days: int
    metrics_config: Dict[str, Dict[str, float]]
    tags: List[str] = []
    notes: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "meditation_001",
                "title": "Impact of Daily Meditation on Focus",
                "description": "Investigating the effects of 20-minute daily meditation on work focus and stress levels",
                "researcher_id": "researcher_001",
                "status": "in_progress",
                "created_at": "2024-01-01T00:00:00",
                "target_participants": 10,
                "duration_days": 30,
                "metrics_config": {
                    "focus_duration": {
                        "min": 0,
                        "max": 60
                    },
                    "task_completion_rate": {
                        "min": 0,
                        "max": 100
                    },
                    "stress_level": {
                        "min": 1,
                        "max": 10
                    }
                },
                "tags": ["meditation", "focus", "stress", "productivity"]
            }
        }
    )

class ParticipantData(BaseModel):
    """Participant data model."""
    participant_id: str
    hypothesis_id: str
    group: str
    status: ParticipantStatus
    start_date: datetime
    end_date: Optional[datetime] = None
    demographics: Dict[str, str]
    daily_metrics: List[ParticipantMetrics]
    notes: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "participant_id": "meditation_001_participant_1",
                "hypothesis_id": "meditation_001",
                "group": "experimental",
                "status": "active",
                "start_date": "2024-01-01T00:00:00",
                "demographics": {
                    "age": "30",
                    "gender": "female",
                    "occupation": "engineer"
                },
                "daily_metrics": [
                    {
                        "date": "2024-01-01T00:00:00",
                        "metrics": {
                            "focus_duration": 35.5,
                            "task_completion_rate": 80.0,
                            "meditation_minutes": 20,
                            "stress_level": 6.5
                        }
                    }
                ]
            }
        }
    ) 