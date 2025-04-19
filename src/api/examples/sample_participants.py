"""
Generate sample participant data for research hypotheses.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio
import os
import logging

from src.api.models.research import ParticipantData, ParticipantStatus, ParticipantMetrics
from src.api.config.mongodb import (
    connect_to_mongodb,
    get_db,
    COLLECTION_PARTICIPANTS,
    validate_document,
    MongoDBConnectionError,
    MongoDBValidationError,
    close_mongodb_connection
)

# Configure logging
logger = logging.getLogger(__name__)

def generate_meditation_data(day: int, is_experimental: bool) -> Dict[str, float]:
    """Generate daily meditation study metrics with realistic progression."""
    try:
        base_focus = 30 + (day * 0.5 if is_experimental else day * 0.2)
        base_completion = 75 + (day * 0.3 if is_experimental else day * 0.1)
        
        return {
            "focus_duration": max(20, min(60, base_focus + random.uniform(-5, 5))),
            "task_completion_rate": max(60, min(95, base_completion + random.uniform(-3, 3))),
            "meditation_minutes": 20 if is_experimental else 0,
            "stress_level": max(1, min(10, 7 - (day * 0.1 if is_experimental else day * 0.05) + random.uniform(-1, 1)))
        }
    except Exception as e:
        logger.error(f"Error generating meditation data: {e}")
        raise

def generate_sleep_data(day: int, is_experimental: bool) -> Dict[str, float]:
    """Generate daily sleep study metrics with realistic progression."""
    try:
        base_quality = 6.5 + (day * 0.05 if is_experimental else day * 0.02)
        base_creativity = 70 + (day * 0.4 if is_experimental else day * 0.15)
        
        return {
            "sleep_quality": max(3, min(10, base_quality + random.uniform(-1, 1))),
            "creativity_score": max(50, min(100, base_creativity + random.uniform(-5, 5))),
            "problem_solving_time": max(10, 45 - (day * 0.3 if is_experimental else day * 0.1) + random.uniform(-3, 3)),
            "rem_sleep_percentage": max(15, min(30, 20 + (day * 0.1 if is_experimental else day * 0.05) + random.uniform(-2, 2)))
        }
    except Exception as e:
        logger.error(f"Error generating sleep data: {e}")
        raise

def generate_exercise_data(day: int, is_experimental: bool) -> Dict[str, float]:
    """Generate daily exercise study metrics with realistic progression."""
    try:
        base_resilience = 65 + (day * 0.5 if is_experimental else day * 0.2)
        base_recovery = 75 + (day * 0.3 if is_experimental else day * 0.1)
        
        return {
            "exercise_intensity": 8 if is_experimental else 3,
            "exercise_duration": 45 if is_experimental else 20,
            "stress_resilience": max(50, min(95, base_resilience + random.uniform(-5, 5))),
            "recovery_rate": max(60, min(95, base_recovery + random.uniform(-3, 3))),
            "cortisol_level": max(100, 300 - (day * 2 if is_experimental else day * 0.5) + random.uniform(-20, 20))
        }
    except Exception as e:
        logger.error(f"Error generating exercise data: {e}")
        raise

async def create_sample_participants():
    """Create and store sample participant data for all hypotheses."""
    try:
        # Connect to MongoDB
        await connect_to_mongodb()
        db = get_db()
        
        # Sample demographics
        occupations = ["engineer", "teacher", "manager", "researcher", "designer", "doctor"]
        genders = ["male", "female", "non-binary"]
        
        # Study configurations
        studies = [
            {
                "id": "meditation_001",
                "duration": 30,
                "participants": 10,
                "generator": generate_meditation_data
            },
            {
                "id": "sleep_001",
                "duration": 45,
                "participants": 8,
                "generator": generate_sleep_data
            },
            {
                "id": "exercise_001",
                "duration": 60,
                "participants": 6,
                "generator": generate_exercise_data
            }
        ]
        
        total_participants = sum(study["participants"] for study in studies)
        created_participants = 0
        
        for study in studies:
            logger.info(f"Creating participants for study {study['id']}")
            
            for i in range(study["participants"]):
                try:
                    is_experimental = i < study["participants"] // 2
                    start_date = datetime.now() - timedelta(days=study["duration"])
                    
                    participant = ParticipantData(
                        hypothesis_id=study["id"],
                        participant_id=f"{study['id']}_participant_{i+1}",
                        group="experimental" if is_experimental else "control",
                        status=ParticipantStatus.ACTIVE,
                        start_date=start_date,
                        demographics={
                            "age": str(random.randint(25, 55)),
                            "gender": random.choice(genders),
                            "occupation": random.choice(occupations)
                        },
                        daily_metrics=[]
                    )
                    
                    # Generate daily metrics
                    for day in range(study["duration"]):
                        current_date = start_date + timedelta(days=day)
                        metrics = study["generator"](day, is_experimental)
                        
                        participant.daily_metrics.append(
                            ParticipantMetrics(
                                date=current_date,
                                metrics=metrics
                            )
                        )
                    
                    # Validate document before insertion
                    await validate_document(COLLECTION_PARTICIPANTS, participant.dict())
                    
                    # Store participant data
                    await db[COLLECTION_PARTICIPANTS].insert_one(participant.dict())
                    created_participants += 1
                    logger.info(f"Created participant {participant.participant_id} for study {study['id']} ({created_participants}/{total_participants})")
                    
                except MongoDBValidationError as e:
                    logger.error(f"Validation error for participant {study['id']}_participant_{i+1}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error creating participant {study['id']}_participant_{i+1}: {e}")
                    continue
        
        logger.info(f"Successfully created {created_participants}/{total_participants} participants")
        
    except MongoDBConnectionError as e:
        logger.error(f"MongoDB connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        # Close MongoDB connection
        try:
            await close_mongodb_connection()
        except MongoDBConnectionError as e:
            logger.error(f"Error closing MongoDB connection: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(create_sample_participants()) 