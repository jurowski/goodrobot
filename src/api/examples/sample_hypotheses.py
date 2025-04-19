"""
Sample hypotheses for testing the Confidence Correlations platform.
"""

from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from ..models.research import (
    ResearchHypothesis,
    ExperimentGroup,
    ParticipantCriteria,
    ExperimentStatus
)

async def create_sample_hypotheses():
    """Create sample hypotheses in the database."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.goodrobot
    collection = db.research_hypotheses

    # Sample 1: Meditation and Focus
    meditation_hypothesis = ResearchHypothesis(
        title="Impact of Daily Meditation on Focus and Productivity",
        description="Investigating how consistent daily meditation practice affects workplace focus and productivity metrics",
        researcher_id="researcher_001",
        status=ExperimentStatus.RECRUITING,
        hypothesis="Daily meditation practice of 20 minutes will improve focus duration and task completion rates",
        independent_variable="Daily meditation duration",
        dependent_variable="Focus metrics (task completion, error rates)",
        control_group=ExperimentGroup(
            name="Non-meditation group",
            description="Participants maintain normal daily routine",
            intervention="No meditation practice",
            measurement_method="Daily focus tests and productivity tracking",
            sample_size=50
        ),
        experimental_group=ExperimentGroup(
            name="Meditation group",
            description="Participants practice guided meditation",
            intervention="20-minute daily guided meditation",
            measurement_method="Daily focus tests and productivity tracking",
            sample_size=50
        ),
        participant_criteria=ParticipantCriteria(
            min_age=21,
            max_age=65,
            required_conditions=[],
            excluded_conditions=["diagnosed attention disorders"],
            location_requirements=["remote work environment"],
            other_requirements=["full-time knowledge worker"]
        ),
        duration_days=30,
        required_participants=100,
        methodology="Randomized controlled trial with daily measurements",
        expected_outcomes="20% improvement in focus duration and task completion",
        measurement_tools=["Focus app tracker", "Productivity metrics", "Self-assessment"],
        ethical_considerations="Voluntary participation, data privacy",
        tags=["meditation", "productivity", "focus", "workplace"]
    )

    # Sample 2: Sleep Quality
    sleep_hypothesis = ResearchHypothesis(
        title="Correlation Between Sleep Quality and Creative Problem Solving",
        description="Examining how sleep quality affects creative problem-solving abilities",
        researcher_id="researcher_002",
        status=ExperimentStatus.PROPOSED,
        hypothesis="Higher sleep quality leads to improved creative problem-solving capabilities",
        independent_variable="Sleep quality score",
        dependent_variable="Creative problem-solving test scores",
        control_group=ExperimentGroup(
            name="Normal sleep group",
            description="Participants maintain usual sleep patterns",
            intervention="No sleep intervention",
            measurement_method="Daily sleep tracking and creativity tests",
            sample_size=40
        ),
        experimental_group=ExperimentGroup(
            name="Sleep hygiene group",
            description="Participants follow sleep optimization protocol",
            intervention="Structured sleep hygiene program",
            measurement_method="Daily sleep tracking and creativity tests",
            sample_size=40
        ),
        participant_criteria=ParticipantCriteria(
            min_age=18,
            max_age=60,
            required_conditions=[],
            excluded_conditions=["sleep disorders", "shift work"],
            location_requirements=[],
            other_requirements=["regular work schedule"]
        ),
        duration_days=45,
        required_participants=80,
        methodology="Controlled study with sleep tracking and daily assessments",
        expected_outcomes="30% improvement in creative problem-solving scores",
        measurement_tools=["Sleep tracker", "Creativity assessment", "Daily log"],
        ethical_considerations="Non-invasive monitoring, regular check-ins",
        tags=["sleep", "creativity", "problem-solving", "cognitive"]
    )

    # Sample 3: Exercise and Stress
    exercise_hypothesis = ResearchHypothesis(
        title="High-Intensity Exercise Impact on Stress Resilience",
        description="Studying the relationship between regular high-intensity exercise and stress response",
        researcher_id="researcher_003",
        status=ExperimentStatus.IN_PROGRESS,
        hypothesis="Regular high-intensity exercise increases stress resilience",
        independent_variable="Exercise intensity and frequency",
        dependent_variable="Stress resilience metrics",
        control_group=ExperimentGroup(
            name="Light activity group",
            description="Participants maintain light physical activity",
            intervention="Walking 30 minutes daily",
            measurement_method="Stress tests and physiological markers",
            sample_size=30
        ),
        experimental_group=ExperimentGroup(
            name="High-intensity group",
            description="Participants perform HIIT workouts",
            intervention="30-minute HIIT 3x weekly",
            measurement_method="Stress tests and physiological markers",
            sample_size=30
        ),
        participant_criteria=ParticipantCriteria(
            min_age=25,
            max_age=45,
            required_conditions=["physically healthy"],
            excluded_conditions=["cardiovascular conditions", "injuries"],
            location_requirements=["access to fitness facility"],
            other_requirements=["sedentary job"]
        ),
        duration_days=60,
        required_participants=60,
        methodology="Randomized intervention study with regular assessments",
        expected_outcomes="40% improvement in stress resilience metrics",
        measurement_tools=["Heart rate variability", "Cortisol levels", "Stress assessment"],
        ethical_considerations="Health screening, supervised exercise",
        tags=["exercise", "stress", "HIIT", "resilience"]
    )

    # Insert hypotheses into database
    await collection.insert_many([
        meditation_hypothesis.dict(),
        sleep_hypothesis.dict(),
        exercise_hypothesis.dict()
    ])

    print("Sample hypotheses created successfully!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_sample_hypotheses()) 