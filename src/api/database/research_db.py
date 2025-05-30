"""
MongoDB configuration and connection management.
"""

from typing import List, Optional
from datetime import datetime
from ..config.mongodb import get_database, connect_to_mongodb
from ..models.research import (
    ResearchHypothesis,
    ParticipantData,
    ResearchResult,
    ExperimentStatus
)
from ..config.mongodb import COLLECTION_HYPOTHESES

class ResearchDatabase:
    """Database operations for research-related data."""
    
    def __init__(self):
        """Initialize the database connection."""
        self.db = None
        self.hypotheses = None
        self.participant_data = None
        self.results = None
    
    async def initialize(self):
        """Initialize the database connection asynchronously."""
        if self.db is None:
            await connect_to_mongodb()
            self.db = get_database()
            self.hypotheses = self.db[COLLECTION_HYPOTHESES]
            self.participant_data = self.db.participant_data
            self.results = self.db.research_results
    
    async def create_hypothesis(self, hypothesis: ResearchHypothesis) -> str:
        """Create a new research hypothesis."""
        await self.initialize()
        result = await self.hypotheses.insert_one(hypothesis.dict())
        return str(result.inserted_id)

    async def get_hypothesis(self, hypothesis_id: str) -> Optional[ResearchHypothesis]:
        """Get a specific research hypothesis."""
        await self.initialize()
        result = await self.hypotheses.find_one({"id": hypothesis_id})
        return ResearchHypothesis(**result) if result else None

    async def list_hypotheses(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[ResearchHypothesis]:
        """List research hypotheses with optional filtering."""
        await self.initialize()
        query = {}
        if status:
            query["status"] = status
        if tags:
            query["tags"] = {"$all": tags}
        
        cursor = self.hypotheses.find(query).skip(skip).limit(limit)
        return [ResearchHypothesis(**doc) async for doc in cursor]

    async def update_hypothesis(self, hypothesis_id: str, updates: dict) -> bool:
        """Update a research hypothesis."""
        await self.initialize()
        result = await self.hypotheses.update_one(
            {"id": hypothesis_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    async def add_participant(self, participant_data: ParticipantData) -> str:
        """Add a participant to an experiment."""
        await self.initialize()
        # First check if we can add more participants
        hypothesis = await self.get_hypothesis(participant_data.hypothesis_id)
        if not hypothesis or hypothesis.current_participants >= hypothesis.required_participants:
            raise ValueError("Cannot add more participants to this study")

        # Update participant count
        await self.hypotheses.update_one(
            {"id": participant_data.hypothesis_id},
            {"$inc": {"current_participants": 1}}
        )

        # Add participant data
        result = await self.participant_data.insert_one(participant_data.dict())
        return str(result.inserted_id)

    async def update_participant_data(
        self,
        participant_id: str,
        hypothesis_id: str,
        measurements: dict
    ) -> bool:
        """Update participant measurement data."""
        await self.initialize()
        result = await self.participant_data.update_one(
            {
                "participant_id": participant_id,
                "hypothesis_id": hypothesis_id
            },
            {
                "$push": {"measurements": measurements},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return result.modified_count > 0

    async def complete_participant_data(
        self,
        participant_id: str,
        hypothesis_id: str
    ) -> bool:
        """Mark participant data as complete."""
        await self.initialize()
        result = await self.participant_data.update_one(
            {
                "participant_id": participant_id,
                "hypothesis_id": hypothesis_id
            },
            {
                "$set": {
                    "completed": True,
                    "end_date": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    async def create_result(self, result: ResearchResult) -> str:
        """Create a new research result."""
        await self.initialize()
        # Update hypothesis status
        await self.hypotheses.update_one(
            {"id": result.hypothesis_id},
            {"$set": {"status": ExperimentStatus.COMPLETED}}
        )
        
        # Store results
        result = await self.results.insert_one(result.dict())
        return str(result.inserted_id)

    async def update_peer_review(
        self,
        result_id: str,
        reviewer_id: str,
        comments: dict
    ) -> bool:
        """Update peer review comments for a result."""
        await self.initialize()
        result = await self.results.update_one(
            {"id": result_id},
            {
                "$push": {
                    "peer_review_comments": {
                        "reviewer_id": reviewer_id,
                        "comments": comments,
                        "timestamp": datetime.utcnow()
                    }
                }
            }
        )
        return result.modified_count > 0

    async def search_hypotheses(
        self,
        query: str,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[ResearchHypothesis]:
        """Search for research hypotheses."""
        await self.initialize()
        search_query = {
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
                {"hypothesis": {"$regex": query, "$options": "i"}},
                {"tags": {"$regex": query, "$options": "i"}}
            ]
        }
        
        if status:
            search_query["status"] = status
        if tags:
            search_query["tags"] = {"$all": tags}
        
        cursor = self.hypotheses.find(search_query).skip(skip).limit(limit)
        return [ResearchHypothesis(**doc) async for doc in cursor] 