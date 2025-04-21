from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from ..models.research import (
    ResearchHypothesis,
    ParticipantData,
    ResearchResult,
    ExperimentStatus
)
from ..database.research_db import ResearchDatabase
from ..dependencies import get_research_db

router = APIRouter(prefix="/research", tags=["research"])

# Initialize database
db = ResearchDatabase()

@router.post("/hypotheses", response_model=str)
async def create_hypothesis(
    hypothesis: ResearchHypothesis,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Create a new research hypothesis."""
    try:
        hypothesis_id = await db.create_hypothesis(hypothesis)
        return hypothesis_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hypotheses/{hypothesis_id}", response_model=ResearchHypothesis)
async def get_hypothesis(
    hypothesis_id: str,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Get a specific research hypothesis."""
    hypothesis = await db.get_hypothesis(hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return hypothesis

@router.get("/hypotheses", response_model=List[ResearchHypothesis])
async def list_hypotheses(
    status: Optional[ExperimentStatus] = None,
    tags: Optional[List[str]] = Query(None),
    skip: int = 0,
    limit: int = 20,
    db: ResearchDatabase = Depends(get_research_db)
):
    """List research hypotheses with optional filtering."""
    try:
        return await db.list_hypotheses(status, tags, skip, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hypotheses/{hypothesis_id}/participate", response_model=str)
async def join_experiment(
    hypothesis_id: str,
    participant_id: str,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Join an experiment as a participant."""
    hypothesis = await db.get_hypothesis(hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    
    if hypothesis.status != ExperimentStatus.RECRUITING:
        raise HTTPException(status_code=400, detail="This study is not currently recruiting")
    
    participant_data = ParticipantData(
        hypothesis_id=hypothesis_id,
        participant_id=participant_id,
        group="pending"  # Will be assigned to control or experimental
    )
    
    try:
        return await db.add_participant(participant_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/data/{hypothesis_id}", response_model=bool)
async def submit_measurement(
    hypothesis_id: str,
    participant_id: str,
    measurements: dict,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Submit measurement data for an experiment."""
    return await db.update_participant_data(
        participant_id,
        hypothesis_id,
        measurements
    )

@router.post("/hypotheses/{hypothesis_id}/complete", response_model=bool)
async def complete_participation(
    hypothesis_id: str,
    participant_id: str,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Mark participation in an experiment as complete."""
    return await db.complete_participant_data(
        participant_id,
        hypothesis_id
    )

@router.post("/results/{hypothesis_id}", response_model=str)
async def submit_results(
    hypothesis_id: str,
    result: ResearchResult,
    researcher_id: str,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Submit results for a completed experiment."""
    hypothesis = await db.get_hypothesis(hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    
    if hypothesis.researcher_id != researcher_id:
        raise HTTPException(status_code=403, detail="Not authorized to submit results")
    
    if hypothesis.status != ExperimentStatus.IN_PROGRESS:
        raise HTTPException(status_code=400, detail="Experiment is not in progress")
    
    return await db.create_result(result)

@router.post("/results/{result_id}/review", response_model=bool)
async def submit_peer_review(
    result_id: str,
    reviewer_id: str,
    comments: dict,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Submit a peer review for experiment results."""
    return await db.update_peer_review(
        result_id,
        reviewer_id,
        comments
    )

@router.get("/search", response_model=List[ResearchHypothesis])
async def search_hypotheses(
    query: str,
    status: Optional[ExperimentStatus] = None,
    tags: Optional[List[str]] = Query(None),
    skip: int = 0,
    limit: int = 20,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Search for research hypotheses."""
    return await db.search_hypotheses(query, status, tags, skip, limit)

@router.patch("/hypotheses/{hypothesis_id}")
async def update_hypothesis(
    hypothesis_id: str,
    updates: dict,
    db: ResearchDatabase = Depends(get_research_db)
):
    """Update a research hypothesis."""
    success = await db.update_hypothesis(hypothesis_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return {"status": "success"} 