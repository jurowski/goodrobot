"""
FastAPI dependency functions.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from .database.research_db import ResearchDatabase
from typing import Optional

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Get the current authenticated user."""
    # TODO: Implement proper JWT token validation
    return {"id": "test-user-id", "username": "test-user"}

async def get_research_db():
    """Dependency to get ResearchDatabase instance."""
    db = ResearchDatabase()
    await db.initialize()
    return db 