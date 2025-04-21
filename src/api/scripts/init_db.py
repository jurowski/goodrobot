"""
Database initialization script.
"""

import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
sys.path.append(project_root)

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import CollectionInvalid
from src.api.config.mongodb import connect_to_mongodb, close_mongodb_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_collections(db):
    """Create collections and indexes."""
    try:
        # Collections to create
        collections = ["experiments", "participants", "measurements", "results", "peer_reviews"]
        
        # Create collections if they don't exist
        for collection in collections:
            try:
                await db.create_collection(collection)
                logger.info(f"Created collection: {collection}")
            except CollectionInvalid:
                logger.info(f"Collection {collection} already exists")
        
        # Create indexes
        index_operations = [
            (db.experiments, [("researcher_id", 1)]),
            (db.participants, [("experiment_id", 1)]),
            (db.measurements, [("participant_id", 1), ("experiment_id", 1)]),
            (db.results, [("experiment_id", 1), ("researcher_id", 1)]),
            (db.peer_reviews, [("experiment_id", 1), ("reviewer_id", 1)])
        ]
        
        for collection, index in index_operations:
            try:
                await collection.create_index(index)
                logger.info(f"Created index for {collection.name}: {index}")
            except Exception as e:
                logger.warning(f"Failed to create index for {collection.name}: {e}")
        
        logger.info("Collections and indexes setup completed")
    except Exception as e:
        logger.error(f"Error in database setup: {e}")
        raise

async def init_db():
    """Initialize the database with required collections and indexes."""
    try:
        # Connect to MongoDB
        client = await connect_to_mongodb()
        db = client.goodrobot
        
        # Create collections and indexes
        await create_collections(db)
        
        # Close connection
        await close_mongodb_connection()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db()) 