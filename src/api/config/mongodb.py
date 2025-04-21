"""
MongoDB configuration and connection management.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, OperationFailure
import os
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "goodrobot")

# Collection names
COLLECTION_HYPOTHESES = "research_hypotheses"
COLLECTION_PARTICIPANTS = "participants"
COLLECTION_RESULTS = "research_results"
COLLECTION_SAMPLES = "samples"
COLLECTION_USERS = "users"

# Global database client
db_client: Optional[AsyncIOMotorClient] = None

class MongoDBConnectionError(Exception):
    """Custom exception for MongoDB connection errors."""
    pass

class MongoDBIndexError(Exception):
    """Custom exception for MongoDB index creation errors."""
    pass

class MongoDBValidationError(Exception):
    """Custom exception for MongoDB data validation errors."""
    pass

async def connect_to_mongodb():
    """Connect to MongoDB and initialize the database client."""
    global db_client
    
    try:
        # Create connection options
        connection_options = {
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS": 10000,
            "retryWrites": True
        }
        
        # Create a new client and connect to the server
        db_client = AsyncIOMotorClient(
            MONGODB_URL,
            **connection_options
        )
        
        # Verify the connection
        await db_client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        # Get database reference
        db = db_client[MONGODB_DB_NAME]
        
        # Create collections if they don't exist
        collections = [
            COLLECTION_HYPOTHESES,
            COLLECTION_PARTICIPANTS,
            COLLECTION_RESULTS,
            COLLECTION_SAMPLES,
            COLLECTION_USERS
        ]
        
        for collection in collections:
            if collection not in await db.list_collection_names():
                await db.create_collection(collection)
                logger.info(f"Created collection: {collection}")
        
        return db
        
    except ConnectionFailure as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise MongoDBConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred while connecting to MongoDB: {e}")
        raise MongoDBConnectionError(f"Failed to connect to MongoDB: {str(e)}")

async def create_indexes():
    """Create necessary indexes for collections."""
    try:
        if not db_client:
            raise MongoDBConnectionError("Database client not initialized")
            
        db = db_client[MONGODB_DB_NAME]
        
        # Define indexes
        indexes = {
            COLLECTION_HYPOTHESES: [("created_at", 1)],
            COLLECTION_SAMPLES: [("created_at", 1)],
            COLLECTION_USERS: [("username", 1)]
        }
        
        # Create indexes with error handling for each collection
        for collection, index_list in indexes.items():
            try:
                await db[collection].create_index(index_list)
                logger.info(f"Created index for collection {collection}")
            except OperationFailure as e:
                logger.warning(f"Failed to create index for collection {collection}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error creating index for collection {collection}: {e}")
                continue
        
        logger.info("MongoDB indexes creation completed")
    except Exception as e:
        logger.warning(f"Error during index creation: {e}")
        pass

async def validate_document(collection: str, document: dict) -> bool:
    """Validate a document before insertion."""
    try:
        # Basic validation rules
        if collection == COLLECTION_HYPOTHESES:
            required_fields = ["id", "title", "description", "researcher_id", "status"]
        elif collection == COLLECTION_PARTICIPANTS:
            required_fields = ["participant_id", "hypothesis_id", "status", "start_date"]
        elif collection == COLLECTION_RESULTS:
            required_fields = ["id", "hypothesis_id", "completion_date", "findings"]
        else:
            raise MongoDBValidationError(f"Unknown collection: {collection}")
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in document]
        if missing_fields:
            raise MongoDBValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate dates
        if "start_date" in document and not isinstance(document["start_date"], datetime):
            raise MongoDBValidationError("start_date must be a datetime object")
        if "end_date" in document and document["end_date"] is not None and not isinstance(document["end_date"], datetime):
            raise MongoDBValidationError("end_date must be a datetime object or None")
        
        return True
    except Exception as e:
        logger.error(f"Document validation failed: {e}")
        raise MongoDBValidationError(f"Document validation failed: {str(e)}")

async def close_mongodb_connection():
    """Close the MongoDB connection."""
    global db_client
    
    if db_client is not None:
        try:
            db_client.close()
            logger.info("MongoDB connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
        finally:
            db_client = None

def get_database():
    """Get the database instance."""
    global db_client
    if db_client is None:
        raise MongoDBConnectionError("Database not initialized. Call connect_to_mongodb() first.")
    return db_client[MONGODB_DB_NAME]

__all__ = [
    'connect_to_mongodb',
    'get_database',
    'close_mongodb_connection',
    'validate_document',
    'MongoDBConnectionError',
    'MongoDBIndexError',
    'MongoDBValidationError',
    'COLLECTION_HYPOTHESES',
    'COLLECTION_PARTICIPANTS',
    'COLLECTION_RESULTS',
    'COLLECTION_SAMPLES',
    'COLLECTION_USERS'
] 