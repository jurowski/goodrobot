"""
MongoDB configuration and connection management.
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "goodrobot")
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

# Collection names
COLLECTION_HYPOTHESES = "research_hypotheses"
COLLECTION_PARTICIPANTS = "participants"
COLLECTION_RESULTS = "research_results"

# MongoDB client instance
client: Optional[AsyncIOMotorClient] = None
db = None

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
    """Initialize MongoDB connection with enhanced error handling."""
    global client, db
    try:
        # Validate environment variables
        if not MONGODB_URL:
            raise MongoDBConnectionError("MONGODB_URL environment variable is not set")
        
        # Construct connection string with authentication if credentials are provided
        if MONGODB_USERNAME and MONGODB_PASSWORD:
            # Check if it's a MongoDB Atlas URL
            if "mongodb+srv://" in MONGODB_URL:
                connection_string = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_URL.split('mongodb+srv://')[1]}"
            else:
                # For standard MongoDB URLs
                connection_string = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_URL.split('mongodb://')[1]}"
        else:
            connection_string = MONGODB_URL
        
        logger.info(f"Connecting to MongoDB at {MONGODB_URL}")
        logger.debug(f"Using connection string: {connection_string}")
        
        # Initialize client with connection timeout and retry settings
        client = AsyncIOMotorClient(
            connection_string,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            retryWrites=True,
            retryReads=True
        )
        
        # Test the connection
        try:
            await client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB at {MONGODB_URL}")
        except Exception as e:
            raise MongoDBConnectionError(f"Failed to ping MongoDB server: {str(e)}")
        
        db = client[MONGODB_DB_NAME]
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        raise MongoDBConnectionError(f"Failed to connect to MongoDB: {str(e)}")

async def create_indexes():
    """Create necessary indexes for collections with enhanced error handling."""
    try:
        # Indexes for hypotheses collection
        await db[COLLECTION_HYPOTHESES].create_index("id", unique=True)
        await db[COLLECTION_HYPOTHESES].create_index("status")
        await db[COLLECTION_HYPOTHESES].create_index("researcher_id")
        await db[COLLECTION_HYPOTHESES].create_index("created_at")
        await db[COLLECTION_HYPOTHESES].create_index("tags")
        await db[COLLECTION_HYPOTHESES].create_index([("title", "text"), ("description", "text")])
        
        # Indexes for participants collection
        await db[COLLECTION_PARTICIPANTS].create_index("participant_id", unique=True)
        await db[COLLECTION_PARTICIPANTS].create_index("hypothesis_id")
        await db[COLLECTION_PARTICIPANTS].create_index("status")
        await db[COLLECTION_PARTICIPANTS].create_index("group")
        await db[COLLECTION_PARTICIPANTS].create_index("start_date")
        await db[COLLECTION_PARTICIPANTS].create_index("end_date")
        await db[COLLECTION_PARTICIPANTS].create_index([("hypothesis_id", 1), ("status", 1)])
        await db[COLLECTION_PARTICIPANTS].create_index([("hypothesis_id", 1), ("group", 1)])
        
        # Indexes for results collection
        await db[COLLECTION_RESULTS].create_index("id", unique=True)
        await db[COLLECTION_RESULTS].create_index("hypothesis_id")
        await db[COLLECTION_RESULTS].create_index("peer_review_status")
        await db[COLLECTION_RESULTS].create_index("completion_date")
        await db[COLLECTION_RESULTS].create_index([("hypothesis_id", 1), ("peer_review_status", 1)])
        
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        raise MongoDBIndexError(f"Failed to create MongoDB indexes: {str(e)}")

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
    """Close MongoDB connection with error handling."""
    global client
    try:
        if client:
            client.close()
            logger.info("MongoDB connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")
        raise MongoDBConnectionError(f"Failed to close MongoDB connection: {str(e)}")

def get_db():
    """Get database instance with validation."""
    if db is None:
        raise MongoDBConnectionError("Database not initialized. Call connect_to_mongodb() first.")
    return db 