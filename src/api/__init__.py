"""
API module for GoodRobot.
This module provides the REST API interface for the GoodRobot system.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log important paths
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Static directory exists: {os.path.exists('static')}")
logger.info(f"Utilities directory exists: {os.path.exists('utils')}")
logger.info(f"Speech-to-text file exists: {os.path.exists('speech_to_text.py')}") 