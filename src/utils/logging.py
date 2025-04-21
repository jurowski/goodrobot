import logging
import logging.handlers
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import asyncio

class StructuredLogger:
    def __init__(self, name: str, log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}_error.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", '
            '"function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set formatters
        file_handler.setFormatter(detailed_formatter)
        error_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
        
        # Store handlers for potential WebSocket integration
        self.handlers = {
            'file': file_handler,
            'error': error_handler,
            'console': console_handler
        }
    
    def add_websocket_handler(self, websocket, client_id: str):
        """Add a WebSocket handler for real-time log streaming."""
        ws_handler = WebSocketLogHandler(websocket, client_id)
        ws_handler.setLevel(logging.INFO)
        ws_handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
        ))
        self.logger.addHandler(ws_handler)
        return ws_handler
    
    def remove_websocket_handler(self, handler):
        """Remove a WebSocket handler."""
        self.logger.removeHandler(handler)

class WebSocketLogHandler(logging.Handler):
    def __init__(self, websocket, client_id: str):
        super().__init__()
        self.websocket = websocket
        self.client_id = client_id
    
    def emit(self, record):
        try:
            log_entry = {
                "type": "log",
                "client_id": self.client_id,
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage()
            }
            
            # Using _sync to avoid issues with async
            self.websocket._sync({"type": "websocket.send", "text": json.dumps(log_entry)})
        except Exception:
            self.handleError(record)

class WebSocketHandler(logging.Handler):
    def __init__(self, websocket):
        super().__init__()
        self.websocket = websocket
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'message': self.format(record),
                'source': record.name,
                'type': 'log'
            }
            
            # Use the correct WebSocket send method
            if hasattr(self.websocket, 'send_text'):
                asyncio.create_task(self.websocket.send_text(json.dumps(log_entry)))
            elif hasattr(self.websocket, 'send'):
                asyncio.create_task(self.websocket.send(json.dumps(log_entry)))
            else:
                print(f"Warning: WebSocket has no send method. Log entry: {log_entry}")
                
        except Exception as e:
            print(f"Error sending log to WebSocket: {e}")
            self.handleError(record)

class Logger:
    def __init__(self, name, websocket=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logging
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('app.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # WebSocket handler if provided
        if websocket:
            ws_handler = WebSocketHandler(websocket)
            ws_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(ws_handler)
            
    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)
        
    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
        
    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
        
    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
        
    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)

# Create logger instances for different components
api_logger = StructuredLogger("api")
websocket_logger = StructuredLogger("websocket")
audio_logger = StructuredLogger("audio") 