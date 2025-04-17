# GoodRobot API Documentation

## Related Documentation
- [README.md](../README.md) - Project overview and setup instructions
- [Sequence Optimization Documentation](sequence_optimization.md) - Detailed explanation of sequence optimization features
- [Developer Guide](../docs/DEVELOPER.md) - General development guidelines

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Required dependencies installed (see `requirements.txt`)

### Starting the API Server

1. Activate your virtual environment:
```bash
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
```bash
python run_api.py
```

The server will start on `http://localhost:8000` with hot reload enabled for development.

### API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative documentation: `http://localhost:8000/redoc`

## API Endpoints

### Sequence Optimization

#### POST /api/sequence/optimize
Optimizes a sequence of tasks based on various parameters.

**Request Body:**
```json
{
    "tasks": [
        {
            "id": "task1",
            "priority": 0.8,
            "dependencies": [],
            "resources": ["resource1"],
            "estimated_duration": 3600
        }
    ],
    "constraints": {
        "max_duration": 86400,
        "resource_limits": {
            "resource1": 1
        }
    }
}
```

**Response:**
```json
{
    "optimized_sequence": [...],
    "metrics": {
        "efficiency": 0.95,
        "resource_utilization": 0.85
    }
}
```

### Pattern Analysis

#### POST /api/pattern/analyze
Analyzes patterns in a sequence of tasks.

**Request Body:**
```json
{
    "sequence": [...],
    "analysis_type": "comprehensive"
}
```

**Response:**
```json
{
    "pattern_metrics": {
        "complexity": 0.75,
        "stability": 0.85,
        "innovation": 0.65
    },
    "recommendations": [...]
}
```

## Development

### Environment Variables

Create a `.env` file in the root directory with the following variables:
```
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

### Testing

Run the test suite:
```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines. Use the provided pre-commit hooks:
```bash
pre-commit install
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running from the project root directory
   - Verify your virtual environment is activated
   - Check that all dependencies are installed

2. **Port Conflicts**
   - If port 8000 is in use, modify the port in `run_api.py`
   - Check for other running instances of the API

3. **Hot Reload Not Working**
   - Ensure you're using the development server (`run_api.py`)
   - Check file permissions
   - Verify the file is being saved correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

For more detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
