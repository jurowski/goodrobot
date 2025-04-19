# Confidence Correlations Research Platform

## Overview
The Confidence Correlations platform is a research-focused feature that enables users to propose, test, and validate hypotheses about factors affecting human potential and well-being. The platform follows scientific methodology principles, including control groups, experimental design, and peer review.

## Features

### 1. Research Hypothesis Management
- Create and propose new research hypotheses
- Define independent and dependent variables
- Specify control and experimental groups
- Set participant criteria and sample sizes
- Track experiment status and progress

### 2. Participant Management
- Join ongoing experiments
- Submit measurements and data
- Track participation progress
- Complete experiment participation

### 3. Results and Analysis
- Submit research results with statistical analysis
- Peer review system for validating findings
- Publication of verified results
- Integration of findings into the platform

## Current Implementation

### Data Models
- `ResearchHypothesis`: Stores hypothesis details and experimental design
- `ParticipantData`: Tracks participant measurements and progress
- `ResearchResult`: Contains experiment outcomes and analysis

### Sample Studies
1. **Meditation and Focus Study**
   - Duration: 30 days
   - Metrics: Focus duration, task completion, meditation minutes, stress levels
   - Sample size: 10 participants (5 control, 5 experimental)

2. **Sleep Quality Study**
   - Duration: 45 days
   - Metrics: Sleep quality, creativity scores, problem-solving time, REM sleep
   - Sample size: 8 participants (4 control, 4 experimental)

3. **Exercise and Stress Study**
   - Duration: 60 days
   - Metrics: Exercise intensity, stress resilience, recovery rate, cortisol levels
   - Sample size: 6 participants (3 control, 3 experimental)

### Data Generation
- Realistic data generation with progressive improvements
- Control vs. experimental group differentiation
- Random variations to simulate real-world conditions
- Daily metrics with timestamps

## Technical Implementation

### API Endpoints
- `POST /research/hypotheses`: Create new hypothesis
- `GET /research/hypotheses`: List all hypotheses
- `GET /research/hypotheses/{id}`: Get specific hypothesis
- `POST /research/hypotheses/{id}/participate`: Join an experiment
- `POST /research/data/{id}`: Submit measurement data
- `POST /research/results/{id}`: Submit experiment results
- `POST /research/results/{id}/review`: Submit peer review

### Database
- MongoDB integration using Motor for async operations
- Efficient querying and filtering capabilities
- Secure data storage and access control

## Getting Started

### Prerequisites
- MongoDB installed and running
- Python 3.7+
- FastAPI

### Setup
1. Install required packages:
   ```bash
   pip install motor pymongo fastapi
   ```

2. Configure MongoDB connection:
   ```python
   MONGODB_URL = "mongodb://localhost:27017"
   ```

3. Start the application:
   ```bash
   uvicorn src.api.api:app --reload
   ```

### Usage Example
1. Create a new hypothesis:
   ```python
   hypothesis = {
       "title": "Meditation Impact on Focus",
       "description": "Testing the effect of daily meditation on attention span",
       "independent_variable": "Daily meditation duration",
       "dependent_variable": "Attention span score",
       "duration_days": 30,
       "required_participants": 100
   }
   ```

2. Submit participant data:
   ```python
   measurement = {
       "attention_score": 85,
       "meditation_minutes": 20,
       "date": "2024-03-20"
   }
   ```

## Best Practices
1. **Hypothesis Design**
   - Clearly define variables
   - Use measurable outcomes
   - Consider potential confounding factors

2. **Data Collection**
   - Standardize measurement methods
   - Ensure consistent timing
   - Document any deviations

3. **Analysis**
   - Use appropriate statistical methods
   - Account for sample size
   - Document limitations

## Security
- Authentication required for sensitive operations
- Data validation at all levels
- Secure storage of participant data

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
MIT License 