# House Price Prediction ML Project

A professional machine learning project for predicting house prices using industry-standard tools and workflows.

## 🏠 Project Overview

This project demonstrates a complete ML pipeline from data ingestion to model deployment, following professional development practices used in real data science teams.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Git installed
- 4GB+ free disk space

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd house-price-ml

# Build and start the development environment
docker-compose up --build

# Access Jupyter notebooks
# Open http://localhost:8888 in your browser

# Access MLflow tracking UI
# Open http://localhost:5000 in your browser
```

## 📁 Project Structure

```
├── data/           # Data storage (raw, processed, external)
├── notebooks/      # Jupyter notebooks for exploration
├── src/           # Production-ready source code
├── tests/         # Unit tests
├── models/        # Trained model artifacts
├── reports/       # Generated analysis and reports
└── deployment/    # Deployment configurations
```

## 🛠️ Development Workflow

1. **Data Exploration:** Use notebooks in `notebooks/`
2. **Feature Engineering:** Implement in `src/features/`
3. **Model Training:** Code in `src/models/`
4. **Testing:** Write tests in `tests/`
5. **Experiment Tracking:** Use MLflow UI at localhost:5000

## 📊 Model Performance

Current best model: [To be updated]

- Algorithm: [To be updated]
- RMSE: [To be updated]
- R²: [To be updated]

## 👥 Contributing

1. Create feature branch: `git checkout -b feature-name`
2. Make changes and add tests
3. Run tests: `docker-compose exec ml-dev pytest`
4. Create pull request

## 📚 Documentation

- [Data Dictionary](docs/data-dictionary.md)
- [Model Documentation](docs/models.md)
- [API Documentation](docs/api.md)

## 📚 Helpful Commands

- On load > git pull (update branch with changes from main)
- docker-compose up -d
- docker-compose exec ml-dev _insert command_ >> runs commands in the container
- Tests:
  - docker-compose exec ml-dev black/flake8/mypy src/ tests/
  - (Run all tests) docker-compose exec ml-dev pytest tests/ -v
- Closing out > git push > docker-compose down
