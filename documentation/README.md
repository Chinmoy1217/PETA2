# PETA2 - Predictive ETA Analysis System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Snowflake](https://img.shields.io/badge/Snowflake-Data_Warehouse-29B5E8.svg)](https://www.snowflake.com/)

## 📋 Overview

**PETA2** (Predictive ETA Analysis 2) is an advanced shipment tracking and ETA prediction system designed for maritime and multimodal logistics. The system leverages machine learning, real-time data integration, and comprehensive risk analysis to provide accurate delivery predictions and actionable insights for supply chain stakeholders.

### Key Features

- 🤖 **ML-Powered ETA Prediction**: XGBoost-based model for accurate delivery time estimation
- 📊 **Real-time Risk Assessment**: Multi-factor risk scoring (weather, geopolitics, strikes, congestion)
- 🔄 **Automated ETL Pipeline**: Azure Blob Storage → Snowflake → Data Transformation
- 📈 **Carrier Reliability Scoring**: Historical performance-based carrier rankings
- 🎯 **Data Quality Gates**: Automated validation with 80% accuracy threshold
- 🌐 **RESTful API**: FastAPI-powered endpoints for integration
- 💾 **Data Warehouse**: Snowflake-based normalized schema

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Azure Blob     │────▶│  Snowflake   │────▶│   ETL Pipeline  │
│  Storage (CSV)  │     │  (Staging)   │     │  (Transform)    │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                       │
                                                       ▼
                        ┌──────────────────────────────────────┐
                        │    Snowflake Data Warehouse          │
                        │  ┌────────────┐  ┌────────────────┐  │
                        │  │ FACT_TRIP  │  │ FACT_LANE_ETA  │  │
                        │  │ DIM_PORT   │  │ DIM_CARRIER    │  │
                        │  │ DIM_VEHICLE│  │                │  │
                        │  └────────────┘  └────────────────┘  │
                        └──────────────────────────────────────┘
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
              ┌───────────────────┐           ┌──────────────────┐
              │  ML Models        │           │  FastAPI Server  │
              │  • XGBoost ETA    │           │  • /predict      │
              │  • Risk Classifier│           │  • /upload       │
              │  • Reliability    │           │  • /ingest       │
              └───────────────────┘           └──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Snowflake Account**: With access to HACAKATHON database
- **Azure Blob Storage**: For file staging (optional)
- **Git**: For version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chinmoy1217/PETA2.git
   cd PETA2
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Backend dependencies
   cd backend
   pip install -r requirements.txt
   
   # Frontend dependencies (optional)
   cd ../frontend
   npm install
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   SNOWFLAKE_PASSWORD=your_jwt_token_here
   AZURE_STORAGE_SAS_TOKEN=your_sas_token_here
   ```
   
   Update `backend/config.py` with your Snowflake credentials:
   ```python
   SNOWFLAKE_CONFIG = {
       'account': 'YOUR_ACCOUNT',
       'user': 'YOUR_USER',
       'password': 'YOUR_PASSWORD_OR_JWT',
       'database': 'HACAKATHON',
       'schema': 'DT_INGESTION',
       'warehouse': 'YOUR_WAREHOUSE'
   }
   ```

5. **Initialize the database**
   ```bash
   # Run SQL scripts to create tables
   # Execute the SQL files in backend/*.sql using Snowflake UI or CLI
   ```

---

## 📦 Project Structure

```
PETA2/
├── backend/                      # Backend Python application
│   ├── main.py                   # FastAPI application entry point
│   ├── config.py                 # Configuration (Snowflake credentials)
│   ├── data_loader.py            # Data fetching from Snowflake
│   ├── data_transformation.py    # ETL transformation logic
│   ├── ingestion_service.py      # Azure → Snowflake pipeline
│   ├── quality_check.py          # Data validation module
│   ├── model_engine.py           # XGBoost ETA prediction model
│   ├── risk_model.py            # Risk classification model
│   ├── reliability_model.py      # Carrier scoring model
│   ├── prediction_pipeline.py    # Batch prediction orchestrator
│   └── requirements.txt          # Python dependencies
├── frontend/                     # React.js frontend (optional)
│   ├── src/
│   └── package.json
├── model/                        # Trained ML models and artifacts
│   ├── eta_xgboost.json         # Trained XGBoost model
│   ├── label_encoders.joblib    # Feature encoders
│   └── route_stats.json         # Historical route statistics
├── documentation/                # Technical documentation
│   └── PETA2_Technical_Documentation.md
├── .env                         # Environment variables
├── .gitignore
├── docker-compose.yml           # Docker orchestration
├── Dockerfile                   # Container definition
└── README.md                    # This file
```

---

## 🎯 Running the Application

### Backend (FastAPI Server)

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation**: `http://localhost:8000/docs` (Swagger UI)

### Frontend (React App)

```bash
cd frontend
npm start
```

The frontend will be available at `http://localhost:3000`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 🔑 API Endpoints

### Authentication
- **POST** `/login` - User authentication
- **POST** `/signup` - User registration

### Prediction
- **POST** `/simulate-trip` - Get ETA prediction for a shipment
  ```json
  {
    "PolCode": "CNSHA",
    "PodCode": "USLAX",
    "ModeOfTransport": "OCEAN",
    "Carrier": "MSCU"
  }
  ```

### Data Management
- **POST** `/upload-to-cloud` - Upload CSV file for processing
- **POST** `/ingest` - Trigger data ingestion pipeline
- **GET** `/active-shipments` - Retrieve current shipments

### Monitoring
- **GET** `/metrics` - Get model performance metrics
- **GET** `/` - Health check

---

## 🧪 ML Models

### 1. ETA Prediction Model
- **Algorithm**: XGBoost Gradient Boosting
- **Features**: Port of Loading, Port of Discharge, Mode of Transport, Risk Scores
- **Target**: Transit time in days
- **Accuracy**: RMSE ~0.52 days

### 2. Risk Classification Model
- **Algorithm**: Random Forest Classifier
- **Purpose**: Predict shipment delay probability
- **Features**: Weather, Congestion, Strikes, Geopolitical factors
- **Output**: Risk probability (0.0 - 1.0)

### 3. Carrier Reliability Model
- **Type**: Statistical scoring
- **Metrics**: On-time percentage, Average delay
- **Output**: Reliability score (0-100)

---

## 📊 Data Pipeline

### Ingestion Flow

1. **File Upload**: CSV files uploaded to Azure Blob Storage (`input` container)
2. **Quality Check**: Validation against 80% accuracy threshold
3. **Stage Loading**: `COPY INTO` Snowflake staging table (`DT_PREP.PETA`)
4. **Transformation**: SQL-based transformation into fact/dimension tables
5. **Model Training**: Incremental learning from new data
6. **Archival**: Processed files moved to `archive` container

### Database Schema

- **DT_PREP**: Staging layer
  - `PETA` - Raw ingestion table

- **DT_INGESTION**: Normalized data warehouse
  - `FACT_TRIP` - Shipment records
  - `FACT_LANE_ETA_FEATURES` - Risk and timing metrics
  - `DIM_CARRIER` - Carrier master data
  - `DIM_VEHICLE` - Vessel information
  - `DIM_PORT` - Port master data

---

## 🛠️ Development

### Training the ML Model

```bash
cd backend
python -c "from model_engine import train_model_task; train_model_task()"
```

Or use the Jupyter notebook:
```bash
jupyter notebook ETA_Training_Walkthrough.ipynb
```

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest

# API validation
python test_api_validation.py
```

### Code Quality

```bash
# Format code
black backend/

# Lint
flake8 backend/

# Type checking
mypy backend/
```

---

## 📝 Configuration

### Snowflake Connection

Edit `backend/config.py`:
```python
SNOWFLAKE_CONFIG = {
    'account': 'YOUR-ACCOUNT',
    'user': 'YOUR_USER',
    'password': 'YOUR_PASSWORD',  # Or JWT token
    'database': 'HACAKATHON',
    'schema': 'DT_INGESTION',
    'warehouse': 'YOUR_WAREHOUSE'
}
```

### Azure Blob Storage

Required environment variables:
```env
AZURE_STORAGE_ACCOUNT_NAME=your_account_name
AZURE_STORAGE_SAS_TOKEN=your_sas_token
AZURE_STORAGE_CONTAINER_INPUT=input
AZURE_STORAGE_CONTAINER_ARCHIVE=archive
```

---

## 🔒 Security

- **JWT Authentication**: All endpoints protected (except `/login`, `/signup`)
- **CORS Protection**: Configurable allowed origins
- **Snowflake JWT**: Password-less authentication support
- **SAS Tokens**: Time-limited Azure Blob access

---

## 📚 Documentation

For detailed technical documentation, see:
- **[Technical Documentation](documentation/PETA2_Technical_Documentation.md)** - Comprehensive system architecture and ML model details
- **[API Template](API_TEMPLATE.md)** - API specification and examples
- **[User Guide](PETA_USER_GUIDE.md)** - End-user instructions
- **[Start Guide](START_GUIDE.md)** - Quick start guide

---

## 🐛 Troubleshooting

### Common Issues

1. **Snowflake Connection Error**
   - Verify credentials in `config.py`
   - Check network access to Snowflake account
   - Ensure JWT token is valid (not expired)

2. **Model Not Found**
   - Run training pipeline first: `python -c "from model_engine import train_model_task; train_model_task()"`
   - Check if `model/eta_xgboost.json` exists

3. **Azure Blob Access Denied**
   - Verify SAS token in `.env` file
   - Check token expiration date
   - Ensure correct container names

4. **Data Quality Check Failed**
   - Review CSV file format
   - Ensure all critical fields are present
   - Check field formats (port codes, SCAC codes, IMO numbers)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is proprietary software developed by Cozentus.

---

## 👥 Team

- **Project Lead**: Cozentus Data Practice
- **Development Team**: Hackathon DT Team
- **Contact**: support@cozentus.com

---

## 🙏 Acknowledgments

- Snowflake for data warehouse infrastructure
- XGBoost team for the ML framework
- FastAPI for the excellent Python web framework
- Azure for cloud storage services

---

## 📈 Roadmap

- [ ] Real-time streaming ingestion (Kafka)
- [ ] Multi-model ensemble (LSTM + XGBoost)
- [ ] AutoML hyperparameter tuning
- [ ] GraphQL API support
- [ ] Kubernetes deployment
- [ ] Mobile app integration
- [ ] Advanced geospatial tracking

---

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/Chinmoy1217/PETA2/issues)
- **Email**: support@cozentus.com
- **Documentation**: See `/documentation` folder

---

**Made with ❤️ by Cozentus Data Practice Team**
