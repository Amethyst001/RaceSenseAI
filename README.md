# RaceSenseAI - Toyota GR Cup Race Analytics

An advanced machine learning platform for comprehensive race performance analysis, lap time prediction, and driver insights for the Toyota GR Cup Indianapolis Race 1.

## Overview

RaceSense AI is a comprehensive racing intelligence platform providing predictions, sector insights, telemetry analysis, and improvement guidance for GR Cup drivers.

## Live Demo

🚀 **[View Live Dashboard](https://racesense-ai.streamlit.app)**

## Key Features

### Machine Learning & Predictions
- **Multi-Model Ensemble**: 6 ML algorithms (XGBoost, Random Forest, Gradient Boosting, Ridge, LightGBM, CatBoost)
- **High Accuracy**: 0.513s Mean Absolute Error (0.51% prediction error)
- **Per-Driver Optimization**: Individual model selection for each driver based on performance
- **Confidence Intervals**: 90% prediction ranges with statistical validation

### Performance Analytics
- **Actionable Insights**: 294 comprehensive recommendations with quantified time gains
- **Sector Analysis**: 3-sector lap breakdown with consistency scoring and comparative analysis
- **Telemetry Processing**: Real-time analysis of speed, throttle, brake, and RPM data
- **Driver Profiling**: AI-powered driving style classification and comparative performance metrics

### Interactive Dashboard
- **5 Analysis Modules**: Driver Overview, Lap Time Prediction, Driving Insights, Sector Breakdown, Leaderboard
- **Real-Time Visualizations**: Professional-grade charts with Plotly integration
- **Full Vehicle Identification**: Complete chassis tracking (e.g., GR86-049-88)
- **Responsive Design**: Optimized for desktop and mobile viewing

## Installation

### Prerequisites
- Python 3.8 or higher
- 2GB RAM minimum
- 500MB available disk space

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Amethyst001/RaceSenseAI.git
cd RaceSenseAI

# Install required dependencies
pip install -r requirements.txt

# Download race data
# Visit: https://trddev.com/hackathon-2025/
# Download: indianapolis.zip
# Extract to project root (creates indianapolis/ folder)
```

## Usage

### Quick Start (Windows)
```bash
run_full_analysis.bat
```

### Manual Execution
```bash
# Run complete analysis pipeline
python scripts\analyze_race.py

# Launch dashboard
cd dashboard
streamlit run dashboard.py
```

The dashboard will be accessible at `http://localhost:8502`

## System Architecture

## Project Structure

```
Toyota/
├── scripts/                     # Analysis scripts
│   └── analyze_race.py          # Main analysis pipeline
├── run_full_analysis.bat        # One-click launcher
├── requirements.txt             # Python dependencies
├── dashboard/                   # Streamlit dashboard
├── indianapolis/                # Race data (download separately)
├── outputs/                     # Generated results
└── models/                      # Trained ML models
```

## Data Source

Download race data from https://trddev.com/hackathon-2025/

File: `indianapolis.zip`

Extract to project root to create the following structure:
```
Toyota/
└── indianapolis/
    └── indianapolis/
        ├── R1_indianapolis_BestLap.csv
        ├── R1_indianapolis_Laps.csv
        ├── R1_indianapolis_Results.csv
        └── [other CSV files]
```

## Requirements

- Python 3.8+
- Windows (for batch files) or modify for Linux/Mac
- 2GB RAM minimum
- 500MB disk space

## Analysis Pipeline

The system executes a comprehensive analysis workflow:

| Module | Description |
|--------|-------------|
| Data Loading | Import and validate race data, calculate driver statistics |
| Telemetry Processing | Analyze speed, throttle, brake, RPM, and G-force data |
| Model Training | Train 6 ML algorithms per driver, select optimal model |
| Sector Analysis | 3-sector lap breakdown with consistency metrics |
| Insight Generation | Generate 294 actionable recommendations with time gains |
| Validation | Cross-reference predictions with official race results |
| AI Commentary | Intelligent driver profiling and comparative analysis |

## Dashboard Modules

### 1. Driver Overview
- Performance metrics and comparative analysis
- Quick stats with prediction confidence
- Field ranking and gap analysis

### 2. Lap Time Prediction
- ML-powered next lap predictions
- 90% confidence intervals
- Model performance metrics
- Prediction quality dashboard

### 3. Driving Insights
- 294 prioritized recommendations (HIGH/MEDIUM/LOW)
- GR86-specific technical guidance
- Priority distribution visualization
- Collapsible driver profiles

### 4. Sector Breakdown
- 3-sector performance analysis
- Color-coded sector comparison (Amber/Cyan/Green)
- Strongest/weakest sector identification
- Improvement potential quantification

### 5. Leaderboard
- Comprehensive driver rankings
- Best lap times and consistency scores
- Field-wide performance comparison

## Performance Metrics

- **Prediction Accuracy**: 0.513s MAE (0.51% error rate)
- **Data Coverage**: 19 of 29 drivers analyzed (minimum 20 laps required)
- **Insights Generated**: 294 actionable recommendations
- **Validation Score**: Grade A- (90/100)
- **Model Agreement**: 6-model ensemble with per-driver optimization

## Troubleshooting

**Dashboard won't start:**
```bash
pip install streamlit --upgrade
streamlit run dashboard/dashboard.py
```

**Missing data:**
Ensure `indianapolis/` folder exists with CSV files from https://trddev.com/hackathon-2025/

**Port conflict:**
Edit dashboard.py or use: `streamlit run dashboard.py --server.port 8503`

## Documentation

- `dashboard/README.md` - Dashboard documentation
- `VALIDATION_REPORT.md` - Logic validation results

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary development language
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Professional data visualization

### Machine Learning
- **Scikit-learn**: Base ML algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **LightGBM**: High-performance gradient boosting
- **CatBoost**: Categorical feature optimization
- **Pandas & NumPy**: Data manipulation and analysis

### Data Processing
- **8 CSV Files**: Race 1 telemetry, lap times, sector data, weather, official results
- **544 Telemetry Records**: Comprehensive vehicle data
- **27 Drivers**: Sector timing data coverage

## Contributing

This project was developed for the Toyota GR Cup Race Analytics challenge. Contributions, issues, and feature requests are welcome.

## Acknowledgments

- **Data Source**: TRD (Toyota Racing Development) - https://trddev.com/hackathon-2025/
- **Race Event**: Toyota GR Cup Indianapolis Race 1
- **Vehicle**: GR86 Cup Car (spec racing series)

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact & Support

For questions, issues, or collaboration inquiries:
- **GitHub**: [@Amethyst001](https://github.com/Amethyst001)
- **Repository**: [RaceSenseAI](https://github.com/Amethyst001/RaceSenseAI)

## Version

**v1.0.0** - Production Release (November 2025)

---

**Built with ❤️ for motorsport analytics and data-driven racing performance optimization.**
