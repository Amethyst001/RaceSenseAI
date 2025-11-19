@echo off
color 0A
echo ================================================================================
echo TOYOTA GR CUP RACE ANALYSIS - COMPLETE EXECUTION
echo ================================================================================
echo.
echo This script will:
echo  [1/2] Execute complete race analysis (3-5 minutes)
echo  [2/2] Launch interactive dashboard
echo.
echo Analysis includes:
echo  - ML-powered lap time predictions (0.513s accuracy)
echo  - 95 actionable performance recommendations
echo  - Sector analysis and telemetry insights
echo  - AI-generated commentary
echo.
pause

echo.
echo ================================================================================
echo [1/2] Executing Race Analysis Pipeline...
echo ================================================================================
py scripts\analyze_race.py
if errorlevel 1 (
    echo.
    echo ERROR: Analysis execution failed
    echo Please verify data files exist in indianapolis/indianapolis/
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [2/2] Launching Interactive Dashboard...
echo ================================================================================
echo.
echo Dashboard URL: http://localhost:8502
echo.
echo Dashboard Features:
echo  - Driver Performance: Key metrics and comparisons
echo  - Predictions and Analysis: ML predictions with confidence intervals
echo  - Telemetry Insights: 95 specific recommendations
echo  - Sector Analysis: Three-sector lap breakdown
echo  - Leaderboard: Complete driver rankings
echo.
echo Press Ctrl+C to stop the dashboard server
echo ================================================================================
echo.

py -m streamlit run dashboard/dashboard.py --server.headless true --server.port 8502
