@echo off
echo ========================================
echo  Automotive FMCW SAR Simulation System
echo ========================================
echo.
echo Starting Backend Server...
echo.

cd backend
start cmd /k "python main.py"

timeout /t 3 /nobreak > nul

echo.
echo Starting Frontend Application...
echo.

cd ..\frontend
start cmd /k "npm run dev"

echo.
echo ========================================
echo  Both servers are starting!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit this window...
pause > nul
