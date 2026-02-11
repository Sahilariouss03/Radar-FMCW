#!/bin/bash

echo "========================================"
echo " Automotive FMCW SAR Simulation System"
echo "========================================"
echo ""
echo "Starting Backend Server..."
echo ""

cd backend
python main.py &
BACKEND_PID=$!

sleep 3

echo ""
echo "Starting Frontend Application..."
echo ""

cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo " Both servers are running!"
echo "========================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
