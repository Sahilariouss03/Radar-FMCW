"""
Main entry point for the backend server.
Run with: python main.py
"""

import uvicorn
from radar.api import app

if __name__ == "__main__":
    uvicorn.run(
        "radar.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
