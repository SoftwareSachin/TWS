import sys
import os

# Add /app/dist to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import uvicorn
try:
    from app.main import app
    print("Successfully imported app.main")
except ImportError as e:
    print(f"ImportError: {e}")
    raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)