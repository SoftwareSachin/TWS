import sys
import os

# Add /app/dist to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from app.be_core.celery import celery
    print("Successfully imported app.be_core.celery")
except ImportError as e:
    print(f"ImportError: {e}")
    raise

if __name__ == "__main__":
    celery.beat().run(
        scheduler="sqlalchemy_celery_beat.schedulers:DatabaseScheduler",
        loglevel="info",
    )