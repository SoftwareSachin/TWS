import sys
import os

# Add /app/dist to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Use the minimal video-only celery configuration
    from app.be_core.celery_video import celery
    print("Successfully imported app.be_core.celery_video")
    
    # Check if video models are already available before initializing
    from app.utils.startup_video_models import initialize_video_models_on_startup
    from app.utils.video_model_manager import is_video_models_ready, get_video_models_status
    from app.be_core.logger import logger
    
    print("Checking video models availability at worker startup...")
    logger.info("Video worker starting - checking video models availability...")
    
    # First check if models are already ready
    if is_video_models_ready():
        print("✅ Video models are already available and ready")
        logger.info("Video models are already available and ready - skipping initialization")
        
        # Log current status for monitoring
        status = get_video_models_status()
        logger.info(f"Current video models status: {status}")
    else:
        print("Video models not ready - initializing now...")
        logger.info("Video models not ready - proceeding with initialization...")
        
        success = initialize_video_models_on_startup()
        if success:
            print("✅ Video models initialized successfully at startup")
            logger.info("Video models initialized successfully at worker startup")
        else:
            print("⚠️ Video model initialization failed at startup - will initialize on first use")
            logger.warning("Video model initialization failed at worker startup - will initialize on first use")
        
except ImportError as e:
    print(f"ImportError: {e}")
    raise

if __name__ == "__main__":
    # Video worker only handles video_ingestion_queue
    queue_list = "video_ingestion_queue"

    # Get pool type from environment variable, default to 'threads'
    pool = os.getenv("CELERY_POOL", "threads")

    # Get concurrency from environment variable, default to None (Celery decides)
    concurrency = os.getenv("CELERY_CONCURRENCY")
    if concurrency:
        concurrency = int(concurrency)
    else:
        concurrency = None

    # Build the worker arguments (similar to run_celery.py)
    args = ["worker", "--loglevel=info", "-Q", queue_list]
    if pool:
        args.extend(["--pool", pool])
    if concurrency:
        args.extend(["--concurrency", str(concurrency)])

    print(f"Starting video Celery worker with args: {args}")
    
    # Start the Celery worker (same pattern as run_celery.py)
    celery.worker_main(args)
