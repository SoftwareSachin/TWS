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
    # Get queues from environment variable or default to all queues in task_routes
    queues = os.getenv("CELERY_QUEUES")
    if queues:
        queue_list = queues
    else:
        # Fallback to all queues defined in task_routes
        queues = set()
        for route in celery.conf.task_routes.values():
            if isinstance(route, dict) and "queue" in route:
                queues.add(route["queue"])
        queue_list = ",".join(queues)

    # Get pool type from environment variable, default to 'prefork'
    pool = os.getenv("CELERY_POOL", "prefork")

    # Get concurrency from environment variable, default to None (Celery decides)
    concurrency = os.getenv("CELERY_CONCURRENCY")
    if concurrency:
        concurrency = int(concurrency)
    else:
        concurrency = None

    # Build the worker arguments
    args = ["worker", "--loglevel=info", "-Q", queue_list]
    if pool:
        args.extend(["--pool", pool])
    if concurrency:
        args.extend(["--concurrency", str(concurrency)])

    # Start the Celery worker
    celery.worker_main(args)