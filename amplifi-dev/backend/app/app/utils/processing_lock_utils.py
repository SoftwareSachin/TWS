"""
Processing lock utility functions extracted from document_processing_utils.py
Provides lock management functionality without docling dependencies.
Used by video ingestion and other processing tasks.
"""

import time

from celery.result import AsyncResult

from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger

# Lua script for atomic lock takeover if original task is dead
LUA_TAKEOVER_LOCK = """
local key = KEYS[1]
local new_task_id = ARGV[1]
local expiry = ARGV[2]
local current_value = redis.call('GET', key)
if current_value then
    redis.call('SETEX', key, expiry, new_task_id)
    return current_value
else
    return nil
end
"""


def _check_task_is_alive(existing_task_id: str, current_task_id: str) -> bool:
    """
    Robust task liveness detection with multiple fallback mechanisms.
    Handles graceful shutdowns, forceful kills, and worker crashes.
    """
    from celery.states import FAILURE, PENDING, RETRY, REVOKED, STARTED, SUCCESS

    DEAD_STATES = [SUCCESS, FAILURE, REVOKED]

    try:
        if not existing_task_id:
            return False

        result = AsyncResult(existing_task_id, app=celery)
        state = result.state

        logger.debug(
            f"Checking task {existing_task_id} state: {state}, type: {type(state)}"
        )

        # 1. Final states - definitely dead
        if state in DEAD_STATES:
            logger.debug(f"Task {existing_task_id} in final state: {state}")
            return False

        # 2. RETRY state - graceful shutdown occurred
        if state == RETRY:
            logger.info(f"Task {existing_task_id} was gracefully requeued")
            return False

        # 3. STARTED - check if still active on workers
        if state == STARTED:
            return _verify_task_on_active_workers_with_retry(
                existing_task_id=existing_task_id,
                current_task_id=current_task_id,
                max_retries=2,
            )

        # 4. PENDING - most problematic state, needs careful checking
        if state == PENDING:
            # First check: is it actually running on any worker?
            if _verify_task_on_active_workers_with_retry(
                existing_task_id=existing_task_id,
                current_task_id=current_task_id,
                max_retries=2,
            ):
                return True

            # Second check: has it been pending too long? (likely stuck)
            logger.warning(
                f"Task {existing_task_id} is PENDING but not found on any worker - considering dead"
            )
            return False

        # 5. Any other state - be conservative
        logger.info(f"Task {existing_task_id} in state {state} - considering alive")
        return True

    except Exception as e:
        logger.warning(
            f"Error checking task {existing_task_id}: {e} - assuming dead for safety"
        )
        return False


def _verify_task_on_active_workers_with_retry(
    existing_task_id: str, current_task_id: str = None, max_retries: int = 2
) -> bool:
    """Verify task with retry logic for network issues"""
    for attempt in range(max_retries + 1):
        try:
            result = _verify_task_on_active_workers(existing_task_id, current_task_id)
            return result
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Worker verification attempt {attempt + 1} failed: {e}, retrying in {wait_time}s"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All worker verification attempts failed: {e}")
                return False  # Assume dead for safety


def _verify_task_on_active_workers(
    existing_task_id: str, current_task_id: str = None
) -> bool:
    """
    Verify if a task is actually running on any alive worker.
    Enhanced with ping verification to detect zombie/stale workers.

    Args:
        existing_task_id: The task ID to check
        current_task_id: Current task ID to exclude from check
    """
    from celery import current_app

    try:
        inspect = current_app.control.inspect()

        if not inspect:
            logger.warning("Celery inspect failed - assuming task is dead")
            return False

        # First: Ping all workers to see who's actually alive
        try:
            ping_results = inspect.ping()
        except Exception as e:
            logger.warning(f"Worker ping failed: {e} - assuming task is dead")
            return False

        if not ping_results:
            logger.warning("No workers responded to ping - assuming task is dead")
            return False

        alive_workers = list(ping_results.keys())
        logger.debug(f"Found {len(alive_workers)} alive workers: {alive_workers}")

        # Second: Get active tasks from workers
        try:
            active_tasks = inspect.active()
        except Exception as e:
            logger.warning(
                f"Active task inspection failed: {e} - assuming task is dead"
            )
            return False

        if not active_tasks:
            logger.debug("No active tasks found on any worker")
            return False

        # Third: Check active tasks only from workers that responded to ping
        for worker_name, tasks in active_tasks.items():
            # Skip workers that didn't respond to ping (zombie workers)
            if worker_name not in alive_workers:
                logger.warning(
                    f"Worker {worker_name} reported active tasks but didn't respond to ping - skipping (zombie worker)"
                )
                continue

            # Check tasks on this alive worker
            for task_info in tasks or []:
                task_id = task_info.get("id")
                if task_id == existing_task_id:
                    # Don't consider the current task as blocking itself
                    if current_task_id and task_id == current_task_id:
                        continue

                    logger.info(
                        f"Task {existing_task_id} confirmed active on alive worker {worker_name}"
                    )
                    return True

        logger.debug(f"Task {existing_task_id} not found on any alive worker")
        return False

    except Exception as e:
        logger.error(f"Error during worker verification: {e} - assuming task is dead")
        return False


def _handle_stale_lock(
    redis_client, processing_key: str, existing_task_id: str, current_task_id: str
) -> bool:
    """
    Handle a potentially stale processing lock by checking if the original task is still alive.

    Args:
        redis_client: Redis client instance
        processing_key: The key being locked
        existing_task_id: Task ID that currently holds the lock
        current_task_id: Current task ID attempting to acquire the lock

    Returns:
        bool: True if lock was successfully taken over, False if original task is still alive
    """
    try:
        try:
            ttl = redis_client.ttl(processing_key)
            if ttl > 0:
                # Calculate how long the lock has been held
                original_ttl = 3600  # 1 hour
                lock_age_seconds = original_ttl - ttl

                # If lock is older than 10 minutes, force takeover regardless
                if lock_age_seconds > settings.MAX_LOCK_AGE:
                    logger.warning(
                        f"Lock {processing_key} held for {lock_age_seconds//60} minutes - forcing takeover"
                    )
                    original_task_id = redis_client.eval(
                        LUA_TAKEOVER_LOCK, 1, processing_key, current_task_id, 3600
                    )
                    return True
        except Exception as e:
            logger.debug(f"Could not check lock age: {e}")

        # Check if the existing task is still alive
        if _check_task_is_alive(existing_task_id, current_task_id):
            logger.info(
                f"Original task {existing_task_id} is still alive - respecting lock"
            )
            return False

        logger.warning(
            f"Original task {existing_task_id} appears to be dead - attempting takeover"
        )

        # Use Lua script for atomic takeover
        original_task_id = redis_client.eval(
            LUA_TAKEOVER_LOCK, 1, processing_key, current_task_id, 3600
        )

        if original_task_id:
            # Handle both bytes and string types safely
            if isinstance(original_task_id, bytes):
                original_task_id = original_task_id.decode("utf-8")
            else:
                original_task_id = str(original_task_id)
            logger.info(
                f"Successfully took over lock from dead task {original_task_id}"
            )
            return True
        else:
            logger.info(f"Lock {processing_key} disappeared during takeover attempt")
            return True  # Lock is gone, we can proceed

    except Exception as e:
        logger.error(f"Error handling stale lock {processing_key}: {str(e)}")
        return False


def _cleanup_processing_lock(redis_client, processing_key: str, task_id: str) -> bool:
    """
    Atomically clean up processing lock, but only if owned by current task.
    Uses Lua script to ensure atomicity.

    Returns:
        bool: True if lock was cleaned up, False if not owned by current task
    """
    lua_script = """
    local key = KEYS[1]
    local expected_value = ARGV[1]
    local current_value = redis.call('GET', key)
    if current_value == expected_value then
        redis.call('DEL', key)
        return 1
    else
        return 0
    end
    """

    try:
        result = redis_client.eval(lua_script, 1, processing_key, task_id)
        success = result == 1
        if success:
            logger.debug(f"Successfully released processing lock: {processing_key}")
        else:
            logger.debug(
                f"Lock {processing_key} not owned by current task - not cleaned"
            )
        return success
    except Exception as e:
        logger.warning(f"Error cleaning up processing lock: {e}")
        return False
