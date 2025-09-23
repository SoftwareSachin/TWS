from typing import Generator

from sqlalchemy_celery_beat.session import SessionManager

from app.be_core.config import settings


def get_job_db() -> Generator:
    session_manager = SessionManager()
    engine, _session = session_manager.create_session(
        str(settings.SYNC_CELERY_BEAT_DATABASE_URI)
    )

    with _session() as session:
        yield session
