from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from app.be_core.config import settings
from app.be_core.logger import logger
from app.utils.vanna_class import VannaCustom


def get_server_connection_string():
    return f"postgresql+{settings.ONGC_VECTOR_DRIVER}://{settings.ONGC_VECTOR_USERNAME}:{settings.ONGC_VECTOR_PASSWORD}@{settings.ONGC_VECTOR_SERVER}:{settings.ONGC_VECTOR_PORT}/postgres"


def build_vector_db_connection_string(db_name: str):
    return f"postgresql+{settings.ONGC_VECTOR_DRIVER}://{settings.ONGC_VECTOR_USERNAME}:{settings.ONGC_VECTOR_PASSWORD}@{settings.ONGC_VECTOR_SERVER}:{settings.ONGC_VECTOR_PORT}/{db_name}"


def ensure_database_exists(db_name: str):
    engine = create_engine(get_server_connection_string(), isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname=:db"), {"db": db_name}
        )
        exists = result.scalar()
        if not exists:
            logger.info(f"Database '{db_name}' does not exist. Creating it...")
            try:
                conn.execute(
                    text(f'CREATE DATABASE "{db_name}"')
                )  # double quotes for safety
                logger.info(f"Database '{db_name}' created.")
            except ProgrammingError as e:
                logger.error(f"Failed to create database '{db_name}': {e}")
        else:
            logger.info(f"Database '{db_name}' already exists.")


class VannaConnectorManager:
    _instances = {}

    def __new__(cls, source_db_name: str = None, llm_model: str = "gpt-4o"):
        key = f"{source_db_name}-{llm_model}"
        if source_db_name is None:
            raise ValueError("source_db_name must be provided")

        if key not in cls._instances:
            logger.info(f"[VannaConnectorManager] Creating new instance for key: {key}")
            cls._instances[key] = super(VannaConnectorManager, cls).__new__(cls)
        return cls._instances[key]

    def __init__(self, source_db_name: str = None, llm_model: str = "gpt-4o"):
        if source_db_name is None:
            raise ValueError("source_db_name must be provided")

        # key = f"{source_db_name}-{llm_model}"
        if getattr(self, "initialized", False):
            logger.info(
                f"[VannaConnectorManager] Instance already initialized for DB: {source_db_name}"
            )
            return

        logger.info(
            f"[VannaConnectorManager] Initializing VannaConnectorManager for DB: {source_db_name} with model: {llm_model}"
        )
        logger.info(
            f"[VannaConnectorManager] Checking if vector DB exists for: {source_db_name}"
        )

        ensure_database_exists(source_db_name)
        connection_string = build_vector_db_connection_string(source_db_name)

        logger.info(
            f"[VannaConnectorManager] Initializing VannaCustom with model: {llm_model}"
        )

        self.vn = VannaCustom(
            llm_model=llm_model,
            config={
                "model": llm_model,
                "connection_string": connection_string,
            },
        )
        self.initialized = True
        logger.info(
            f"[VannaConnectorManager] Initialization complete for DB: {source_db_name}"
        )
