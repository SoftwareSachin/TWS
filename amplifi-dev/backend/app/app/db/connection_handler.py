from enum import Enum  # adjust based on actual import

from app.be_core.logger import logger


# Database type enumeration
class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class DatabaseConnectionHandler:
    """Handles database-specific operations for both PostgreSQL and MySQL"""

    @staticmethod
    def connect_database(vn, db_details: dict, db_type: DatabaseType):
        logger.info(f"🔌 Connecting to database of type: {db_type}")
        logger.info(f"Details of the databses {db_details}")
        logger.debug(
            f"📍  DB Host: {db_details['host']}, DB Name: {db_details['database_name']}, User: {db_details['username']}, Port: {db_details.get('port', 'default')}"
        )
        try:
            if db_type == DatabaseType.POSTGRESQL:
                vn.connect_to_postgres(
                    host=db_details["host"],
                    dbname=db_details["database_name"],
                    user=db_details["username"],
                    password=db_details["password"],
                    port=db_details["port"],
                )
                logger.info("✅ Successfully connected to PostgreSQL.")
            elif db_type == DatabaseType.MYSQL:
                logger.debug(f"in the ssl_mode {db_details.get("ssl_mode")} ")
                ssl_options = {}
                if db_details.get("ssl_mode") == "required":
                    ssl_options = {"ssl": {"ssl": True}}
                vn.connect_to_mysql(
                    host=db_details["host"],
                    dbname=db_details["database_name"],
                    user=db_details["username"],
                    password=db_details["password"],
                    port=db_details.get("port", 3306),
                    **ssl_options,
                )
                logger.info("✅ Successfully connected to MySQL.")
            else:
                logger.error(f"❌ Unsupported database type provided: {db_type}")
                raise ValueError(f"Unsupported database type: {db_type}")
        except Exception as e:
            logger.exception(f"❗ Failed to connect to {db_type} database: {str(e)}")
            raise

    @staticmethod
    def get_information_schema_query(db_type: DatabaseType) -> str:
        if db_type == DatabaseType.POSTGRESQL:
            return (
                "SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = 'public'"
            )
        elif db_type == DatabaseType.MYSQL:
            return "SELECT * FROM INFORMATION_SCHEMA.COLUMNS"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    @staticmethod
    def get_database_type_from_details(db_details: dict) -> DatabaseType:
        db_type_str = db_details.get("db_type", "").lower()
        if db_type_str in ("postgresql", "postgres"):
            return DatabaseType.POSTGRESQL
        elif db_type_str == "mysql":
            return DatabaseType.MYSQL
        else:
            port = db_details.get("port", 5432)
            return DatabaseType.MYSQL if port == 3306 else DatabaseType.POSTGRESQL
