from abc import ABC, abstractmethod

import psycopg2
import requests
from psycopg2 import sql
from tenacity import retry, stop_after_attempt, wait_fixed

from app.be_core.logger import logger


class DestinationWriterException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"DestinationWriterException: {self.message}"


class DestinationWriter(ABC):
    @abstractmethod
    def write(self, vectors):
        raise DestinationWriterException

    @abstractmethod
    def connect(self):
        raise DestinationWriterException

    @abstractmethod
    def backup(self):
        raise DestinationWriterException

    @abstractmethod
    def restore(self, backup_data):
        raise DestinationWriterException

    @abstractmethod
    def delete_backup(self):
        raise DestinationWriterException


class PGVectorWriter(DestinationWriter):
    """
    PostgreSQL vector database writer that handles storing, backing up, and restoring
    vector embeddings using pgvector extension.
    """

    def __init__(self, connection_details, dataset_name):
        """
        Initialize the PostgreSQL vector writer.

        Args:
            connection_details: Dictionary containing PostgreSQL connection parameters
            dataset_name: Name of the dataset being processed
        """
        self.connection_details = connection_details
        self.dataset_name = dataset_name
        self.connection = None

    def __enter__(self):
        """
        Context manager entry point - establishes database connection and cursor.
        Returns the cursor for query execution.
        """
        if not self.connection:
            self.connect()
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - handles transaction management and cleanup.
        Commits transaction if successful, rolls back if exception occurred.

        Args:
            exc_type: Exception type if raised, None otherwise
            exc_val: Exception value if raised, None otherwise
            exc_tb: Exception traceback if raised, None otherwise
        """
        if exc_type is not None:
            self.connection.rollback()
            logger.error(f"Exception occurred: {exc_val}")
        else:
            self.connection.commit()
        self.cursor.close()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def connect(self):
        """
        Establishes connection to PostgreSQL database.
        Retries up to 3 times with 5 second delay between attempts.

        Raises:
            DestinationWriterException: If connection fails after retries
        """
        try:
            logger.info(
                f"Connecting to PostgreSQL at host: {self.connection_details['host']} and port: {self.connection_details['port']}"
            )
            self.connection = psycopg2.connect(
                host=self.connection_details["host"],
                port=self.connection_details["port"],
                dbname=self.connection_details["database_name"],
                user=self.connection_details["username"],
                password=self.connection_details["password"],
            )
            logger.info("Successfully Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise DestinationWriterException(
                f"Failed to connect to PostgreSQL: {str(e)}"
            )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def write(self, vectors):
        """
        Writes vector data to PostgreSQL using pgvector extension.
        Creates table if it doesn't exist and inserts vector data.

        Args:
            vectors: List of dictionaries containing file_name and vector data

        Raises:
            DestinationWriterException: If write operation fails after retries
        """
        try:
            with self:
                # Enable the pgvector extension if not already enabled
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create SQL-safe identifier for table name
                pg_table_name = sql.Identifier(self.connection_details["table_name"])

                # Determine vector dimensions from the first vector
                embedding_dimension = (
                    len(vectors[0]["vector"])
                    if vectors and "vector" in vectors[0]
                    else 1024
                )
                logger.info(f"Detected vector dimension: {embedding_dimension}")

                # Create table with vector data type if it doesn't exist
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            file_name TEXT NOT NULL,
                            data VECTOR({}) NOT NULL
                        )
                    """
                    ).format(pg_table_name, sql.Literal(embedding_dimension)),
                )
                logger.info(
                    f"Table {self.connection_details['table_name']} created (if does not exist) with {embedding_dimension} dimensions"
                )

                # Insert vectors into the table
                logger.info(f"Writing {len(vectors)} vectors to PostgreSQL")
                for vector_item in vectors:
                    if len(vector_item["vector"]) != embedding_dimension:
                        logger.warning(
                            f"Vector dimension mismatch: expected {embedding_dimension}, got {len(vector_item['vector'])}"
                        )
                        continue

                    self.cursor.execute(
                        sql.SQL(
                            "INSERT INTO {} (file_name, data) VALUES (%s, %s)"
                        ).format(pg_table_name),
                        (vector_item["file_name"], vector_item["vector"]),
                    )
                logger.info("Successfully written vectors to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to write to PostgreSQL: {str(e)}")
            raise DestinationWriterException(f"Failed to write to PostgreSQL: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def backup(self):
        """
        Creates a backup of the vector table by renaming it.

        Returns:
            bool: True if backup was created, None if source table doesn't exist

        Raises:
            DestinationWriterException: If backup operation fails after retries
        """
        try:
            with self:
                # Extract the plain string table name for use with parameters
                original_table_name = self.connection_details["table_name"]

                # Create an Identifier object for safe SQL usage
                original_table_sql = sql.Identifier(original_table_name)

                # Check if the table exists before attempting backup
                self.cursor.execute(
                    sql.SQL(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = %s
                        )
                    """
                    ),
                    (original_table_name,),
                )
                table_exists = self.cursor.fetchone()[0]
                if not table_exists:
                    logger.info(
                        f"Table {original_table_name} does not exist, no backup made."
                    )
                    return None

                # Create backup by renaming the table
                backup_table_sql = sql.Identifier(f"backup_{original_table_name}")
                self.cursor.execute(
                    sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                        original_table_sql, backup_table_sql
                    )
                )
                logger.info(
                    f"Successfully backed up '{original_table_name}' data in PostgreSQL"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to backup data in PostgreSQL: {str(e)}")
            raise DestinationWriterException(f"Failed to backup PostgreSQL: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def restore(self):
        """
        Restores the table from its backup by dropping the current table (if it exists)
        and renaming the backup table to the original name.

        Returns:
            bool: True if restore succeeded, False if backup doesn't exist

        Raises:
            DestinationWriterException: If restore operation fails after retries
        """
        try:
            with self:
                # Extract table name strings
                original_table_name = self.connection_details["table_name"]
                backup_table_name = f"backup_{original_table_name}"

                # Create SQL-safe identifiers
                original_table_sql = sql.Identifier(original_table_name)
                backup_table_sql = sql.Identifier(backup_table_name)

                # Check if backup table exists
                self.cursor.execute(
                    sql.SQL(
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = %s AND column_name = 'data'
                        """
                    ),
                    (backup_table_name,),
                )
                backup_exists = self.cursor.fetchone() is not None

                if not backup_exists:
                    logger.warning(f"No backup table found for {original_table_name}")
                    return False

                # Drop current table if it exists
                self.cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(original_table_sql)
                )

                # Rename backup table to original name
                self.cursor.execute(
                    sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                        backup_table_sql, original_table_sql
                    )
                )
                logger.info(
                    f"Successfully restored {original_table_name} table from backup in PostgreSQL"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to restore PostgreSQL: {str(e)}")
            raise DestinationWriterException(f"Failed to restore PostgreSQL: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def delete_backup(self):
        """
        Removes the backup table if it exists.

        Raises:
            DestinationWriterException: If delete operation fails after retries
        """
        try:
            with self:
                backup_table_name = f"backup_{self.connection_details['table_name']}"
                backup_table_sql = sql.Identifier(backup_table_name)

                self.cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(backup_table_sql)
                )
                logger.info(
                    f"Successfully deleted backup table '{backup_table_name}' in PostgreSQL"
                )
        except Exception as e:
            logger.error(f"Failed to delete backup table in PostgreSQL: {str(e)}")
            raise DestinationWriterException(
                f"Failed to delete backup table in PostgreSQL: {str(e)}"
            )


class DatabricksWriter(DestinationWriter):
    def __init__(self, connection_details):
        self.workspace_url = connection_details["workspace_url"]
        self.token = connection_details["token"]
        self.warehouse_id = connection_details["warehouse_id"]
        self.database_name = connection_details["database_name"]
        self.table_name = connection_details["table_name"]
        self.connection = None  # Simulating a connection for consistency

    def __enter__(self):
        """Simulating an open connection to Databricks"""
        if not self.connection:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle transaction commit or rollback"""
        if exc_type is not None:
            logger.error(f"Exception occurred: {exc_val}")
        else:
            logger.info("✅ Successfully completed Databricks operations.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def connect(self):
        """Databricks does not require a persistent connection."""
        logger.info("✅ Databricks does not require an explicit connection step.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def create_table(self):
        """Ensures that the vector table exists before inserting data."""
        try:
            # Ensure the schema exists
            create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {self.database_name};"
            self.run_sql_query(create_schema_sql)

            # Create the table if it does not exist
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database_name}.{self.table_name} (
                id BIGINT,
                embedding ARRAY<FLOAT>,  -- Store vector as an array
                metadata STRING
            )
            USING DELTA;
            """
            self.run_sql_query(create_table_sql)

            logger.info(
                f"✅ Table `{self.database_name}.{self.table_name}` created (or already exists) in Databricks."
            )

        except Exception as e:
            logger.error(f"🚨 Failed to create table in Databricks: {str(e)}")
            raise DestinationWriterException(
                f"Failed to create table in Databricks: {str(e)}"
            )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def write(self, vectors):
        """Inserts vectors into the Databricks Delta table with the correct schema."""
        try:
            # Ensure the table exists before writing
            sql_query = f"""
                CREATE TABLE IF NOT EXISTS {self.database_name}.{self.table_name} (
                    file_name STRING,
                    embedding ARRAY<FLOAT>
                )
            """
            self.execute_sql(sql_query)

            vector_dimensions = (
                len(vectors[0]["vector"])
                if vectors and "vector" in vectors[0]
                else 1024
            )
            logger.info(f"Detected vector dimension: {vector_dimensions}")

            values = []
            for vector in vectors:
                file_name = vector.get("file_name")
                embedding = vector.get("vector")

                if not file_name or not embedding:
                    logger.warning(
                        "⚠️ Skipping entry due to missing file_name or embedding."
                    )
                    continue

                if len(embedding) != vector_dimensions:
                    logger.warning(
                        f"⚠️ Vector dimension mismatch: expected {vector_dimensions}, got {len(embedding)}"
                    )
                    continue  # Skip invalid vectors

                # Convert embedding list to Databricks ARRAY format
                embedding_sql = f"ARRAY({', '.join(map(str, embedding))})"
                values.append(f"('{file_name}', {embedding_sql})")

            if not values:
                logger.warning("⚠️ No valid vectors to insert.")
                return

            values_str = ", ".join(values)
            sql_query = f"INSERT INTO {self.database_name}.{self.table_name} (file_name, embedding) VALUES {values_str};"  # nosec
            self.execute_sql(sql_query)

            logger.info(f"✅ Successfully stored {len(vectors)} vectors in Databricks.")
        except Exception as e:
            logger.error(f"🚨 Failed to store vectors in Databricks: {str(e)}")
            raise DestinationWriterException(f"Failed to write to Databricks: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def backup(self):
        """Creates a backup of the table in Databricks."""
        backup_table_name = f"backup_{self.table_name}"
        sql_query = f"CREATE TABLE IF NOT EXISTS {self.database_name}.{backup_table_name} AS SELECT * FROM {self.database_name}.{self.table_name};"  # nosec
        try:
            self.execute_sql(sql_query)
            logger.info(
                f"✅ Successfully backed up `{self.database_name}.{self.table_name}` to `{backup_table_name}`."
            )
        except Exception as e:
            logger.error(f"🚨 Failed to backup Databricks table: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def restore(self):
        """Restores a backup of the table in Databricks."""
        backup_table_name = f"backup_{self.table_name}"
        sql_query = f"""
            DROP TABLE IF EXISTS {self.database_name}.{self.table_name};
            CREATE TABLE {self.database_name}.{self.table_name} AS SELECT * FROM {self.database_name}.{backup_table_name};
        """  # nosec
        try:
            self.execute_sql(sql_query)
            logger.info(
                f"✅ Successfully restored `{self.database_name}.{self.table_name}` from backup."
            )
        except Exception as e:
            logger.error(f"🚨 Failed to restore Databricks table: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def delete_backup(self):
        """Deletes the backup of the table in Databricks."""
        backup_table_name = f"backup_{self.table_name}"
        sql_query = f"DROP TABLE IF EXISTS {self.database_name}.{backup_table_name};"
        try:
            self.execute_sql(sql_query)
            logger.info(
                f"✅ Successfully deleted backup `{backup_table_name}` in Databricks."
            )
        except Exception as e:
            logger.error(f"🚨 Failed to delete backup table in Databricks: {str(e)}")
            raise DestinationWriterException(
                f"Failed to delete backup in Databricks: {str(e)}"
            )

    def execute_sql(self, sql_query):
        """Runs a SQL query in Databricks and logs errors properly."""
        sql_url = f"{self.workspace_url}/api/2.0/sql/statements"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "statement": sql_query,
            "warehouse_id": self.warehouse_id,  # Ensure warehouse_id is used
            "format": "JSON",
        }
        response = requests.post(sql_url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            logger.error(
                f"🚨 Databricks SQL Execution Failed: HTTP {response.status_code}"
            )
            raise DestinationWriterException(
                f"Failed to execute SQL in Databricks: HTTP {response.status_code}"
            )
        logger.info("✅ Databricks SQL query executed successfully.")


class DestinationWriterFactory:
    @staticmethod
    def get(destination_info):
        logger.debug(
            f"🔍 Creating destination writer for type: {destination_info.get('destination_type', 'unknown')}"
        )
        if destination_info["destination_type"] == "pg_vector":
            return PGVectorWriter(destination_info, destination_info["dataset_name"])
        elif destination_info["destination_type"] == "databricks":
            logger.debug(
                f"🟢 Using DatabricksWriter for dataset: {destination_info['dataset_name']}"
            )
            return DatabricksWriter(destination_info)
        else:
            raise DestinationWriterException("Unsupported destination type")
