import os
import secrets
from enum import Enum
from typing import Any, List, Optional, Union
from uuid import UUID

from pydantic import AnyHttpUrl, EmailStr, Field, PostgresDsn, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModeEnum(str, Enum):
    development = "development"
    production = "production"
    testing = "testing"


class Settings(BaseSettings):
    MODE: ModeEnum = ModeEnum.development
    API_VERSION: str = "v1"
    API_V1_STR: str = f"/api/{API_VERSION}"
    API_VERSION_V2: str = "v2"
    API_V2_STR: str = f"/api/{API_VERSION_V2}"
    PROJECT_NAME: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 1  # 1 hour
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 100  # 100 days
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_NAME: str
    DATABASE_CELERY_NAME: str = "celery_schedule_jobs"
    REDIS_HOST: str
    REDIS_PORT: int
    RABBITMQ_USER: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_HOST: str
    RABBITMQ_PORT: int
    LOGINRADIUS_API_KEY: str = ""
    LOGINRADIUS_API_SECRET: str = ""
    DB_POOL_SIZE: int = 83
    WEB_CONCURRENCY: int = 9
    POOL_SIZE: int = max(DB_POOL_SIZE // WEB_CONCURRENCY, 5)
    ASYNC_DATABASE_URI: Union[PostgresDsn, str] = ""
    SYNC_DATABASE_URI: Union[PostgresDsn, str] = ""
    VAULT_ADDR: str
    VAULT_TOKEN: str
    VAULT_SECRET_PATH: str
    LOGIN_APP: Optional[str] = ""
    DEPLOYED_ENV: str = "local"
    TABLE_EXTRACTION_DATA_PATH: Optional[str] = "/app/table-extraction-data"
    EVAL_QUESTION_ANSWER_FILES_PATH: Optional[str] = "/app/eval-question-answer-files"
    APPLICATIONINSIGHTS_CONNECTION_STRING: str = ""
    VECTOR_DATA_POINT_LIMIT: int = 5
    GROOVE_API_URL: str = "https://api.groovehq.com/v2/graphql"
    GROOVE_SOURCE_DIR: str = "/app/groove-files"
    GROOVE_DAYS_TO_INGEST: int = 30
    ALLOWED_FILE_EXTENSIONS: set = {
        # "doc",
        "docx",
        "pptx",
        "pdf",
        # "txt",
        # "json",
        "html",
        "xlsx",
        "csv",
        "md",
        # "xml",
        "png",
        "jpg",
        "jpeg",
        "wav",
        "mp3",
        "aac",
        # Video extensions
        "mp4",
        "avi",
        "mov",
        "wmv",
        "flv",
        "webm",
        "mkv",
    }
    ALLOWED_MIME_TYPES: set = {
        # "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/pdf",
        # "text/plain",
        # "application/json",
        "text/html",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
        "application/octet-stream",
        # "text/xml",
        "image/png",
        "image/jpg",
        "image/jpeg",
        "audio/wav",
        "audio/mpeg",
        "audio/aac",
        "audio/mp3",
        "audio/vnd.dlna.adts",
        # Video MIME types
        "video/mp4",
        "video/x-msvideo",  # AVI
        "video/quicktime",  # MOV
        "video/x-ms-wmv",  # WMV
        "video/x-flv",  # FLV
        "video/webm",  # WEBM
        "video/x-matroska",  # MKV
    }
    ALLOWED_IMAGE_PROCESSING_EXTENSIONS: set = {
        "png",
        "jpg",
        "jpeg",
        "pdf",
    }
    ALLOWED_IMAGE_PROCESSING_MIME_TYPES: set = {
        "image/png",
        "image/jpg",
        "image/jpeg",
        "application/pdf",
    }
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
    AZURE_KEY_VAULT_NAME: Optional[str] = Field(
        default=None, env="AZURE_KEY_VAULT_NAME"
    )
    AZURE_CLIENT_ID: Optional[str] = Field(default=None, env="AZURE_CLIENT_ID")
    AZURE_TENANT_ID: Optional[str] = Field(default=None, env="AZURE_TENANT_ID")
    AZURE_CLIENT_SECRET: Optional[str] = Field(default=None, env="AZURE_CLIENT_SECRET")
    MSAL_AZURE_CLIENT_ID: Optional[str] = ""

    ONGC_PG_SERVER: str = ""
    ONGC_PG_DRIVER: str = ""
    ONGC_PG_PORT: int = 5432
    ONGC_PG_DATABASE: str = ""
    ONGC_PG_USERNAME: str = ""
    ONGC_PG_PASSWORD: str = ""

    ONGC_VECTOR_SERVER: str = ""
    ONGC_VECTOR_DRIVER: str = ""
    ONGC_VECTOR_PORT: int = 5432
    ONGC_VECTOR_DATABASE: str = ""
    ONGC_VECTOR_USERNAME: str = ""
    ONGC_VECTOR_PASSWORD: str = ""

    INGEST_TASK_SLEEP_SECONDS: int = 0
    RESET_PASSWORD_URL: Optional[str] = ""
    STRUCTURED_SOURCES: List[str] = ["pg_db", "mysql_db"]
    VERIFY_EMAIL_URL: Optional[str] = ""

    UNSTRUCTURED_SOURCES: List[str] = ["azure_storage", "aws_s3", "azure_fabric"]

    # File splitting settings
    MAX_TOKENS_PER_SPLIT: int = 10000  # Default token limit per split
    MIN_SPLIT_SIZE: int = 200  # Minimum tokens per split to ensure meaningful chunks

    # Rate limiter settings for Azure
    AZURE_RATE_LIMIT_TOKENS_PER_MINUTE: int = 900000  # Tokens per minute limit
    RATE_LIMIT_RETRY_SECONDS: int = 30  # Retry after seconds

    LOGINRADIUS_SITEURL: str = ""
    LOGINRADIUS_JWTAPPNAME: str = ""
    MAX_DESCRIPTION_LENGTH: int = 100
    MAX_NAME_LENGTH: int = 25

    # BLOB_SAS_TOKEN_RAG: Optional[str] = Field(default=None, env="BLOB_SAS_TOKEN_RAG")
    # BLOB_URL_RAG: Optional[str] = Field(default=None, env="BLOB_URL_RAG")

    REDIS_MODE: Optional[str] = "standalone"  # standalone | cluster
    REDIS_CLUSTER_SIZE: int = 3

    # Benchmark logging configuration
    ENABLE_BENCHMARK_LOGS: bool = Field(
        default=False,
        env="ENABLE_BENCHMARK_LOGS",
        description="Enable ingestion benchmark logging to Azure Monitor",
    )

    @field_validator("ASYNC_DATABASE_URI", mode="after")
    def assemble_db_connection(
        cls, v: Union[str, None], info: FieldValidationInfo
    ) -> Any:
        if isinstance(v, str):
            if v == "":
                return PostgresDsn.build(
                    scheme="postgresql+asyncpg",
                    username=info.data["DATABASE_USER"],
                    password=info.data["DATABASE_PASSWORD"],
                    host=info.data["DATABASE_HOST"],
                    port=info.data["DATABASE_PORT"],
                    path=info.data["DATABASE_NAME"],
                )
        return v

    @field_validator("SYNC_DATABASE_URI", mode="after")
    def assemble_sync_db_connection(
        cls, v: Union[str, None], info: FieldValidationInfo
    ) -> Any:
        if isinstance(v, str):
            if v == "":
                return PostgresDsn.build(
                    scheme="postgresql+psycopg2",
                    username=info.data["DATABASE_USER"],
                    password=info.data["DATABASE_PASSWORD"],
                    host=info.data["DATABASE_HOST"],
                    port=info.data["DATABASE_PORT"],
                    path=info.data["DATABASE_NAME"],
                )
        return v

    SYNC_CELERY_DATABASE_URI: Union[PostgresDsn, str] = ""

    @field_validator("SYNC_CELERY_DATABASE_URI", mode="after")
    def assemble_celery_db_connection(
        cls, v: Union[str, None], info: FieldValidationInfo
    ) -> Any:
        if isinstance(v, str):
            if v == "":
                return PostgresDsn.build(
                    scheme="postgresql+psycopg2",
                    username=info.data["DATABASE_USER"],
                    password=info.data["DATABASE_PASSWORD"],
                    host=info.data["DATABASE_HOST"],
                    port=info.data["DATABASE_PORT"],
                    path=info.data["DATABASE_CELERY_NAME"],
                )
        return v

    SYNC_CELERY_BEAT_DATABASE_URI: Union[PostgresDsn, str] = ""

    @field_validator("SYNC_CELERY_BEAT_DATABASE_URI", mode="after")
    def assemble_celery_beat_db_connection(
        cls, v: Union[str, None], info: FieldValidationInfo
    ) -> Any:
        if isinstance(v, str):
            if v == "":
                return PostgresDsn.build(
                    scheme="postgresql+psycopg2",
                    username=info.data["DATABASE_USER"],
                    password=info.data["DATABASE_PASSWORD"],
                    host=info.data["DATABASE_HOST"],
                    port=info.data["DATABASE_PORT"],
                    path=info.data["DATABASE_CELERY_NAME"],
                )
        return v

    ASYNC_CELERY_BEAT_DATABASE_URI: Union[PostgresDsn, str] = ""

    @field_validator("ASYNC_CELERY_BEAT_DATABASE_URI", mode="after")
    def assemble_async_celery_beat_db_connection(
        cls, v: Union[str, None], info: FieldValidationInfo
    ) -> Any:
        if isinstance(v, str):
            if v == "":
                return PostgresDsn.build(
                    scheme="postgresql+asyncpg",
                    username=info.data["DATABASE_USER"],
                    password=info.data["DATABASE_PASSWORD"],
                    host=info.data["DATABASE_HOST"],
                    port=info.data["DATABASE_PORT"],
                    path=info.data["DATABASE_CELERY_NAME"],
                )
        return v

    FIRST_SUPERUSER_EMAIL: EmailStr
    FIRST_SUPERUSER_PASSWORD: str

    ENCRYPT_KEY: str = secrets.token_urlsafe(32)
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]]

    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    AZURE_OPENAI_API_KEY: str
    AZURE_BASE_URL: str
    AZURE_API_VERSION: str
    AZURE_SPEECH_KEY: Optional[str] = ""
    AZURE_SPEECH_REGION: Optional[str] = ""
    AZURE_GPT_35_DEPLOYMENT_NAME: str = "gpt-35-turbo"

    AZURE_GPT_4o_URL_BATCH: Optional[str] = Field(
        default=None, env="AZURE_GPT_4o_URL_BATCH"
    )
    AZURE_GPT_4o_KEY_BATCH: Optional[str] = Field(
        default=None, env="AZURE_GPT_4o_KEY_BATCH"
    )
    AZURE_GPT_4o_VERSION_BATCH: Optional[str] = Field(
        default=None, env="AZURE_GPT_4o_VERSION_BATCH"
    )

    AZURE_GPT_4o_URL: str
    AZURE_GPT_4o_KEY: str
    AZURE_GPT_4o_VERSION: str
    AZURE_GPT_4o_DEPLOYMENT_NAME: str = "gpt-4o"

    AZURE_GPT_41_URL: str = ""
    AZURE_GPT_41_KEY: str = ""
    AZURE_GPT_41_VERSION: str = "2025-01-01-preview"
    AZURE_GPT_41_DEPLOYMENT_NAME: str = "gpt-4.1"

    AZURE_GPT_o3_URL: str = ""
    AZURE_GPT_o3_KEY: str = ""
    AZURE_GPT_o3_VERSION: str = "2025-01-01-preview"
    AZURE_GPT_o3_DEPLOYMENT_NAME: str = "o3-mini"

    AZURE_GPT_5_VERSION: str = "2025-04-01-preview"
    AZURE_GPT_5_DEPLOYMENT_NAME: str = "gpt-5"

    TEST_USER_EMAIL: str = "erik.wang+amplifi.test.superuser@thoughtswinsystems.com"
    TEST_USER_PASSWORD: str = "Password_test"
    TEST_ORG_UUID: UUID = "76d1d9a1-6a82-4051-8a60-762025764995"
    TEST_ROLE_ID: UUID = "19659507-69a6-46ab-a4f1-c07043554539"

    LOCAL_USER_EMAIL: str = "local@local.com"
    LOCAL_USER_PASSWORD: str = "Password"
    LOCAL_ORG_UUID: UUID = "76d1d9a1-6a82-4051-8a60-762025764995"
    LOCAL_ROLE_ID: UUID = "19659507-69a6-46ab-a4f1-c07043554539"
    AMPLIFI_DOCS_USERNAME: str = "admin"
    AMPLIFI_DOCS_PASSWORD: str = "password"

    MAX_FILE_SIZE_UPLOADED_FILES: int = 300 * 1024 * 1024
    MAX_CSV_EXPORT_SIZE: int = 10 * 1024 * 1024
    DEFAULT_UPLOAD_FOLDER: str = "/app/uploads"
    CSV_EXPORT_FOLDER: str = "/app/generated-files/csv-exports"
    TEMP_SPLITS_DIR: str = (
        f"{DEFAULT_UPLOAD_FOLDER}/temp_splits"  # Use existing upload folder
    )
    TEMP_PDF_IMG_DIR: str = f"{DEFAULT_UPLOAD_FOLDER}/temp_imgs/"

    # Split file cleanup setting
    SPLIT_CLEANUP_ENABLED: Optional[bool] = True  # Enable/disable split file cleanup

    MAX_INGESTION_RETRIES: int = 20

    MAX_PLOT_DATA: int = 400
    MAX_SQL_ROWS: int = 100
    MAX_SQL_COLUMNS: int = 100

    # Knowledge Graph Settings
    GRAPH_EXTRACTION_TIMEOUT_HOURS: int = 4  # Default to 4 hours

    AI_CONFERENCE_CHATAPP_UUID: UUID = "0196402a-3d85-7565-9bd5-3b749902459e"

    RATE_LIMIT_PER_SECOND: int = 50

    EMBEDDING_DIMENSIONS: int = 1024

    IMAGE_RESOLUTION_SCALE: float = 2.0
    MAX_SUMMARY_PAGES: int = 10  # Maximum number of pages to consider for summary

    DEFAULT_CHUNK_SIZE: int = 1000  # Default chunk size for text processing
    DEFAULT_CHUNK_OVERLAP: int = 100  # Default overlap for text chunks

    MAX_LOCK_AGE: int = 600  # 10 minutes

    # Document timeout settings
    DOCUMENT_PROCESSING_TIMEOUT_SECONDS: Optional[int] = Field(
        default=7200, env="DOCUMENT_PROCESSING_TIMEOUT_SECONDS"
    )

    VECTOR_SEARCH_TOOL_TOP_K: int = 10

    CONTEXT_RETRIEVED_SEARCH_SCORE_THRESHOLD: float = 0.35

    # Docling Model Settings
    DOCLING_DOWNLOAD_MODELS_AT_STARTUP: bool = Field(
        default=True, env="DOCLING_DOWNLOAD_MODELS_AT_STARTUP"
    )  # Whether to download models at startup
    DOCLING_FORCE_DOWNLOAD: bool = Field(
        default=True, env="DOCLING_FORCE_DOWNLOAD"
    )  # Force re-download even if models exist
    DOCLING_MODEL_DOWNLOAD_TIMEOUT: int = Field(
        default=300, env="DOCLING_MODEL_DOWNLOAD_TIMEOUT"
    )  # Timeout in seconds for model download

    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=os.path.expanduser("~/.env")
    )

    TAVILY_API_KEY: str = ""
    LOGFIRE_API_KEY: str = ""
    BRAVE_SEARCH_API_KEY: Optional[str] = Field(None, env="BRAVE_SEARCH_API_KEY")

    FLAGSMITH_ENVIRONMENT_KEY: Optional[str] = Field(
        default=None, env="FLAGSMITH_ENVIRONMENT_KEY"
    )

    # Docling OCR Language Settings
    SUPPORTED_OCR_LANGUAGES: List[str] = [
        "eng",  # English
        "jpn",  # Japanese
        "kor",  # Korean
        "chi_sim",  # Chinese Simplified
        "chi_tra",  # Chinese Traditional
        "spa",  # Spanish
        "fra",  # French
        "deu",  # German
        "ita",  # Italian
        "por",  # Portuguese
        "rus",  # Russian
        "ara",  # Arabic
        "hin",  # Hindi
        "nld",  # Dutch
        "tur",  # Turkish
        "lat",  # Latin
    ]

    KUZU_DB_DIR: Optional[str] = Field(default="/app/kuzu_dbs", env="KUZU_DB_DIR")

    SYSTEM_EXPORT_SOURCE_ID: str = (
        "00000000-0000-0000-0000-000000000001"  # Special UUID for system exports
    )

    ENABLE_VIDEO_INGESTION: bool = Field(
        True, env="ENABLE_VIDEO_INGESTION"
    )  # Enable or disable video ingestion feature
    # Video Processing Settings
    VIDEO_SEGMENT_LENGTH: int = Field(
        default=30, env="VIDEO_SEGMENT_LENGTH"
    )  # seconds per segment
    VIDEO_FRAMES_PER_SEGMENT: int = Field(
        default=5, env="VIDEO_FRAMES_PER_SEGMENT"
    )  # frames for processing
    VIDEO_AUDIO_FORMAT: str = Field(
        default="mp3", env="VIDEO_AUDIO_FORMAT"
    )  # audio output format
    VIDEO_OUTPUT_FORMAT: str = Field(
        default="mp4", env="VIDEO_OUTPUT_FORMAT"
    )  # video segment format
    VIDEO_ENABLE_CAPTIONING: bool = Field(
        default=True, env="VIDEO_ENABLE_CAPTIONING"
    )  # enable AI captioning
    VIDEO_ENABLE_TRANSCRIPTION: bool = Field(
        default=True, env="VIDEO_ENABLE_TRANSCRIPTION"
    )  # enable audio transcription
    # Video Model Settings
    VIDEO_WHISPER_MODEL: str = Field(
        default="Systran/faster-distil-whisper-large-v3", env="VIDEO_WHISPER_MODEL"
    )
    VIDEO_CAPTION_MODEL: str = Field(
        default="openbmb/MiniCPM-V-2_6-int4", env="VIDEO_CAPTION_MODEL"
    )
    # Video storage settings
    VIDEO_SEGMENTS_DIR: str = Field(
        default=f"{DEFAULT_UPLOAD_FOLDER}/video_segments", env="VIDEO_SEGMENTS_DIR"
    )  # Persistent storage for video segments
    VIDEO_TEMP_DIR: str = Field(
        default=f"{DEFAULT_UPLOAD_FOLDER}/temp_video_processing", env="VIDEO_TEMP_DIR"
    )  # Temporary video processing directory
    VIDEO_CLEANUP_ENABLED: bool = Field(
        default=False, env="VIDEO_CLEANUP_ENABLED"
    )  # Don't delete video segments by default
    VIDEO_DELETE_ORIGINAL_ENABLED: bool = Field(
        default=True, env="VIDEO_DELETE_ORIGINAL_ENABLED"
    )  # Don't delete original video files by default
    VIDEO_TRANSCRIPTION_MAX_WORKERS: int = Field(
        default=3, env="VIDEO_TRANSCRIPTION_MAX_WORKERS"
    )
    VIDEO_CAPTIONING_BATCH_SIZE: int = Field(
        default=3, env="VIDEO_CAPTIONING_BATCH_SIZE"
    )


settings = Settings()
