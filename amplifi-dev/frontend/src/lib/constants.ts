export enum SortDirection {
  ASCENDING = "asc",
  DESCENDING = "desc",
}

export const constants = {
  ESC: "Escape",
  ENTER: "Enter",
  UNSTRUCTURED_CHAT_APP: "unstructured_chat_app",
  UNSTRUCTURED_CHAT_APP_LABEL: "Unstructured Chat App",
  SQL_CHAT_APP: "sql_chat_app",
  SQL_CHAT_APP_LABEL: "SQL Chat App",
  AGENTIC_CHAT_APP: "agentic",
  AUTH_TOKEN: "authtoken",
  REFRESH_TOKEN: "refreshToken",
  JWT_TOKEN: "jwtToken",
  USER: "authUser",
  SOURCE: {
    AZURE: "azure",
    AWS: "aws",
    POSTGRES: "postgres",
    MYSQL: "mysql",
    GROOVE: "groove",
  },
  SOURCE_TYPE: {
    AZURE: "azure_storage",
    AWS: "aws",
    POSTGRES: "pg_db",
    MYSQL: "mysql_db",
    GROOVE: "groove_source",
    LOCAL: "Local Data",
  },
  SOURCE_TYPE_LABEL: {
    AZURE: "Azure Blob Datasource",
    AWS: "AWS S3 Datasource",
    POSTGRES: "Postgres SQL Database",
    MYSQL: "MySQL Database",
    GROOVE: "Groove Source",
  },
  SORTING: {
    ASCENDING: SortDirection.ASCENDING,
    DESCENDING: SortDirection.DESCENDING,
  },
  GROOVE_CONNECTOR_FEATURE: "groove_connector_feature",
};

export const ROLES = {
  ADMIN: "Amplifi_Admin",
  DEVELOPER: "Amplifi_Developer",
  MEMBER: "Amplifi_Member",
};

export const IngestionStatus = {
  SUCCESS: "success",
  FAILED: "failed",
  EXCEPTION: "exception",
  PROCESSING: "processing",
  SPLITTING: "splitting",
  EXTRACTION_COMPLETED: "extraction_completed",
  NOT_STARTED: "not_started",
};

export const GraphStatus = {
  SUCCESS: "success",
  FAILED: "failed",
  PENDING: "pending",
  NOT_STARTED: "not_started",
};
