CREATE DATABASE amplifi_db;
CREATE DATABASE celery_schedule_jobs;

-- Connect to amplifi_db and create the vector extension
\c amplifi_db;
CREATE EXTENSION IF NOT EXISTS vector;

--CREATE DATABASE r2r_db;
--CREATE USER r2r_user WITH ENCRYPTED PASSWORD 'r2r_password';
--GRANT ALL PRIVILEGES ON DATABASE r2r_db TO r2r_user;
