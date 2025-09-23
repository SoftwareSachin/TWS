# Use a more compatible PostgreSQL image for ARM64
FROM postgres:15

RUN apt-get update && \
    apt-get install -y postgresql-15-pgvector

# Copy initialization script
COPY create-dbs.sql /docker-entrypoint-initdb.d/

# Set environment variables
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV PGDATA=/var/lib/postgresql/data

# Expose port
EXPOSE 5432