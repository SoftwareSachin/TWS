# Amplifi - AI Data Platform

**Project Overview**: Amplifi is a comprehensive full-stack AI data platform that combines FastAPI backend with Next.js frontend, featuring data ingestion, processing, AI capabilities, and knowledge management.

**Last Updated**: September 23, 2025

## Recent Changes

### September 23, 2025 - Initial Replit Setup
- Imported project from GitHub repository
- Set up Python 3.12 backend environment with FastAPI
- Installed Node.js 20 frontend environment with Next.js
- Configured frontend for Replit proxy compatibility (0.0.0.0:5000)
- Created frontend workflow running on port 5000
- Fixed Next.js configuration warnings

## Project Architecture

### Backend (FastAPI)
- **Location**: `amplifi-dev/backend/app/`
- **Language**: Python 3.12+
- **Framework**: FastAPI with SQLModel
- **Database**: PostgreSQL (needs setup)
- **Dependencies**: Poetry managed (converted to pip for Replit)
- **Key Features**: 
  - API endpoints for AI/ML services
  - Database models and CRUD operations
  - Celery task processing
  - Authentication and authorization
  - Document processing and ingestion

### Frontend (Next.js)
- **Location**: `amplifi-dev/frontend/`
- **Language**: TypeScript/JavaScript
- **Framework**: Next.js 15.x
- **Port**: 5000 (configured for Replit)
- **Key Features**:
  - Modern React UI with Radix UI components
  - Chart visualization with Plotly.js
  - Authentication integration
  - Chat interface for AI interactions

### Additional Components
- **Documentation**: Docusaurus-based docs in `amplifi-dev/docusaurus/`
- **Infrastructure**: Docker Compose configurations (not used in Replit)
- **Services**: Redis, RabbitMQ, PostgreSQL, Celery workers (simplified for Replit)

## User Preferences

- **Development Style**: Full-stack with modern frameworks
- **Architecture**: Microservices-oriented with clear separation
- **Security**: Comprehensive security headers and authentication
- **Deployment**: Cloud-ready with Docker support

## Environment Configuration

The project uses environment variables defined in `.env` file (copied from `.env.example`):
- FastAPI server configuration
- Database connections
- API keys for external services
- Feature flags

## Current Status

✅ **Completed**:
- Project structure analysis
- Frontend and backend dependency installation
- Environment file creation
- Frontend proxy configuration for Replit
- Workflow setup for frontend on port 5000

🚧 **In Progress**:
- Frontend currently running and accessible

⏳ **Pending**:
- PostgreSQL database setup (permission issue to resolve)
- Backend service startup
- Full integration testing
- Deployment configuration

## Notes

- Frontend configured to work with Replit's proxy system using `0.0.0.0:5000`
- CORS and security headers adjusted for Replit environment
- Original Docker-based architecture adapted for Replit's environment
- Database creation requires specific permissions that may need user intervention