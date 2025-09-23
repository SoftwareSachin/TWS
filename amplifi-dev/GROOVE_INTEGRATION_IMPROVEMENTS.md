# Groove Connector Integration Improvements

## Overview
The Groove connector has been refactored to follow the same clean pattern as other source connectors (like Azure Blob), eliminating the raw POC implementation and providing a more robust, maintainable solution.

## Key Improvements

### 1. **Standardized Task Pattern**
- **Before**: Custom `groove_data_fetch_task` with manual triggering
- **After**: Standard `pull_files_from_groove_source_task` following Azure Blob pattern

### 2. **Proper Database Integration**
- **Before**: Files saved to volume without database tracking
- **After**: File records created in database with proper status tracking

### 3. **Unified Workflow**
- **Before**: Two-step process (fetch at dataset creation, ingest separately)
- **After**: Single-step process triggered at source creation/update

### 4. **Status Management**
- **Before**: No pull status tracking
- **After**: Proper `SourcePullStatus` management (STARTED, SUCCESS, FAILED)

### 5. **Error Handling**
- **Before**: Basic error handling
- **After**: Comprehensive error handling with retries and status updates

## New Architecture

### Task Structure
```python
@shared_task(name="tasks.pull_files_from_groove_source_task", bind=True, max_retries=3)
def pull_files_from_groove_source_task(
    self,
    workspace_id: UUID,
    user_id: UUID,
    source_id: UUID,
    api_key: str,
) -> List[Dict[str, Any]]:
```

### Integration Points
1. **Source Creation**: Automatically triggers file pull task
2. **Source Update**: Re-triggers file pull task
3. **Dataset Creation**: Links existing files to dataset
4. **Ingest Process**: Standard file processing pipeline

### Data Flow
1. **Setup**: Groove source created with API key
2. **Validation**: Connection test to Groove API
3. **Trigger**: File pull task launched automatically
4. **Fetch**: GraphQL API calls to get tickets
5. **Process**: Convert tickets to markdown files
6. **Store**: Save files to volume with database records
7. **Link**: Files linked to dataset during ingest

## Configuration Changes

### Celery Queue
```python
# Updated in celery.py
"tasks.pull_files_from_groove_source_task": {"queue": "file_pull_queue"},
```

### Task Mapping
```python
# Updated in source_connector_crud.py
"groove_source": (
    "tasks.pull_files_from_groove_source_task",
    self.get_groove_details,
    ["api_key"],
    self.check_groove_connection,
),
```

## Benefits

### 1. **Consistency**
- Follows same pattern as Azure Blob, AWS S3, etc.
- Standard error handling and status tracking
- Unified task management

### 2. **Reliability**
- Proper retry mechanism
- Database transaction management
- Comprehensive error handling

### 3. **Maintainability**
- Clean separation of concerns
- Standard task structure
- Consistent logging and monitoring

### 4. **Scalability**
- Uses standard Celery queue
- Proper resource management
- Database-driven file tracking

## Migration Notes

### Removed Files
- `backend/app/app/api/groove_data_fetch_task.py` (old POC task)
- `backend/app/app/api/groove_data_fetcher.py` (old task launcher)

### New Files
- `backend/app/app/api/groove_source_file_pull_task.py` (new standard task)

### Updated Files
- `backend/app/app/crud/source_connector_crud.py` (task mapping)
- `backend/app/app/crud/ingest_crud_v2.py` (volume file handling)
- `backend/app/app/crud/dataset_crud_v2.py` (removed custom logic)
- `backend/app/app/be_core/celery.py` (queue configuration)

## Usage

### Creating a Groove Source
```python
# Standard source creation - automatically triggers file pull
source_data = {
    "source_type": "groove_source",
    "source_name": "My Groove Support",
    "groove_api_key": "base64_encoded_api_key"
}
```

### File Processing
- Files are automatically fetched and stored in GROOVE_SOURCE_DIR volume
- Database records track file status and metadata
- Standard ingest process handles file processing

### Monitoring
- Check `SourcePullStatus` for task status
- Monitor Celery queue `file_pull_queue`
- Review logs for detailed processing information

## Future Enhancements

1. **Configuration Options**: Make ticket count configurable
2. **Incremental Updates**: Only fetch new tickets since last pull
3. **File Format Options**: Support different output formats
4. **Advanced Filtering**: Filter tickets by date, status, etc.
5. **Webhook Integration**: Real-time updates from Groove
