# 🎥 Video Ingestion Technical Implementation Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Technical Deep Dive](#technical-deep-dive)
3. [Development Guide](#development-guide)
4. [Troubleshooting](#troubleshooting)
5. [Performance Optimization](#performance-optimization)
6. [Testing Guide](#testing-guide)

## Quick Start

### Prerequisites
- Docker with GPU support (NVIDIA Container Toolkit)
- NVIDIA GPU with CUDA 12.6+ support
- At least 8GB GPU memory recommended
- 50GB+ disk space for models and processing

### Enable Video Ingestion
```bash
# 1. Set environment variables
export ENABLE_VIDEO_INGESTION=true
export ENABLE_GPU=true
export CUDA_VERSION=12.6

# 2. Build and run with video support
make run-dev-build-video

# 3. Verify video worker is running
docker logs celery_video_ingestion_worker
```

### Upload Your First Video
1. Navigate to the file upload interface
2. Select a video file (MP4, AVI, MOV, etc.)
3. Upload and wait for processing
4. Search for content using transcribed text or visual descriptions

## Technical Deep Dive

### Core Components

#### 1. Video Ingestion Task (`video_ingestion_task.py`)
The main Celery task that orchestrates the entire video processing pipeline.

**Key Functions:**
```python
@celery.task(name="tasks.video_ingestion_task", bind=True, acks_late=True)
def video_ingestion_task(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    metadata: Optional[dict] = None,
    skip_successful_files: bool = True,
    retry_count: int = 0,
    max_retries: int = 5,
    retry_reason: Optional[str] = None,
) -> Dict[str, Any]:
```

**Processing Flow:**
1. **Validation**: File existence, format, and feature flag checks
2. **Token Acquisition**: Rate limiting and resource management
3. **Document Management**: Create/update database records
4. **Workspace Processing**: Handle concurrent processing with Redis locks
5. **Content Processing**: Video segmentation, transcription, captioning
6. **Storage**: Save chunks and update document status
7. **Cleanup**: Remove temporary files and release resources

#### 2. Video Model Manager (`video_model_manager.py`)
Centralized management of AI models with GPU optimization.

**Model Initialization:**
```python
class VideoModelManager:
    def get_whisper_model(self) -> Optional[object]:
        """Get Whisper transcription model"""
        
    def get_caption_model(self) -> Tuple[Optional[object], Optional[object]]:
        """Get MiniCPM-V caption model and tokenizer"""
        
    def initialize_models_at_startup(self, force_download: bool = False) -> bool:
        """Pre-load models during container startup"""
```

**GPU Memory Management:**
```python
def cleanup_video_gpu_memory() -> None:
    """Clean up GPU memory after processing"""
    try:
        import torch
        torch.cuda.empty_cache()
        logger.debug("Cleaned up GPU memory")
    except Exception as e:
        logger.warning(f"Failed to cleanup GPU memory: {e}")
```

#### 3. Video Processing Utils (`video_ingestion/utils/`)

**Video Splitting (`video_split.py`):**
```python
def split_video(
    video_path: str,
    working_dir: str,
    segment_length: int,
    num_frames_per_segment: int,
    audio_output_format: str = "mp3",
):
    """Split video into segments and extract audio"""
    # Creates segments with smart merging for short end segments
    # Extracts audio in parallel with video segmentation
    # Returns segment mapping and timing information
```

**Transcription (`transcription.py`):**
```python
def speech_to_text(
    video_name: str,
    working_dir: str,
    segment_index2name: Dict,
    audio_output_format: str,
    whisper_model: object,
    max_workers: int = 4,
) -> Dict[str, str]:
    """Convert audio segments to text using Whisper"""
    # Parallel processing of audio segments
    # Handles multiple audio formats
    # Returns timestamped transcriptions
```

**Captioning (`captioning.py`):**
```python
def segment_caption(
    video_name: str,
    video_path: str,
    segment_index2name: Dict,
    transcripts: Dict,
    segment_times_info: Dict,
    captions: Dict,
    error_queue,
    caption_model: object,
    caption_tokenizer: object,
    batch_size: int = 3,
):
    """Generate visual captions using MiniCPM-V"""
    # Batch processing for efficiency
    # Frame sampling and analysis
    # Context-aware caption generation
```

**Video Cleanup (`video_cleanup_utils.py`):**
```python
async def delete_video_segments(
    workspace_id: UUID,
    file_id: UUID,
    force: bool = False,
    db_session: AsyncSession = None,
) -> bool:
    """Delete video segments with reference checking"""
    # Only deletes if no active references exist or if forced
    # Returns True if deleted, False if preserved due to references

async def cleanup_orphaned_video_segments(
    workspace_id: UUID, 
    db_session: AsyncSession = None
) -> int:
    """Clean up orphaned segments using 4-case logic"""
    # Scans workspace segment directories
    # Applies video present/deleted + reference logic
    # Returns count of cleaned directories

async def check_video_segments_still_referenced(
    workspace_id: UUID, 
    file_id: UUID, 
    db_session: AsyncSession = None
) -> bool:
    """Optimized reference checking with caching"""
    # Single JOIN query with JSON path search
    # In-memory cache with 5-minute TTL
    # Fallback to document existence check
```

### Workspace-Level Processing Architecture

#### Concurrent Processing Strategy
```python
def _handle_workspace_video_processing(
    workspace_id: str,
    file_id: str,
    file_path: str,
    document_id: str,
    dataset_id: str,
    task_id: str,
) -> Dict[str, Any]:
    """Handle workspace-level video processing with concurrent coordination"""
```

**Benefits:**
- **Deduplication**: Same video processed once per workspace
- **Concurrency Safety**: Redis locks prevent duplicate processing
- **Resource Efficiency**: Multiple datasets reuse segments
- **Scalability**: Supports multiple concurrent workers

#### Storage Structure
```
/app/uploads/video_segments/
├── {workspace_id}/
│   └── {file_id}/
│       ├── segment_timestamp-0-0-30.mp4
│       ├── segment_timestamp-1-30-60.mp4
│       ├── ...
│       └── segments_metadata.json
```

**Metadata Format:**
```json
{
  "processing_completed": true,
  "segment_count": 10,
  "total_duration": 300,
  "processing_completed_at": "2024-01-15T10:30:00Z",
  "segments_info": {
    "0": {
      "segment_name": "timestamp-0-0-30",
      "start_time": 0,
      "end_time": 30,
      "transcript": "Hello, welcome to this video...",
      "caption": "A person speaking in front of a whiteboard...",
      "frame_times": [0, 6, 12, 18, 24],
      "video_segment_path": "/app/uploads/video_segments/..."
    }
  },
  "document_data": {
    "description": "Comprehensive video summary...",
    "description_embedding": [0.1, 0.2, ...],
    "document_metadata": {
      "video_name": "example_video",
      "file_size": 52428800,
      "video_duration": 300,
      "total_segments": 10
    }
  }
}
```

### Database Integration

#### Document Model Extensions
```python
class Document(Base):
    # Existing fields...
    document_type = Column(Enum(DocumentTypeEnum))  # Added: Video
    description = Column(Text)  # AI-generated video summary
    description_embedding = Column(Vector(1024))  # Semantic search
    document_metadata = Column(JSONB)  # Video-specific metadata
```

#### Document Chunk Model
```python
class DocumentChunk(Base):
    # Existing fields...
    chunk_type = Column(Enum(ChunkTypeEnum))  # Added: VideoSegment
    chunk_text = Column(Text)  # "Caption: ... Transcript: ..."
    chunk_embedding = Column(Vector(1024))  # Segment embedding
    chunk_metadata = Column(JSONB)  # Segment details
```

### AI Model Integration

#### Model Download Strategy
**Smart Build-Time and Runtime Approach**: The system intelligently handles model downloads based on environment:

**During Docker Build:**
- ✅ **GPU Available** (local builds): Downloads models during build for faster startup
- ✅ **No GPU** (CI/CD): Skips model downloads, installs CPU-compatible dependencies
- ✅ **Automatic Detection**: Uses `nvidia-smi` to detect GPU availability
- ✅ **Fallback Handling**: Gracefully handles both scenarios

**At Runtime:**
- ✅ **Models Present**: Uses pre-downloaded models for immediate processing
- ✅ **Models Missing**: Downloads models automatically when first needed
- ✅ **CUDA Detection**: Automatically uses quantized models when GPU is available
- ✅ **CPU Fallback**: Falls back to non-quantized models when CUDA unavailable

#### Whisper Integration
```python
# Model: Systran/faster-distil-whisper-large-v3
# Downloaded during build (if GPU available) or at runtime
from faster_whisper import WhisperModel

model = WhisperModel(
    "Systran/faster-distil-whisper-large-v3",
    device="cuda"  # Automatically detects CUDA availability
)

segments, info = model.transcribe(
    audio_path,
    beam_size=5,
    language="en"
)
```

#### MiniCPM-V Integration
```python
# Model: openbmb/MiniCPM-V-2_6-int4 (quantized) or openbmb/MiniCPM-V-2_6 (fallback)
# Smart loading based on CUDA availability
import torch
from transformers import AutoModel, AutoTokenizer

# System automatically chooses appropriate model version
if torch.cuda.is_available():
    # Use quantized model for GPU
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-V-2_6-int4",
        revision="06219bd",
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype="auto",
    )
else:
    # Fallback to non-quantized model for CPU
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="float32",
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_name,  # Matches the model version used
    trust_remote_code=True
)

# Process video frames
response = model.chat(
    image=frame,
    msgs=[{"role": "user", "content": "Describe this video frame"}],
    tokenizer=tokenizer
)
```

#### Model Caching and Build Behavior
**Build-Time (Local with GPU):**
- Models downloaded during Docker build
- Immediate availability at container startup
- Optimal for development and production deployments

**Build-Time (CI/CD without GPU):**
- Models skipped during build (no GPU available)
- CPU-compatible dependencies installed
- Models download at runtime when needed

**Runtime Behavior:**
- **Cache location**: `~/.cache/huggingface/`
- **First run**: Downloads missing models (5-10 minutes if not cached)
- **Subsequent runs**: Fast startup using cached models
- **Persistence**: Cache survives container restarts when properly mounted

#### Azure OpenAI Integration
```python
# Video description generation using GPT-4o
def _generate_video_description(segments_info: Dict, config: Dict) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an expert video content analyst..."
        },
        {"role": "user", "content": prompt}
    ]
    
    summary = chat_completion_with_retry(
        messages=messages,
        model="gpt-4o",
        max_tokens=1000,
        temperature=0.3
    )
```

## Development Guide

### Setting Up Development Environment

#### 1. Prerequisites
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### 2. Environment Configuration
```bash
# .env file additions
ENABLE_VIDEO_INGESTION=true
ENABLE_GPU=true
CUDA_VERSION=12.6

# Video processing settings
VIDEO_SEGMENT_LENGTH=30
VIDEO_FRAMES_PER_SEGMENT=5
VIDEO_ENABLE_TRANSCRIPTION=true
VIDEO_ENABLE_CAPTIONING=true

# Model settings
VIDEO_WHISPER_MODEL=Systran/faster-distil-whisper-large-v3
VIDEO_CAPTION_MODEL=openbmb/MiniCPM-V-2_6-int4

# Storage settings
VIDEO_SEGMENTS_DIR=/app/uploads/video_segments
VIDEO_TEMP_DIR=/app/uploads/temp_video_processing
VIDEO_DELETE_ORIGINAL_ENABLED=true
```

#### 3. Development Commands
```bash
# Build containers with video support (smart GPU detection)
make run-dev-build-video

# Run development environment
make run-dev-video

# Stop development environment
make stop-dev

# Run video-specific tests
make pytest-video
make pytest-video-unit
make pytest-video-integration
make pytest-video-api
```

#### 4. Build and Runtime Behavior
**Local Development (with GPU):**
```bash
# Build will detect GPU and download models
make run-dev-build-video

# Expected build output:
# "GPU detected - proceeding with model downloads..."
# "✓ Caption model downloaded successfully"
# "✓ Whisper model downloaded successfully"

# Container starts immediately with models ready
docker logs celery_video_ingestion_worker | grep "Video models initialized"
```

**CI/CD Environment (without GPU):**
```bash
# Build skips model downloads
# Expected build output:
# "No GPU detected (likely CI environment) - skipping model downloads"
# "Models will be downloaded at runtime when GPU is available"

# At runtime, models download when first needed
# First video processing: 5-10 minutes (one-time download)
# Subsequent processing: Normal speed
```

#### 5. Model Cache Management
```bash
# Check model cache status
docker exec celery_video_ingestion_worker python -c "
import os
cache_dir = os.path.expanduser('~/.cache/huggingface')
if os.path.exists(cache_dir):
    size = sum(os.path.getsize(os.path.join(dirpath, filename))
              for dirpath, dirnames, filenames in os.walk(cache_dir)
              for filename in filenames)
    print(f'Model cache size: {size / (1024**3):.2f} GB')
else:
    print('Model cache not found - models will download on first use')
"

# Clear model cache if needed (forces re-download)
docker exec celery_video_ingestion_worker rm -rf ~/.cache/huggingface

# Pre-warm model cache (optional - for faster first video)
docker exec celery_video_ingestion_worker python -c "
from app.utils.video_model_manager import VideoModelManager
manager = VideoModelManager()
success = manager.initialize_models_at_startup(force_download=True)
print(f'Models pre-loaded: {success}')
"
```

### Code Structure

#### Adding New Video Processing Features
```python
# 1. Add utility function in video_ingestion/utils/
def new_processing_function(video_path: str, config: Dict) -> Dict:
    """New video processing capability"""
    pass

# 2. Update video_ingestion_task.py
def _process_video_directly_to_workspace(
    file_path: str,
    segment_dir: str,
    working_dir: str,
) -> Dict[str, Any]:
    # Add your new processing step
    new_results = new_processing_function(file_path, config)
    
# 3. Update configuration in config.py
NEW_VIDEO_SETTING: bool = Field(
    default=True, env="NEW_VIDEO_SETTING"
)

# 4. Add tests
def test_new_processing_function():
    """Test new video processing capability"""
    pass
```

#### Model Integration Pattern
```python
# 1. Add model to video_model_manager.py
def get_new_model(self) -> Optional[object]:
    """Get new AI model instance"""
    if self._new_model is not None:
        return self._new_model
        
    try:
        self._new_model = self._create_new_model()
        return self._new_model
    except Exception as e:
        logger.error(f"Failed to initialize new model: {e}")
        return None

# 2. Add model creation method
def _create_new_model(self) -> Optional[object]:
    """Create new model instance"""
    from some_library import NewModel
    
    model = NewModel.from_pretrained(
        "model/name",
        device_map="cuda"
    )
    return model

# 3. Update public API
def get_new_model() -> Optional[object]:
    """Public API for new model"""
    return _video_model_manager.get_new_model()
```

### Testing Strategy

#### Unit Tests
```python
# test/unit_test/test_video_model_manager.py
def test_video_model_manager_initialization():
    """Test model manager initialization"""
    manager = VideoModelManager()
    assert not manager.is_ready()  # Before initialization
    
def test_whisper_model_loading():
    """Test Whisper model loading with mocks"""
    with patch('faster_whisper.WhisperModel') as mock_whisper:
        model = get_whisper_model()
        assert model is not None
```

#### Integration Tests
```python
# test/integration_tests/test_video_ingestion_task.py
def test_video_processing_pipeline():
    """Test complete video processing pipeline"""
    result = video_ingestion_task.apply(
        args=[file_id, file_path, ingestion_id, dataset_id, user_id]
    )
    assert result.successful()
    assert result.result['success'] is True
```

#### API Tests
```python
# test/api/v2/test_video_ingestion.py
def test_video_file_upload():
    """Test video file upload through API"""
    with open("test_video.mp4", "rb") as video_file:
        response = client.post(
            "/api/v2/files/upload",
            files={"file": video_file}
        )
    assert response.status_code == 200
```

## Troubleshooting

### Common Issues

#### 1. CUDA/cuDNN Issues
**Problem**: `libcudnn_ops_infer.so.8: cannot open shared object file`

**Solution**:
```bash
# Check container library paths
docker exec celery_video_ingestion_worker ls -la /usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/

# Verify LD_LIBRARY_PATH
docker exec celery_video_ingestion_worker echo $LD_LIBRARY_PATH

# Rebuild container with proper CUDA version
export CUDA_VERSION=12.6
make run-dev-build-video
```

#### 2. Model Loading Failures
**Problem**: Models fail to download or load at runtime

**Root Causes & Solutions**:

**A. Build Environment Detection**
```bash
# Check if build detected GPU correctly
docker logs celery_video_ingestion_worker | grep -E "(GPU detected|No GPU detected)"

# For local builds with GPU:
# Expected: "GPU detected - proceeding with model downloads..."

# For CI/CD builds without GPU:
# Expected: "No GPU detected (likely CI environment) - skipping model downloads"
```

**B. CUDA/bitsandbytes Issues**
```bash
# Problem: bitsandbytes fails at runtime
# Check CUDA availability in container
docker exec celery_video_ingestion_worker nvidia-smi

# Check if models are using CPU fallback
docker exec celery_video_ingestion_worker python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
"

# Solution: Ensure GPU is properly mounted at runtime
docker-compose down
docker-compose --profile video-ingestion up
```

**C. Network/Download Issues**
```bash
# Check internet connectivity from container
docker exec celery_video_ingestion_worker curl -I https://huggingface.co

# Check disk space for model downloads
docker exec celery_video_ingestion_worker df -h

# Clear corrupted model cache
docker exec celery_video_ingestion_worker rm -rf ~/.cache/huggingface
```

**D. Model Status Verification**
```bash
# Check model manager status
docker exec celery_video_ingestion_worker python -c "
from app.utils.video_model_manager import get_video_models_status
print(get_video_models_status())
"

# Force model re-download
docker exec celery_video_ingestion_worker python -c "
from app.utils.video_model_manager import VideoModelManager
manager = VideoModelManager()
success = manager.initialize_models_at_startup(force_download=True)
print(f'Models initialized: {success}')
"
```

#### 3. GPU Memory Issues
**Problem**: CUDA out of memory errors

**Solution**:
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch sizes in configuration
export VIDEO_CAPTIONING_BATCH_SIZE=1
export VIDEO_TRANSCRIPTION_MAX_WORKERS=1

# Enable GPU memory cleanup
export VIDEO_CLEANUP_GPU_MEMORY=true
```

#### 4. Processing Timeouts
**Problem**: Videos fail with timeout errors

**Solution**:
```bash
# Increase timeout settings
export DOCUMENT_PROCESSING_TIMEOUT_SECONDS=14400  # 4 hours

# Check worker logs
docker logs celery_video_ingestion_worker

# Reduce video segment length for faster processing
export VIDEO_SEGMENT_LENGTH=15
```

#### 5. Video Segment Cleanup Issues
**Problem**: Segments not being cleaned up or cleaned up incorrectly

**Solution**:
```bash
# Check segment reference status
docker exec celery_video_ingestion_worker python -c "
from app.utils.video_cleanup_utils import check_video_segments_still_referenced
import asyncio
from uuid import UUID
result = asyncio.run(check_video_segments_still_referenced(
    UUID('workspace-id'), UUID('file-id')
))
print(f'Still referenced: {result}')
"

# Manual cleanup of orphaned segments
curl -X POST http://localhost:8000/api/v1/workspace/{workspace_id}/video-segments/cleanup-orphaned \
  -H "Authorization: Bearer {admin_token}"

# Check cleanup logic for specific file
docker exec celery_video_ingestion_worker python -c "
from app.utils.video_cleanup_utils import _should_delete_video_segments
import asyncio
from uuid import UUID
result = asyncio.run(_should_delete_video_segments(
    UUID('workspace-id'), UUID('file-id')
))
print(f'Should delete: {result}')
"

# Clear reference cache if needed
docker exec celery_video_ingestion_worker python -c "
from app.utils.video_cleanup_utils import clear_reference_cache
clear_reference_cache()
print('Reference cache cleared')
"
```

### Debugging Tools

#### 1. Model Status Check
```python
from app.utils.video_model_manager import get_video_models_status
status = get_video_models_status()
print(f"Ready: {status['ready']}")
print(f"Models: {status['downloaded_models']}")
print(f"CUDA: {status['cuda_available']}")
```

#### 2. Processing Status Monitor
```python
# Check Redis locks
import redis
r = redis.Redis(host='redis', port=6379)
keys = r.keys('processing:video:*')
for key in keys:
    print(f"{key}: {r.get(key)}")
```

#### 3. Database Queries
```sql
-- Check video documents
SELECT id, file_id, processing_status, error_message 
FROM documents 
WHERE document_type = 'Video' 
ORDER BY created_at DESC;

-- Check video chunks
SELECT d.id as doc_id, COUNT(dc.id) as chunk_count
FROM documents d
LEFT JOIN document_chunks dc ON d.id = dc.document_id
WHERE d.document_type = 'Video'
GROUP BY d.id;

-- Check video segment references
SELECT 
    f.id as file_id,
    f.filename,
    f.deleted_at,
    COUNT(dc.id) as active_references
FROM files f
LEFT JOIN documents d ON f.id = d.file_id AND d.deleted_at IS NULL
LEFT JOIN document_chunks dc ON d.id = dc.document_id 
    AND dc.chunk_metadata->>'file_id' = f.id::text
WHERE f.mimetype LIKE 'video/%' OR f.filename ~* '\.(mp4|avi|mov|wmv|flv|webm|mkv)$'
GROUP BY f.id, f.filename, f.deleted_at
ORDER BY f.created_at DESC;
```

#### 4. Video Cleanup Status Check
```python
# Check video segments directory structure
import os
from app.be_core.config import settings

def check_workspace_segments(workspace_id: str):
    """Check video segments for a workspace"""
    workspace_dir = os.path.join(settings.VIDEO_SEGMENTS_DIR, workspace_id)
    if not os.path.exists(workspace_dir):
        print(f"No segments directory for workspace {workspace_id}")
        return
    
    file_dirs = os.listdir(workspace_dir)
    print(f"Found {len(file_dirs)} file directories:")
    
    for file_id in file_dirs:
        file_dir = os.path.join(workspace_dir, file_id)
        if os.path.isdir(file_dir):
            files = os.listdir(file_dir)
            total_size = sum(
                os.path.getsize(os.path.join(file_dir, f)) 
                for f in files if os.path.isfile(os.path.join(file_dir, f))
            )
            print(f"  {file_id}: {len(files)} files, {total_size/1024/1024:.1f} MB")

# Usage
check_workspace_segments("your-workspace-id")
```

## Performance Optimization

### GPU Optimization

#### 1. Memory Management
```python
# Implement proper cleanup
def cleanup_after_processing():
    cleanup_video_gpu_memory()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

#### 2. Batch Processing
```python
# Optimize batch sizes based on GPU memory
VIDEO_CAPTIONING_BATCH_SIZE = 3  # For 8GB GPU
VIDEO_CAPTIONING_BATCH_SIZE = 5  # For 16GB GPU
VIDEO_CAPTIONING_BATCH_SIZE = 8  # For 24GB+ GPU
```

#### 3. Model Optimization
```python
# Use quantized models for memory efficiency
VIDEO_CAPTION_MODEL = "openbmb/MiniCPM-V-2_6-int4"  # 4-bit quantization
```

### Processing Optimization

#### 1. Parallel Processing
```python
# Optimize worker counts based on system resources
VIDEO_TRANSCRIPTION_MAX_WORKERS = min(cpu_count(), 5)
```

#### 2. Segment Length Tuning
```python
# Balance between processing speed and accuracy
VIDEO_SEGMENT_LENGTH = 30  # Standard
VIDEO_SEGMENT_LENGTH = 15  # Faster processing, more segments
VIDEO_SEGMENT_LENGTH = 60  # Slower processing, fewer segments
```

#### 3. Storage Optimization
```python
# Enable original video deletion to save space
VIDEO_DELETE_ORIGINAL_ENABLED = true

# Disable segment cleanup to reuse processed segments
VIDEO_CLEANUP_ENABLED = false
```

#### 4. Video Cleanup Optimization
```python
# Optimize reference checking performance
from app.utils.video_cleanup_utils import clear_reference_cache

# Clear cache periodically to prevent memory buildup
clear_reference_cache()

# Batch cleanup operations during maintenance windows
async def maintenance_cleanup(workspace_ids: List[UUID]):
    """Batch cleanup multiple workspaces efficiently"""
    for workspace_id in workspace_ids:
        try:
            cleaned = await cleanup_orphaned_video_segments(workspace_id)
            logger.info(f"Workspace {workspace_id}: {cleaned} segments cleaned")
        except Exception as e:
            logger.error(f"Cleanup failed for {workspace_id}: {e}")

# Monitor cleanup performance
import time
start_time = time.time()
cleaned_count = await cleanup_orphaned_video_segments(workspace_id)
cleanup_time = time.time() - start_time
logger.info(f"Cleaned {cleaned_count} segments in {cleanup_time:.2f}s")
```

### Monitoring and Metrics

#### 1. Processing Metrics
```python
# Track processing times
processing_start = time.time()
# ... processing ...
processing_time = time.time() - processing_start
logger.info(f"Video processed in {processing_time:.2f} seconds")
```

#### 2. Resource Monitoring
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Container resource usage
docker stats celery_video_ingestion_worker

# Disk usage monitoring
watch -n 5 'df -h | grep uploads'
```

#### 3. Queue Monitoring
```bash
# RabbitMQ management
docker exec rabbitmq_dev rabbitmqctl list_queues name messages

# Celery monitoring
docker exec celery_video_ingestion_worker celery -A app.be_core.celery_video inspect active
```

This technical guide provides comprehensive information for developers working with the video ingestion system, from basic setup to advanced optimization and troubleshooting.
