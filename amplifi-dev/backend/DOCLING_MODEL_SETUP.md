# Docling Model Download Setup

This document explains how to configure and use automatic docling model downloading when your FastAPI backend starts.

## Overview

The backend now includes automatic docling model downloading at startup to ensure models are available before processing PDF documents. This eliminates the delay that occurs when models are downloaded on first use.

## Configuration

### Environment Variables

You can control the model downloading behavior using these environment variables:

```bash
# Enable/disable model download at startup (default: true)
DOCLING_DOWNLOAD_MODELS_AT_STARTUP=true

# Force re-download even if models exist (default: false)
DOCLING_FORCE_DOWNLOAD=false

# Timeout for model download in seconds (default: 300)
DOCLING_MODEL_DOWNLOAD_TIMEOUT=300
```

### Settings in config.py

The configuration is defined in `app/be_core/config.py`:

```python
# Docling Model Settings
DOCLING_DOWNLOAD_MODELS_AT_STARTUP: bool = Field(
    default=True, env="DOCLING_DOWNLOAD_MODELS_AT_STARTUP"
)
DOCLING_FORCE_DOWNLOAD: bool = Field(
    default=False, env="DOCLING_FORCE_DOWNLOAD"
)
DOCLING_MODEL_DOWNLOAD_TIMEOUT: int = Field(
    default=300, env="DOCLING_MODEL_DOWNLOAD_TIMEOUT"
)
```

## How It Works

### Startup Process

1. When the FastAPI application starts, the `lifespan` function in `main.py` is called
2. If `DOCLING_DOWNLOAD_MODELS_AT_STARTUP` is enabled, the system attempts to download docling models
3. The models are downloaded and a `DocumentConverter` instance is initialized
4. The converter is stored for reuse in subsequent PDF processing operations

### Model Detection

The system checks for existing models in these locations:
- `~/.cache/docling`
- `~/.cache/huggingface`
- `/tmp/docling_cache`

If models are found and `DOCLING_FORCE_DOWNLOAD` is false, the download is skipped.

### PDF Processing

When processing PDFs, the system:
1. First checks if a pre-initialized converter is available
2. If available, uses the existing converter (faster)
3. If not available, creates a new converter (fallback behavior)

## Monitoring

### Health Check Endpoints

You can monitor the docling model status using these endpoints:

```bash
# Basic health check
GET /health

# Detailed health check including docling status
GET /health/detailed
```

The detailed health check returns:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "docling": {
    "ready": true,
    "downloaded_models": [],
    "converter_initialized": true
  },
  "environment": "local",
  "mode": "development"
}
```

### Logs

The system logs model download progress:

```
INFO: Starting docling model download at startup...
INFO: Initializing docling DocumentConverter to download models...
INFO: Testing docling converter initialization...
INFO: Docling models downloaded successfully in 45.23 seconds
INFO: Docling models initialized successfully
```

## Testing

### Test Script

You can test the model downloading functionality using the provided test script:

```bash
cd backend
python test_docling_models.py
```

This script will:
1. Check initial status
2. Attempt to download models
3. Verify the converter is available
4. Report success or failure

### Manual Testing

You can also test manually by:

1. Starting the FastAPI server
2. Checking the logs for model download messages
3. Calling the health endpoints
4. Processing a PDF to verify the converter works

## Troubleshooting

### Common Issues

1. **Models not downloading**: Check internet connectivity and firewall settings
2. **Download timeout**: Increase `DOCLING_MODEL_DOWNLOAD_TIMEOUT`
3. **Permission errors**: Ensure write access to cache directories
4. **Memory issues**: Models require significant memory; ensure adequate RAM

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.getLogger("app.utils.docling_model_manager").setLevel(logging.DEBUG)
```

### Force Re-download

To force re-download of models:

```bash
export DOCLING_FORCE_DOWNLOAD=true
# Restart your application
```

## Performance Considerations

### Startup Time

- Model download adds to startup time (typically 30-60 seconds)
- Subsequent starts are faster if models are cached
- Consider using `DOCLING_DOWNLOAD_MODELS_AT_STARTUP=false` in development

### Memory Usage

- Models are loaded into memory at startup
- Ensure adequate RAM for your deployment
- Monitor memory usage in production

### Caching

- Models are cached locally for reuse
- Cache location depends on your system configuration
- Clear cache if you encounter model-related issues

## Integration with Existing Code

The implementation is designed to be non-breaking:

- Existing PDF processing code continues to work unchanged
- Fallback to on-demand model creation if startup download fails
- No changes required to existing ingestion tasks or endpoints

## Docker/Kubernetes Considerations

### Volume Mounts

For persistent model storage in containers, mount the cache directory:

```yaml
volumes:
  - name: docling-cache
    persistentVolumeClaim:
      claimName: docling-cache-pvc
volumeMounts:
  - name: docling-cache
    mountPath: /root/.cache/docling
```

### Resource Limits

Ensure adequate resources for model loading:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "1000m"
```

### Init Containers

Consider using an init container for model download:

```yaml
initContainers:
  - name: download-docling-models
    image: your-app-image
    command: ["python", "test_docling_models.py"]
    volumeMounts:
      - name: docling-cache
        mountPath: /root/.cache/docling
``` 