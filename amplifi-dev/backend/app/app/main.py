import gc
import secrets
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi_async_sqlalchemy import SQLAlchemyMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi_pagination.utils import disable_installed_extensions_check
from jwt import DecodeError, ExpiredSignatureError, MissingRequiredClaimError
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool
from starlette.middleware.cors import CORSMiddleware

from app.api.deps import close_async_redis_client, get_redis_client
from app.api.v1.endpoints import websocket
from app.api.v2.api import api_router as api_router_v2
from app.be_core.config import ModeEnum, settings
from app.be_core.logger import logger
from app.be_core.security import decode_token
from app.utils.docling_model_manager import (
    get_docling_status,
    initialize_docling_models,
)
from app.utils.fastapi_globals import GlobalsMiddleware, g
from app.utils.feature_flags import is_video_ingestion_enabled
from app.utils.middleware.sanitize_headers import HeaderSanitizationMiddleware
from app.utils.startup_video_models import (
    check_video_models_health,
)

# To ignore this warning
# It's recommended to use extension "fastapi_pagination.ext.sqlmodel" instead of default 'paginate' implementation.
disable_installed_extensions_check()


def _get_user_id_from_token(token: str) -> str:
    try:
        payload = decode_token(token)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your token has expired. Please log in again.",
        )
    except DecodeError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Error when decoding the token. Please check your request.",
        )
    except MissingRequiredClaimError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="There is no required field in your token. Please contact the administrator.",
        )
    return payload["sub"]


async def user_id_identifier(request: Request):
    print(f"user_id_identifier: scope={request.scope}")
    if request.scope["type"] == "http" or request.scope["type"] == "ws":
        # Retrieve the Authorization header from the request
        auth_header = request.headers.get("Authorization")

        if auth_header is not None:
            # Check that the header is in the correct format
            header_parts = auth_header.split()
            if len(header_parts) == 2 and header_parts[0].lower() == "bearer":
                return _get_user_id_from_token(header_parts[1])

    if request.scope["type"] == "websocket":
        return request.scope["path"]

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]

    client = request.client
    # Ignore  [B104:hardcoded_bind_all_interfaces] Possible binding to all interfaces.
    # We are binding to all interfaces inside docker
    ip = getattr(client, "host", "0.0.0.0")  # nosec
    return ip + ":" + request.scope["path"]


async def ip_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    # X-Forwarded-For may contain a comma-separated list of IPs
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
    return f"ip:{ip}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    redis_client = await get_redis_client()
    FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
    if settings.DEPLOYED_ENV == "azure_dev" or settings.DEPLOYED_ENV == "azure_prod":
        await FastAPILimiter.init(redis_client, identifier=ip_identifier)
    else:
        await FastAPILimiter.init(redis_client)

    # Initialize docling models at startup
    if settings.DOCLING_DOWNLOAD_MODELS_AT_STARTUP:
        logger.info("Initializing docling models at startup")
        docling_success = initialize_docling_models(
            force_download=settings.DOCLING_FORCE_DOWNLOAD
        )
        if docling_success:
            logger.info("Docling models initialized successfully")
        else:
            logger.warning(
                "Docling models initialization failed, will initialize on first use"
            )
    else:
        logger.info(
            "Docling model download at startup disabled, will initialize on first use"
        )

    # Skip video model initialization in FastAPI server - handled by video ingestion worker
    if is_video_ingestion_enabled():
        logger.info(
            "Video ingestion enabled - video models will be initialized by video ingestion worker"
        )
    else:
        logger.info("Video ingestion disabled - skipping video model initialization")

    logger.info("startup fastapi")
    yield
    # shutdown
    await redis_client.close()
    await FastAPICache.clear()
    await FastAPILimiter.close()
    # models.clear()
    g.cleanup()
    gc.collect()


# Core Application Instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=f"{settings.API_V2_STR}/openapi.json",
    lifespan=lifespan,
)


app.add_middleware(
    SQLAlchemyMiddleware,
    db_url=str(settings.ASYNC_DATABASE_URI),
    engine_args={
        "echo": False,
        "poolclass": (
            NullPool if settings.MODE == ModeEnum.testing else AsyncAdaptedQueuePool
        ),
        # "pool_pre_ping": True,
        # "pool_size": settings.POOL_SIZE,
        # "max_overflow": 64,
    },
)
app.add_middleware(GlobalsMiddleware)
app.add_middleware(HeaderSanitizationMiddleware)
# Set all CORS origins enabled
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    if request.url.path == "/docs":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "style-src 'self' https://cdn.jsdelivr.net;"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' https://* data: blob:; "
            "media-src 'self' blob: data:; "
            "child-src 'none';"
        )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(self), microphone=(self)"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"

    return response


DOCS_USERNAME = settings.AMPLIFI_DOCS_USERNAME
DOCS_PASSWORD = settings.AMPLIFI_DOCS_PASSWORD

security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, DOCS_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, DOCS_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


# Override Swagger docs route with auth
@app.get("/docs", include_in_schema=False)
async def get_swagger_documentation(
    credentials: HTTPBasicCredentials = Depends(verify_credentials),
):
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V2_STR}/openapi.json", title="Docs"
    )


# Optional: override ReDoc too
@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation(
    credentials: HTTPBasicCredentials = Depends(verify_credentials),
):
    return get_redoc_html(
        openapi_url=f"{settings.API_V2_STR}/openapi.json", title="ReDoc"
    )


class CustomException(Exception):
    http_code: int
    code: str
    message: str

    def __init__(
        self,
        http_code: int = 500,
        code: str | None = None,
        message: str = "This is an error message",
    ):
        self.http_code = http_code
        self.code = code if code else str(self.http_code)
        self.message = message


@app.get("/")
async def root():
    """
    An example "Hello world" FastAPI route.
    """
    # if oso.is_allowed(user, "read", message):
    return {"message": "Hello World"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/health/detailed")
def detailed_health_check():
    """Detailed health check endpoint including docling and video model status."""
    docling_status = get_docling_status()
    video_status = check_video_models_health()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "docling": docling_status,
        "video_models": video_status,
        "environment": settings.DEPLOYED_ENV,
        "mode": settings.MODE.value,
    }


@app.on_event("shutdown")
async def shutdown_event():
    await close_async_redis_client()


# @app.websocket("/chat/{user_id}")
# async def websocket_endpoint(websocket: WebSocket, user_id: UUID):
#     session_id = str(uuid4())
#     key: str = f"user_id:{user_id}:session:{session_id}"
#     await websocket.accept()
#     redis_client = await get_redis_client()
#     ws_ratelimit = WebSocketRateLimiter(times=200, hours=24)
#     chat = ChatOpenAI(temperature=0, openai_api_key=settings.OPENAI_API_KEY)
#     chat_history = []
#
#     async with db():
#         user = await crud.user.get_by_id_active(id=user_id)
#         if user is not None:
#             await redis_client.set(key, str(websocket))
#
#     active_connection = await redis_client.get(key)
#     if active_connection is None:
#         await websocket.send_text(f"Error: User ID '{user_id}' not found or inactive.")
#         await websocket.close()
#     else:
#         while True:
#             try:
#                 # Receive and send back the client message
#                 data = await websocket.receive_json()
#                 await ws_ratelimit(websocket)
#                 user_message = IUserMessage.model_validate(data)
#                 user_message.user_id = user_id
#
#                 resp = IChatResponse(
#                     sender="you",
#                     message=user_message.message,
#                     type="stream",
#                     message_id=str(uuid7()),
#                     id=str(uuid7()),
#                 )
#                 await websocket.send_json(resp.dict())
#
#                 # # Construct a response
#                 start_resp = IChatResponse(
#                     sender="bot", message="", type="start", message_id="", id=""
#                 )
#                 await websocket.send_json(start_resp.dict())
#
#                 result = chat([HumanMessage(content=resp.message)])
#                 chat_history.append((user_message.message, result.content))
#
#                 end_resp = IChatResponse(
#                     sender="bot",
#                     message=result.content,
#                     type="end",
#                     message_id=str(uuid7()),
#                     id=str(uuid7()),
#                 )
#                 await websocket.send_json(end_resp.dict())
#             except WebSocketDisconnect:
#                 logging.info("websocket disconnect")
#                 break
#             except Exception as e:
#                 logging.error(e)
#                 resp = IChatResponse(
#                     message_id="",
#                     id="",
#                     sender="bot",
#                     message="Sorry, something went wrong. Your user limit of api usages has been reached or check your API key.",
#                     type="error",
#                 )
#                 await websocket.send_json(resp.dict())
#
#         # Remove the live connection from Redis
#         await redis_client.delete(key)

dependencies = []
# Issue with fastAPILimiter in pytests
# Reference - https://github.com/long2ice/fastapi-limiter/issues/51
if "pytest" not in sys.modules:
    dependencies.append(
        Depends(RateLimiter(times=settings.RATE_LIMIT_PER_SECOND, seconds=1))
    )

# Add Routers
# With ratelimiter

app.include_router(
    api_router_v2,
    prefix=settings.API_V2_STR,
    dependencies=dependencies,
)

app.include_router(websocket.router, prefix="/ws")
