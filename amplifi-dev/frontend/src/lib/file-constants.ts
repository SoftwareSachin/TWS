export const MAX_FILE_SIZE_UPLOADED_FILES = 300 * 1024 * 1024; // 300MB in bytes
export const MAX_IMG_SIZE_UPLOADED_FILES = 200 * 1024 * 1024; // 200MB in bytes
export const MAX_AUDIO_SIZE_UPLOADED_FILES = 30 * 1024 * 1024; // 30MB in bytes
export const MAX_VIDEO_SIZE_UPLOADED_FILES = 50 * 1024 * 1024; // 50MB in bytes
export const MAX_QUERY_LENGTH = 10000; // 10000 characters

export const ALLOWED_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"];
export const ALLOWED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".aac"];
export const ALLOWED_VIDEO_EXTENSIONS = [
  ".mp4",
  ".avi",
  ".mov",
  ".wmv",
  ".flv",
  ".webm",
  ".mkv",
];
export const ALLOWED_FILE_EXTENSIONS = [
  // ".doc",
  ".pptx",
  ".pdf",
  // ".txt",
  // ".json",
  ".html",
  ".xlsx",
  ".csv",
  ".md",
  // ".xml",
  ".docx",
  ...ALLOWED_IMG_EXTENSIONS,
  ...ALLOWED_AUDIO_EXTENSIONS,
  ...ALLOWED_VIDEO_EXTENSIONS,
];

export const ALLOWED_IMG_MIME_TYPES = ["image/png", "image/jpg", "image/jpeg"];

export const ALLOWED_AUDIO_MIME_TYPES = [
  "audio/wav",
  "audio/mpeg",
  "audio/aac",
  "audio/mp3",
  "audio/vnd.dlna.adts",
];

export const ALLOWED_VIDEO_MIME_TYPES = [
  "video/mp4",
  "video/x-msvideo", // AVI
  "video/quicktime", // MOV
  "video/x-ms-wmv", // WMV
  "video/x-flv", // FLV
  "video/webm", // WEBM
  "video/x-matroska", // MKV
];

export const ALLOWED_MIME_TYPES = [
  // "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "application/pdf",
  // "text/plain",
  // "application/json",
  "text/html",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "text/csv",
  "application/octet-stream",
  // "text/xml",
  ...ALLOWED_IMG_MIME_TYPES,
  ...ALLOWED_AUDIO_MIME_TYPES,
  ...ALLOWED_VIDEO_MIME_TYPES,
];

export const WORKSPACE_NAME_MAX_LENGTH = 25;
export const DECSRIPTION_MAX_LENGTH = 100;
export const ALLOWED_NAME_REGEX = /^[a-zA-Z][a-zA-Z0-9 _-]*$/;
export const RESERVED_WORDS = ["select", "workspace"];

export const AGENTIC_NAME_MAX_LENGTH = 30;
export const AGENTIC_DESCRIPTION_MAX_LENGTH = 150;
