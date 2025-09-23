import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import azure.cognitiveservices.speech as speechsdk
from pydub.utils import mediainfo
from sqlalchemy.orm import Session

from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType
from app.models.file_model import File
from app.utils.ingestion_utils import publish_ingestion_status
from app.utils.openai_utils import generate_embedding, get_openai_client


class BinaryFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
    """Callback class to handle audio stream reading"""

    def __init__(self, filename: str):
        super().__init__()
        self._file_h = open(filename, "rb")
        logger.info(f"Initialized BinaryFileReaderCallback for file: {filename}")

    def read(self, buffer: memoryview) -> int:
        try:
            size = buffer.nbytes
            frames = self._file_h.read(size)
            buffer[: len(frames)] = frames
            return len(frames)
        except Exception as ex:
            logger.error(f"Exception in audio stream read: {ex}")
            raise

    def close(self) -> None:
        try:
            self._file_h.close()
            logger.info("Closed BinaryFileReaderCallback for file.")
        except Exception as ex:
            logger.error(f"Exception in audio stream close: {ex}")
            raise


def _get_mime_type(file_path: str) -> str:
    """Determine MIME type from file extension"""
    logger.debug(f"Determining MIME type for file: {file_path}")
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".wav":
        return "audio/wav"
    elif extension == ".mp3":
        return "audio/mpeg"
    elif extension == ".aac":
        return "audio/aac"
    else:
        return "application/octet-stream"


def _get_speech_config() -> speechsdk.SpeechConfig:
    """Get Azure Speech Config"""
    logger.info("Fetching Azure Speech configuration.")
    return speechsdk.SpeechConfig(
        subscription=settings.AZURE_SPEECH_KEY, region=settings.AZURE_SPEECH_REGION
    )


def _update_file_ingestion_status(
    db: Session,
    file_id: str,
    ingestion_id: str,
    status: FileIngestionStatusType,
    task_id: Optional[str] = None,
    error_message: Optional[str] = None,
):
    """Update file ingestion status in database"""
    logger.info(f"Updating ingestion status for file ID: {file_id}, Status: {status}")
    ingestion = (
        db.query(FileIngestion)
        .filter(
            FileIngestion.file_id == file_id, FileIngestion.ingestion_id == ingestion_id
        )
        .first()
    )

    if ingestion:
        ingestion.status = status
        if task_id:
            ingestion.task_id = task_id
        if error_message:
            logger.error(f"Ingestion error for file {file_id}: {error_message}")

        ingestion.updated_at = datetime.utcnow()
        if status in [
            FileIngestionStatusType.Success,
            FileIngestionStatusType.Failed,
            FileIngestionStatusType.Exception,
        ]:
            ingestion.finished_at = datetime.utcnow()
        db.commit()
        logger.info(f"Ingestion status updated for file ID: {file_id}")


def _extract_audio_metadata(audio_file_path: str) -> Dict[str, Any]:
    """Extract metadata from audio file"""
    logger.info(f"Extracting metadata for audio file: {audio_file_path}")
    try:
        info = mediainfo(audio_file_path)
        logger.debug(f"Audio metadata: {info}")
        return {
            "duration": float(info.get("duration", 0.0)),
            "sample_rate": int(info.get("sample_rate", 0)),
            "bitrate": int(info.get("bit_rate", 0)),
            "channels": int(info.get("channels", 0)),
            "format": info.get("format_name", "unknown"),
        }
    except Exception as e:
        logger.error(f"Error extracting audio metadata: {str(e)}")
        return {}


def _transcribe_audio(
    speech_config: speechsdk.SpeechConfig, audio_file_path: str
) -> str:
    """Transcribe audio file using Azure Speech Services"""
    logger.info(f"Transcribing audio file: {audio_file_path}")

    file_extension = os.path.splitext(audio_file_path)[1].lower()

    try:
        if file_extension == ".wav":
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )
        elif file_extension in [".mp3", ".aac"]:
            callback = BinaryFileReaderCallback(audio_file_path)

            compressed_format = speechsdk.audio.AudioStreamFormat(
                compressed_stream_format=(
                    speechsdk.AudioStreamContainerFormat.MP3
                    if file_extension == ".mp3"
                    else speechsdk.AudioStreamContainerFormat.ANY
                )
            )

            stream = speechsdk.audio.PullAudioInputStream(
                stream_format=compressed_format, pull_stream_callback=callback
            )

            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported formats are .wav, .mp3, and .aac"
            )

        all_results = []
        done = False

        def handle_result(evt):
            all_results.append(evt.result.text)

        def handle_stop(evt):
            nonlocal done
            done = True

        recognizer.recognized.connect(handle_result)
        recognizer.session_stopped.connect(handle_stop)
        recognizer.canceled.connect(handle_stop)

        recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.5)
        recognizer.stop_continuous_recognition()

        logger.info(f"Transcription completed for file: {audio_file_path}")
        transciption = " ".join(all_results)

        if not transciption.strip():
            logger.debug(transciption)
            raise ValueError("Transcription resulted in empty text")

        return transciption

    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise


def _chunk_text(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[Tuple[str, int, int]]:
    """Split text into chunks with overlap"""
    logger.info(
        f"Chunking text into chunks of size {chunk_size} with overlap {chunk_overlap}"
    )
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append((chunk, start, end))
        start += chunk_size - chunk_overlap

    logger.info(f"Generated {len(chunks)} chunks from text.")
    return chunks


def _extract_speaker_names(transcription: str) -> List[str]:
    """Extract speaker names from transcription using GPT-4"""
    logger.info("Extracting speaker names from transcription.")
    try:
        client = get_openai_client()
        prompt = f"""Extract unique speaker names from the following transcription.Respond in a JSON array of strings like: ["Dr. Smith", "Patient", "Nurse"]
        Transcription:
        {transcription}
        """

        response = client.chat.completions.create(
            model=settings.AZURE_GPT_4o_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts speaker names.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        # Remove markdown code blocks
        import re

        content = re.sub(
            r"^```(?:json)?\s*([\s\S]+?)\s*```$", r"\1", content, flags=re.MULTILINE
        )

        try:
            import json

            speaker_names = json.loads(content)
            return speaker_names
        except json.JSONDecodeError:
            try:
                import ast

                speaker_names = ast.literal_eval(content)
                return speaker_names
            except Exception as e:
                logger.error(f"Speaker name extraction failed: {e}")
                logger.error(f"Raw cleaned response was: {repr(content)}")
                return []

    except Exception as e:
        logger.error(f"Error extracting speaker names: {str(e)}")
        return []


def _create_document_record(
    db: Session,
    document_id: str,
    file_id: str,
    dataset_id: str,
    file_path: str,
    metadata: dict,
    task_id: str,
    ingestion_id: str,
) -> Document:
    """Create and save initial document record to database"""
    logger.info(f"Creating document record with ID: {document_id}")

    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
    mime_type = _get_mime_type(file_path)
    logger.debug(f"File size: {file_size}, MIME type: {mime_type}")

    document = Document(
        id=document_id,
        file_id=file_id,
        dataset_id=dataset_id,
        document_type=DocumentTypeEnum.Audio,
        processing_status=DocumentProcessingStatusEnum.Processing,
        file_path=file_path,
        file_size=file_size,
        mime_type=mime_type,
        document_metadata=metadata or {},
        task_id=task_id,
        ingestion_id=ingestion_id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(f"Created document record with ID: {document_id}")
    return document


def _update_document_metadata(
    db: Session, document: Document, file_path: str
) -> Document:
    """Extract and update document with audio metadata"""
    logger.info("Updating document with audio metadata")

    audio_metadata = _extract_audio_metadata(file_path)
    logger.debug(f"Extracted audio metadata: {audio_metadata}")

    merged_metadata = {
        **(document.document_metadata or {}),
        **audio_metadata,
    }
    document.document_metadata = merged_metadata
    document = db.merge(document)
    db.commit()
    db.refresh(document)
    logger.info("Updated document metadata in the database.")
    return document


def _process_transcription(
    db: Session, document: Document, file_path: str
) -> Tuple[str, List[float]]:
    """Transcribe audio and generate embedding"""
    logger.info("Processing audio transcription")

    document.processing_status = DocumentProcessingStatusEnum.Extracting
    db.commit()
    logger.info(f"Set document status to Extracting for document ID: {document.id}")

    speech_config = _get_speech_config()
    logger.info("Retrieved Azure Speech configuration.")

    transcription = _transcribe_audio(speech_config, file_path)
    logger.info("Transcribed audio file.")
    logger.debug(f"Transcription: {transcription}")

    if not transcription.strip():
        raise ValueError("Empty transcription result")

    try:
        transcription_embedding = generate_embedding(transcription)
        if len(transcription_embedding) != settings.EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Generated embedding has {len(transcription_embedding)} dimensions, "
                f"expected {settings.EMBEDDING_DIMENSIONS}"
            )
        logger.info("Generated embedding for the transcription.")
        return transcription, transcription_embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise ValueError(f"Embedding generation failed: {str(e)}")


def _create_transcription_chunks(
    db: Session,
    document_id: str,
    transcription: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Create and save transcription chunks to database"""
    logger.info("Creating transcription chunks")

    chunk_ids = []
    chunks = _chunk_text(transcription, chunk_size, chunk_overlap)
    logger.debug(f"Generated {len(chunks)} chunks from transcription.")

    for i, (chunk_text, start_idx, end_idx) in enumerate(chunks):
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_type=ChunkTypeEnum.AudioSegment,
            chunk_text=chunk_text,
            chunk_embedding=generate_embedding(chunk_text),
            chunk_metadata={
                "chunk_order": i,
                "start_index": start_idx,
                "end_index": end_idx,
                "chunked_by_engine": "CustomChunkerV1",
            },
        )
        db.add(chunk)
        chunk_ids.append(chunk.id)

    logger.info(f"Processed {len(chunk_ids)} transcription chunks.")
    return chunk_ids


def _create_speaker_chunks(
    db: Session, document_id: str, transcription: str
) -> List[str]:
    """Create and save speaker chunks to database"""
    logger.info("Creating speaker chunks")

    chunk_ids = []
    speaker_names = _extract_speaker_names(transcription)
    logger.debug(f"Extracted speaker names: {speaker_names}")

    if speaker_names:
        combined_speakers = "; ".join(speaker_names)
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_type=ChunkTypeEnum.Speaker,
            chunk_text=combined_speakers,
            chunk_embedding=generate_embedding(combined_speakers),
            chunk_metadata={
                "speaker_id": 0,
                "chunked_by_engine": "GPT4SpeakerExtractor",
            },
        )
        db.add(chunk)
        chunk_ids.append(chunk.id)
        logger.info("Processed 1 speaker chunk with all names.")
    else:
        logger.info("No speaker names extracted, so no chunks created.")

    return chunk_ids


def _finalize_document_success(
    db: Session,
    document: Document,
    transcription: str,
    transcription_embedding: List[float],
) -> None:
    """Update document with final transcription and mark as successful"""
    logger.info("Finalizing document as successful")

    try:
        document.description = transcription
        document.description_embedding = transcription_embedding
        document.processing_status = DocumentProcessingStatusEnum.ExtractionCompleted
        db.commit()
        logger.info("Updated document with transcription and completed status.")
    except Exception as e:
        db.rollback()
        logger.error(f"failed to update document: {str(e)}")
        raise ValueError(f"Document update failed: {str(e)}")

    document.processing_status = DocumentProcessingStatusEnum.Success
    document.processed_at = datetime.utcnow()
    db.commit()
    logger.info(f"Marked document {document.id} as successfully processed.")


def _handle_success_notification(
    db: Session,
    file_id: str,
    file_path: str,
    document_id: str,
    chunk_ids: List[str],
    ingestion_id: str,
    user_id: uuid,
    task_id: str,
) -> None:
    """Send success notification to user"""
    logger.info("Handling success notification")

    _update_file_ingestion_status(
        db=db,
        file_id=file_id,
        ingestion_id=ingestion_id,
        status=FileIngestionStatusType.Success,
    )
    logger.info(f"Set file ingestion status to Success for file ID: {file_id}")

    file_info = db.query(File).filter(File.id == uuid.UUID(file_id)).first()
    file_name = file_info.filename if file_info else Path(file_path).name
    logger.debug(f"File name for notification: {file_name}")

    if user_id:
        result_data = {
            "file_id": file_id,
            "file_name": file_name,
            "document_id": document_id,
            "status": FileIngestionStatusType.Success.value,
            "success": True,
            "chunk_count": len(chunk_ids),
            "document_type": DocumentTypeEnum.Audio.value,
            "ingestion_id": ingestion_id,
            "finished_at": datetime.utcnow().isoformat(),
            "task_id": task_id,
        }
        publish_ingestion_status(
            user_id=user_id,
            ingestion_id=ingestion_id,
            task_id=task_id,
            ingestion_result=result_data,
        )
        logger.info("Sent success notification via WebSocket.")


def _handle_error_notification(
    db: Session,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    user_id: uuid,
    task_id: str,
    error: Exception,
    document_id: Optional[str] = None,
) -> None:
    """Handle error notification and database updates"""
    logger.info("Handling error notification")

    if document_id:
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_status = DocumentProcessingStatusEnum.Failed
                document.error_message = str(error)
                document.processed_at = datetime.utcnow()
                db.commit()
                logger.info(f"Marked document {document_id} as Failed.")
        except Exception as e:
            logger.error(f"Failed to update document error status: {e}")

    _update_file_ingestion_status(
        db=db,
        file_id=file_id,
        ingestion_id=ingestion_id,
        status=FileIngestionStatusType.Failed,
        error_message=str(error),
    )
    logger.info(f"Set file ingestion status to Failed for file ID: {file_id}")

    file_info = db.query(File).filter(File.id == uuid.UUID(file_id)).first()
    file_name = file_info.filename if file_info else Path(file_path).name
    logger.debug(f"File name for failure notification: {file_name}")

    if user_id:
        error_data = {
            "file_id": file_id,
            "file_name": file_name,
            "status": FileIngestionStatusType.Failed.value,
            "success": False,
            "error": str(error),
            "ingestion_id": ingestion_id,
            "finished_at": datetime.utcnow().isoformat(),
            "task_id": task_id,
        }
        publish_ingestion_status(
            user_id=user_id,
            ingestion_id=ingestion_id,
            task_id=task_id,
            ingestion_result=error_data,
        )
        logger.info("Sent failure notification via WebSocket.")


@celery.task(name="tasks.audio_ingestion_task", bind=True, acks_late=True)
def audio_ingestion_task(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: uuid,
    metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Any]:
    """
    Process an audio file, transcribe it, and extract metadata.

    Args:
        file_id: ID of the file being processed
        file_path: Path to the audio file
        ingestion_id: ID of the current ingestion batch
        dataset_id: ID of the dataset
        metadata: Additional metadata to store with the document
        user_id: ID of the user who initiated the ingestion
        chunk_size: Size of text chunks (in words)
        chunk_overlap: Number of overlapping words between chunks

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting audio ingestion task for file: {file_path}")
    logger.debug(
        f"File ID: {file_id}, Ingestion ID: {ingestion_id}, Dataset ID: {dataset_id}"
    )

    with SyncSessionLocal() as db:
        _update_file_ingestion_status(
            db=db,
            file_id=file_id,
            ingestion_id=ingestion_id,
            status=FileIngestionStatusType.Processing,
            task_id=self.request.id,
        )
        logger.info(f"Set file ingestion status to Processing for file ID: {file_id}")

        try:
            document_id = str(uuid.uuid4())

            document = _create_document_record(
                db=db,
                document_id=document_id,
                file_id=file_id,
                dataset_id=dataset_id,
                file_path=file_path,
                metadata=metadata,
                task_id=self.request.id,
                ingestion_id=ingestion_id,
            )

            document = _update_document_metadata(
                db=db, document=document, file_path=file_path
            )

            transcription, transcription_embedding = _process_transcription(
                db=db, document=document, file_path=file_path
            )

            chunk_ids = _create_transcription_chunks(
                db=db,
                document_id=document_id,
                transcription=transcription,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            chunk_ids += _create_speaker_chunks(
                db=db, document_id=document_id, transcription=transcription
            )

            _finalize_document_success(
                db=db,
                document=document,
                transcription=transcription,
                transcription_embedding=transcription_embedding,
            )

            _handle_success_notification(
                db=db,
                file_id=file_id,
                file_path=file_path,
                document_id=document_id,
                chunk_ids=chunk_ids,
                ingestion_id=ingestion_id,
                user_id=user_id,
                task_id=self.request.id,
            )

            return {
                "success": True,
                "document_id": document_id,
                "file_id": file_id,
                "dataset_id": dataset_id,
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
            }

        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}", exc_info=True)

            # Handle error notification and database updates
            _handle_error_notification(
                db=db,
                file_id=file_id,
                file_path=file_path,
                ingestion_id=ingestion_id,
                user_id=user_id,
                task_id=self.request.id,
                error=e,
                document_id=document_id if "document_id" in locals() else None,
            )

            return {"success": False, "file_id": file_id, "error": str(e)}
