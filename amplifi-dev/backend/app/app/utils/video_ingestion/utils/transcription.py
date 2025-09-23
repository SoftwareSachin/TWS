"""
Speech-to-text transcription utilities
"""

import os

from tqdm import tqdm

from app.be_core.logger import logger


def speech_to_text(
    video_name: str,
    working_dir: str,
    segment_index2name: dict,
    audio_output_format: str,
    whisper_model=None,
    max_workers: int = 4,  # Number of parallel transcription workers
):
    """Transcribe audio segments to text using Whisper on GPU with parallel processing"""
    # Check if dependencies are available
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("CUDA not available - cannot perform GPU transcription")
            return dict.fromkeys(segment_index2name, "")
    except ImportError:
        logger.info("PyTorch not available - video ingestion disabled")
        return dict.fromkeys(segment_index2name, "")

    # Use provided model or return empty transcripts if not available
    if whisper_model is None:
        logger.warning("No Whisper model provided - returning empty transcripts")
        return dict.fromkeys(segment_index2name, "")

    model = whisper_model
    logger.info(
        f"Using pre-loaded Whisper model for parallel transcription with {max_workers} workers"
    )

    cache_path = os.path.join(working_dir, "_cache", video_name)

    # Prepare all audio files upfront
    audio_tasks = []
    for index in segment_index2name:
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        if os.path.exists(audio_file):
            audio_tasks.append((index, audio_file))

    total_tasks = len(audio_tasks)
    logger.info(f"Processing {total_tasks} audio segments in parallel")

    # Process all segments in parallel
    transcripts = {}

    if total_tasks == 0:
        # No valid audio files
        return dict.fromkeys(segment_index2name, "")

    try:
        # Use ThreadPoolExecutor for true parallel processing
        import concurrent.futures

        def transcribe_segment_task(task_data):
            """Process a single transcription task"""
            index, audio_file = task_data
            try:
                segments, info = model.transcribe(audio_file)
                result = ""
                for segment in segments:
                    result += (
                        f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                    )
                return index, result.strip()
            except Exception as e:
                logger.warning(
                    f"Failed to transcribe segment {index} ({audio_file}): {str(e)}"
                )
                return index, ""

        # Process all segments in parallel with progress bar
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(transcribe_segment_task, task): task[0]
                for task in audio_tasks
            }

            # Collect results with progress tracking
            completed = 0
            with tqdm(
                total=total_tasks, desc=f"Speech Recognition {video_name} (Parallel)"
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        index, transcript = future.result()
                        transcripts[index] = transcript
                        completed += 1
                        pbar.update(1)
                    except Exception as e:
                        index = future_to_index[future]
                        logger.error(f"Task failed for segment {index}: {str(e)}")
                        transcripts[index] = ""
                        completed += 1
                        pbar.update(1)

    except Exception as e:
        logger.error(f"Error in parallel transcription: {str(e)}")
        # Fallback to sequential processing
        logger.info("Falling back to sequential transcription processing")
        for index, audio_file in tqdm(
            audio_tasks, desc=f"Speech Recognition {video_name} (Sequential Fallback)"
        ):
            transcripts[index] = _transcribe_single(model, audio_file)

    # Fill in empty transcripts for segments without audio files
    for index in segment_index2name:
        if index not in transcripts:
            transcripts[index] = ""

    logger.info(f"Completed transcription for {len(transcripts)} segments")
    return transcripts


def _transcribe_batch(model, audio_files):
    """Process a batch of audio files for transcription (legacy function for compatibility)"""
    results = []
    for audio_file in audio_files:
        results.append(_transcribe_single(model, audio_file))
    return results


def _transcribe_single(model, audio_file):
    """Transcribe a single audio file (fallback method)"""
    try:
        segments, info = model.transcribe(audio_file)
        result = ""
        for segment in segments:
            result += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
        return result.strip()
    except Exception as e:
        logger.warning(f"Failed to transcribe {audio_file}: {str(e)}")
        return ""
