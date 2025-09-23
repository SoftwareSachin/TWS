"""
Video captioning and segment processing utilities
"""

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from app.be_core.logger import logger


def encode_video_frames(video, frame_times):
    """Extract and encode video frames at specified times with optimized performance"""
    frames = []
    try:
        # Batch extract frames to reduce I/O overhead
        for t in frame_times:
            frame = video.get_frame(t)
            frames.append(frame)

        # Stack frames efficiently
        frames = np.stack(frames, axis=0)

        # Resize frames with optimized parameters (smaller size for faster processing)
        frames = [
            Image.fromarray(v.astype("uint8")).resize(
                (1280, 720), Image.Resampling.LANCZOS
            )
            for v in frames
        ]
        return frames
    except Exception as e:
        logger.warning(f"Frame encoding error: {e}")
        # Fallback to original size if resize fails
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        return frames


def segment_caption(
    video_name: str,
    video_path: str,
    segment_index2name: dict,
    transcripts: dict,
    segment_times_info: dict,
    caption_result: dict,
    error_queue,
    caption_model=None,
    caption_tokenizer=None,
    batch_size: int = 3,  # Process 3 segments at once for captioning
):
    """Generate captions for video segments"""
    try:
        # Check if CUDA is available
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError:
            # torch not available - video ingestion disabled
            for index in segment_index2name:
                caption_result[index] = ""
            return

        # Load model if not provided
        if caption_model is None or caption_tokenizer is None:
            try:
                logger.info(
                    f"Loading caption model... (CUDA available: {cuda_available})"
                )

                if not cuda_available:
                    logger.info(
                        "CUDA not available, skipping GPU-dependent captioning model"
                    )
                    # Provide empty captions for all segments when CUDA is not available
                    for index in segment_index2name:
                        caption_result[index] = ""
                    return

                # Load model with explicit processor configuration
                from transformers import AutoProcessor

                caption_model = AutoModel.from_pretrained(
                    "openbmb/MiniCPM-V-2_6-int4",
                    revision="06219bd",
                    trust_remote_code=True,
                    device_map="cuda",
                    torch_dtype="auto",
                    # Use standard cache location (~/.cache/huggingface)
                )
                caption_tokenizer = AutoTokenizer.from_pretrained(
                    "openbmb/MiniCPM-V-2_6-int4",
                    revision="06219bd",
                    trust_remote_code=True,
                    use_fast=True,
                    legacy=False,
                    # Use standard cache location (~/.cache/huggingface)
                )

                # Try to load processor with fast configuration
                try:
                    AutoProcessor.from_pretrained(
                        "openbmb/MiniCPM-V-2_6-int4",
                        revision="06219bd",
                        trust_remote_code=True,
                        use_fast=True,
                        # Use standard cache location (~/.cache/huggingface)
                    )
                    logger.info("Loaded fast image processor successfully")
                except Exception as e:
                    logger.warning(f"Could not load fast processor: {e}")
                caption_model.eval()
            except Exception as e:
                # If model loading fails, provide empty captions
                logger.error(f"Error loading caption model: {e}")
                for index in segment_index2name:
                    caption_result[index] = ""
                return

        with VideoFileClip(video_path) as video:
            # Prepare batch data
            segment_indices = list(segment_index2name.keys())
            total_segments = len(segment_indices)

            logger.info(
                f"Processing {total_segments} segments with batch size {batch_size}"
            )

            # Process segments in batches
            for batch_start in tqdm(
                range(0, total_segments, batch_size),
                desc=f"Captioning Video {video_name} (Batched)",
            ):
                batch_end = min(batch_start + batch_size, total_segments)
                batch_indices = segment_indices[batch_start:batch_end]

                try:
                    # Process batch of segments
                    batch_results = _caption_batch(
                        video,
                        batch_indices,
                        segment_index2name,
                        segment_times_info,
                        transcripts,
                        caption_model,
                        caption_tokenizer,
                    )

                    # Assign results to corresponding indices
                    for i, index in enumerate(batch_indices):
                        if i < len(batch_results):
                            caption_result[index] = batch_results[i]
                        else:
                            caption_result[index] = ""

                    # Clear GPU cache after batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(
                        f"Failed to caption batch starting at {batch_start}: {str(e)}"
                    )
                    # Fallback to individual processing for this batch
                    for index in batch_indices:
                        try:
                            frame_times = segment_times_info[index]["frame_times"]
                            video_frames = encode_video_frames(video, frame_times)
                            segment_transcript = transcripts.get(index, "")

                            query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                            msgs = [{"role": "user", "content": video_frames + [query]}]
                            params = {"use_image_id": False, "max_slice_nums": 2}

                            segment_caption_text = caption_model.chat(
                                image=None,
                                msgs=msgs,
                                tokenizer=caption_tokenizer,
                                **params,
                            )

                            caption_result[index] = segment_caption_text.replace(
                                "\n", ""
                            ).replace("<|endoftext|>", "")

                        except Exception as e:
                            logger.info(
                                f"Warning: Failed to caption segment {index}: {str(e)}"
                            )
                            caption_result[index] = ""

    except Exception as e:
        logger.error(f"Error in segment_caption: {str(e)}")
        # Provide empty captions for all segments
        for index in segment_index2name:
            caption_result[index] = ""


def _caption_batch(
    video,
    batch_indices,
    segment_index2name,
    segment_times_info,
    transcripts,
    caption_model,
    caption_tokenizer,
):
    """Process a batch of video segments for captioning"""
    results = []

    try:
        # Prepare batch data
        batch_frames = []
        batch_queries = []

        for index in batch_indices:
            frame_times = segment_times_info[index]["frame_times"]
            video_frames = encode_video_frames(video, frame_times)
            segment_transcript = transcripts.get(index, "")

            query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."

            batch_frames.append(video_frames)
            batch_queries.append(query)

        # Process batch - for now we'll process them sequentially but with optimized memory usage
        # Future enhancement: true batch processing if the model supports it
        for i, (video_frames, query) in enumerate(zip(batch_frames, batch_queries)):
            try:
                msgs = [{"role": "user", "content": video_frames + [query]}]
                params = {"use_image_id": False, "max_slice_nums": 2}

                segment_caption_text = caption_model.chat(
                    image=None, msgs=msgs, tokenizer=caption_tokenizer, **params
                )

                cleaned_caption = segment_caption_text.replace("\n", "").replace(
                    "<|endoftext|>", ""
                )
                results.append(cleaned_caption)

            except Exception as e:
                logger.warning(
                    f"Failed to caption segment in batch at position {i}: {str(e)}"
                )
                results.append("")

    except Exception as e:
        logger.error(f"Error in batch captioning: {str(e)}")
        # Return empty results for all segments in batch
        results = [""] * len(batch_indices)

    return results


def merge_segment_information(
    segment_index2name: dict,
    segment_times_info: dict,
    transcripts: dict,
    captions: dict,
):
    """Merge all segment information into a unified structure"""
    segments_information = {}

    for index in segment_index2name:
        segment_name = segment_index2name[index]
        start_time, end_time = segment_times_info[index]["timestamp"]

        segments_information[segment_name] = {
            "content": f"Caption:\n{captions.get(index, '')}\nTranscript:\n{transcripts.get(index, '')}\n\n",
            "caption": captions.get(index, ""),
            "transcript": transcripts.get(index, ""),
            "time": f"{start_time}-{end_time}",
            "start_time": start_time,
            "end_time": end_time,
            "frame_times": segment_times_info[index]["frame_times"].tolist(),
            "video_segment_path": segment_times_info[index].get(
                "video_segment_path", ""
            ),
            "frame_count": len(segment_times_info[index]["frame_times"]),
        }

    return segments_information
