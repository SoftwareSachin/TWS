"""
Video splitting and processing utilities
"""

import os
import shutil
import time

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


def split_video(
    video_path: str,
    working_dir: str,
    segment_length: int,
    num_frames_per_segment: int,
    audio_output_format: str = "mp3",
):
    """Split video into segments and extract audio"""
    unique_timestamp = str(int(time.time() * 1000))
    video_name = os.path.basename(video_path).split(".")[0]
    video_segment_cache_path = os.path.join(working_dir, "_cache", video_name)

    # Clean and create cache directory
    if os.path.exists(video_segment_cache_path):
        shutil.rmtree(video_segment_cache_path)
    os.makedirs(video_segment_cache_path, exist_ok=False)

    segment_index = 0
    segment_index2name, segment_times_info = {}, {}

    with VideoFileClip(video_path) as video:
        total_video_length = int(video.duration)
        start_times = list(range(0, total_video_length, segment_length))

        # Merge short last segment if < 5 seconds
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1]

        for start in tqdm(start_times, desc=f"Splitting Video {video_name}"):
            if start != start_times[-1]:
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length

            subvideo = video.subclip(start, end)
            subvideo_length = subvideo.duration
            frame_times = np.linspace(
                0, subvideo_length, num_frames_per_segment, endpoint=False
            )
            frame_times += start

            segment_name = f"{unique_timestamp}-{segment_index}-{start}-{end}"
            segment_index2name[f"{segment_index}"] = segment_name
            segment_times_info[f"{segment_index}"] = {
                "frame_times": frame_times,
                "timestamp": (start, end),
                "start_time": start,
                "end_time": end,
            }

            # Save audio
            audio_file = f"{segment_name}.{audio_output_format}"
            subaudio = subvideo.audio
            subaudio.write_audiofile(
                os.path.join(video_segment_cache_path, audio_file),
                codec="mp3",
                verbose=False,
                logger=None,
            )

            segment_index += 1

    return segment_index2name, segment_times_info
