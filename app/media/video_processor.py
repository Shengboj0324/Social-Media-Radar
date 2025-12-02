"""Industrial-grade video processing with transcription, scene detection, and metadata extraction."""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import aiofiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VideoMetadata(BaseModel):
    """Comprehensive video metadata."""

    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    file_size_bytes: int
    format: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_bitrate: Optional[int] = None
    audio_sample_rate: Optional[int] = None


class VideoScene(BaseModel):
    """Detected scene in video."""

    start_time: float
    end_time: float
    duration: float
    keyframe_path: Optional[str] = None
    description: Optional[str] = None  # AI-generated description


class VideoTranscript(BaseModel):
    """Video transcript with timestamps."""

    segments: List[Dict[str, Any]]  # [{text, start, end, confidence}]
    full_text: str
    language: str
    confidence: float


class ProcessedVideo(BaseModel):
    """Complete processed video data."""

    video_path: str
    metadata: VideoMetadata
    transcript: Optional[VideoTranscript] = None
    scenes: List[VideoScene] = []
    keyframes: List[str] = []
    thumbnail_path: Optional[str] = None
    audio_path: Optional[str] = None
    processed_at: datetime = datetime.utcnow()


class VideoProcessor:
    """Industrial-grade video processor with transcription and scene detection."""

    def __init__(
        self,
        output_dir: str = "/tmp/video_processing",
        enable_transcription: bool = True,
        enable_scene_detection: bool = True,
        enable_keyframe_extraction: bool = True,
    ):
        """Initialize video processor.

        Args:
            output_dir: Directory for processed outputs
            enable_transcription: Enable audio transcription
            enable_scene_detection: Enable scene detection
            enable_keyframe_extraction: Enable keyframe extraction
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_transcription = enable_transcription
        self.enable_scene_detection = enable_scene_detection
        self.enable_keyframe_extraction = enable_keyframe_extraction

    async def process_video(
        self,
        video_path: str,
        extract_audio: bool = True,
        generate_thumbnail: bool = True,
    ) -> ProcessedVideo:
        """Process video with full analysis pipeline.

        Args:
            video_path: Path to video file
            extract_audio: Extract audio track
            generate_thumbnail: Generate thumbnail image

        Returns:
            Processed video data with metadata, transcript, scenes
        """
        logger.info(f"Processing video: {video_path}")

        # Extract metadata
        metadata = await self._extract_metadata(video_path)
        logger.info(f"Video metadata: {metadata.duration_seconds}s, {metadata.width}x{metadata.height}")

        # Generate thumbnail
        thumbnail_path = None
        if generate_thumbnail:
            thumbnail_path = await self._generate_thumbnail(video_path)
            logger.info(f"Generated thumbnail: {thumbnail_path}")

        # Extract audio
        audio_path = None
        if extract_audio and metadata.has_audio:
            audio_path = await self._extract_audio(video_path)
            logger.info(f"Extracted audio: {audio_path}")

        # Transcribe audio
        transcript = None
        if self.enable_transcription and audio_path:
            transcript = await self._transcribe_audio(audio_path)
            logger.info(f"Transcribed {len(transcript.segments)} segments")

        # Detect scenes
        scenes = []
        if self.enable_scene_detection:
            scenes = await self._detect_scenes(video_path)
            logger.info(f"Detected {len(scenes)} scenes")

        # Extract keyframes
        keyframes = []
        if self.enable_keyframe_extraction:
            keyframes = await self._extract_keyframes(video_path, num_frames=10)
            logger.info(f"Extracted {len(keyframes)} keyframes")

        return ProcessedVideo(
            video_path=video_path,
            metadata=metadata,
            transcript=transcript,
            scenes=scenes,
            keyframes=keyframes,
            thumbnail_path=thumbnail_path,
            audio_path=audio_path,
        )

    async def _extract_metadata(self, video_path: str) -> VideoMetadata:
        """Extract comprehensive video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {stderr.decode()}")

            data = json.loads(stdout.decode())

            # Find video stream
            video_stream = next(
                (s for s in data["streams"] if s["codec_type"] == "video"),
                None
            )
            if not video_stream:
                raise ValueError("No video stream found")

            # Find audio stream
            audio_stream = next(
                (s for s in data["streams"] if s["codec_type"] == "audio"),
                None
            )

            # Parse FPS
            fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 0.0

            return VideoMetadata(
                duration_seconds=float(data["format"].get("duration", 0)),
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                fps=fps,
                codec=video_stream.get("codec_name", "unknown"),
                bitrate=int(data["format"].get("bit_rate", 0)),
                file_size_bytes=int(data["format"].get("size", 0)),
                format=data["format"].get("format_name", "unknown"),
                has_audio=audio_stream is not None,
                audio_codec=audio_stream.get("codec_name") if audio_stream else None,
                audio_bitrate=int(audio_stream.get("bit_rate", 0)) if audio_stream else None,
                audio_sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
            )

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise

    async def _generate_thumbnail(self, video_path: str, timestamp: float = 1.0) -> str:
        """Generate thumbnail from video at specified timestamp."""
        output_path = self.output_dir / f"{Path(video_path).stem}_thumb.jpg"

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",  # High quality
            str(output_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError("Failed to generate thumbnail")

        return str(output_path)

    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio track from video."""
        output_path = self.output_dir / f"{Path(video_path).stem}_audio.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # WAV format
            "-ar", "16000",  # 16kHz sample rate (optimal for speech)
            "-ac", "1",  # Mono
            str(output_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError("Failed to extract audio")

        return str(output_path)

    async def _transcribe_audio(self, audio_path: str) -> VideoTranscript:
        """Transcribe audio using Whisper API.

        Note: This uses OpenAI Whisper API. For production, consider:
        - Local Whisper model for cost savings
        - Batch processing for efficiency
        - Multiple language support
        """
        try:
            import openai
            from app.core.config import settings

            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

            # Read audio file
            async with aiofiles.open(audio_path, "rb") as f:
                audio_data = await f.read()

            # Transcribe with timestamps
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

            # Parse response
            segments = []
            full_text_parts = []

            for segment in response.segments:
                segments.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": getattr(segment, "confidence", 1.0),
                })
                full_text_parts.append(segment.text)

            return VideoTranscript(
                segments=segments,
                full_text=" ".join(full_text_parts),
                language=response.language,
                confidence=sum(s["confidence"] for s in segments) / len(segments) if segments else 0.0,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Return empty transcript on failure
            return VideoTranscript(
                segments=[],
                full_text="",
                language="unknown",
                confidence=0.0,
            )

    async def _detect_scenes(self, video_path: str, threshold: float = 0.3) -> List[VideoScene]:
        """Detect scene changes in video using ffmpeg.

        Args:
            video_path: Path to video file
            threshold: Scene detection sensitivity (0.0-1.0, lower = more sensitive)

        Returns:
            List of detected scenes with timestamps
        """
        try:
            # Use ffmpeg scene detection filter
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-filter:v", f"select='gt(scene,{threshold})',showinfo",
                "-f", "null",
                "-",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            # Parse scene timestamps from stderr (ffmpeg outputs to stderr)
            stderr_text = stderr.decode()
            scenes = []

            # Extract timestamps from showinfo output
            import re
            pattern = r"pts_time:([\d.]+)"
            timestamps = [float(m.group(1)) for m in re.finditer(pattern, stderr_text)]

            # Create scenes from timestamps
            for i, start_time in enumerate(timestamps):
                end_time = timestamps[i + 1] if i + 1 < len(timestamps) else None
                if end_time:
                    scenes.append(VideoScene(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                    ))

            logger.info(f"Detected {len(scenes)} scenes")
            return scenes

        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []

    async def _extract_keyframes(
        self,
        video_path: str,
        num_frames: int = 10,
    ) -> List[str]:
        """Extract keyframes from video at regular intervals.

        Args:
            video_path: Path to video file
            num_frames: Number of keyframes to extract

        Returns:
            List of paths to extracted keyframe images
        """
        try:
            # Get video duration first
            metadata = await self._extract_metadata(video_path)
            duration = metadata.duration_seconds

            if duration <= 0:
                return []

            # Calculate frame extraction interval
            interval = duration / (num_frames + 1)

            keyframe_paths = []
            for i in range(1, num_frames + 1):
                timestamp = interval * i
                output_path = self.output_dir / f"{Path(video_path).stem}_keyframe_{i:03d}.jpg"

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss", str(timestamp),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",
                    str(output_path),
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

                if proc.returncode == 0:
                    keyframe_paths.append(str(output_path))

            logger.info(f"Extracted {len(keyframe_paths)} keyframes")
            return keyframe_paths

        except Exception as e:
            logger.error(f"Keyframe extraction failed: {e}")
            return []

    async def batch_process_videos(
        self,
        video_paths: List[str],
        max_concurrent: int = 3,
    ) -> List[ProcessedVideo]:
        """Process multiple videos concurrently.

        Args:
            video_paths: List of video file paths
            max_concurrent: Maximum concurrent processing tasks

        Returns:
            List of processed video data
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_with_semaphore(path: str) -> Optional[ProcessedVideo]:
            async with semaphore:
                try:
                    return await self.process_video(path)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    return None

        tasks = [_process_with_semaphore(path) for path in video_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        return [r for r in results if isinstance(r, ProcessedVideo)]



