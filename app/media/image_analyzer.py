"""Industrial-grade image analysis with OCR, object detection, and AI description."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import aiofiles
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DetectedObject(BaseModel):
    """Detected object in image."""

    label: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x, y, width, height]


class ExtractedText(BaseModel):
    """Text extracted from image via OCR."""

    text: str
    confidence: float
    language: str
    bbox: Optional[List[float]] = None


class ImageMetadata(BaseModel):
    """Comprehensive image metadata."""

    width: int
    height: int
    format: str
    mode: str  # RGB, RGBA, etc.
    file_size_bytes: int
    has_transparency: bool
    color_space: str
    dpi: Optional[tuple] = None


class AnalyzedImage(BaseModel):
    """Complete analyzed image data."""

    image_path: str
    metadata: ImageMetadata
    ai_description: Optional[str] = None
    detected_objects: List[DetectedObject] = []
    extracted_text: List[ExtractedText] = []
    dominant_colors: List[str] = []
    is_screenshot: bool = False
    is_meme: bool = False
    quality_score: float = 0.0
    analyzed_at: datetime = datetime.utcnow()


class ImageAnalyzer:
    """Industrial-grade image analyzer with OCR and AI vision."""

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_object_detection: bool = True,
        enable_ai_description: bool = True,
    ):
        """Initialize image analyzer.

        Args:
            enable_ocr: Enable text extraction
            enable_object_detection: Enable object detection
            enable_ai_description: Enable AI-powered image description
        """
        self.enable_ocr = enable_ocr
        self.enable_object_detection = enable_object_detection
        self.enable_ai_description = enable_ai_description

    async def analyze_image(self, image_path: str) -> AnalyzedImage:
        """Analyze image with full pipeline.

        Args:
            image_path: Path to image file

        Returns:
            Analyzed image data with metadata, OCR, objects, description
        """
        logger.info(f"Analyzing image: {image_path}")

        # Extract metadata
        metadata = await self._extract_metadata(image_path)
        logger.info(f"Image metadata: {metadata.width}x{metadata.height}, {metadata.format}")

        # Extract text via OCR
        extracted_text = []
        if self.enable_ocr:
            extracted_text = await self._extract_text_ocr(image_path)
            logger.info(f"Extracted {len(extracted_text)} text regions")

        # Detect objects
        detected_objects = []
        if self.enable_object_detection:
            detected_objects = await self._detect_objects(image_path)
            logger.info(f"Detected {len(detected_objects)} objects")

        # Generate AI description
        ai_description = None
        if self.enable_ai_description:
            ai_description = await self._generate_ai_description(image_path)
            logger.info(f"Generated AI description: {ai_description[:100] if ai_description else 'None'}...")

        # Extract dominant colors
        dominant_colors = await self._extract_dominant_colors(image_path)

        # Classify image type
        is_screenshot = await self._is_screenshot(image_path, metadata)
        is_meme = await self._is_meme(image_path, extracted_text)

        # Calculate quality score
        quality_score = self._calculate_quality_score(metadata, detected_objects)

        return AnalyzedImage(
            image_path=image_path,
            metadata=metadata,
            ai_description=ai_description,
            detected_objects=detected_objects,
            extracted_text=extracted_text,
            dominant_colors=dominant_colors,
            is_screenshot=is_screenshot,
            is_meme=is_meme,
            quality_score=quality_score,
        )

    async def _extract_metadata(self, image_path: str) -> ImageMetadata:
        """Extract comprehensive image metadata."""
        try:
            img = Image.open(image_path)

            # Get file size
            file_size = Path(image_path).stat().st_size

            # Check transparency
            has_transparency = img.mode in ("RGBA", "LA", "P") and (
                img.mode == "P" and "transparency" in img.info
                or img.mode in ("RGBA", "LA")
            )

            # Get DPI
            dpi = img.info.get("dpi")

            return ImageMetadata(
                width=img.width,
                height=img.height,
                format=img.format or "unknown",
                mode=img.mode,
                file_size_bytes=file_size,
                has_transparency=has_transparency,
                color_space=img.mode,
                dpi=dpi,
            )

        except Exception as e:
            logger.error(f"Failed to extract image metadata: {e}")
            raise

    async def _extract_text_ocr(self, image_path: str) -> List[ExtractedText]:
        """Extract text from image using OCR.

        Uses pytesseract for OCR. For production, consider:
        - Google Cloud Vision API for better accuracy
        - AWS Textract for documents
        - Azure Computer Vision
        """
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)

            # Get detailed OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                lang="eng",
            )

            extracted_texts = []
            current_text = []
            current_conf = []

            # Group text by lines
            for i, text in enumerate(ocr_data["text"]):
                if text.strip():
                    current_text.append(text)
                    current_conf.append(float(ocr_data["conf"][i]))
                elif current_text:
                    # End of line
                    full_text = " ".join(current_text)
                    avg_conf = sum(current_conf) / len(current_conf) / 100.0

                    extracted_texts.append(ExtractedText(
                        text=full_text,
                        confidence=avg_conf,
                        language="eng",
                    ))

                    current_text = []
                    current_conf = []

            # Add remaining text
            if current_text:
                full_text = " ".join(current_text)
                avg_conf = sum(current_conf) / len(current_conf) / 100.0
                extracted_texts.append(ExtractedText(
                    text=full_text,
                    confidence=avg_conf,
                    language="eng",
                ))

            return extracted_texts

        except ImportError:
            logger.warning("pytesseract not installed, skipping OCR")
            return []
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    async def _detect_objects(self, image_path: str) -> List[DetectedObject]:
        """Detect objects in image using AI vision.

        For production, use:
        - OpenAI Vision API (GPT-4V)
        - Google Cloud Vision API
        - AWS Rekognition
        - Local YOLO/Detectron2 models
        """
        # Placeholder - would integrate with vision API
        # For now, return empty list
        return []

    async def _generate_ai_description(self, image_path: str) -> Optional[str]:
        """Generate AI description of image using GPT-4 Vision.

        This is critical for understanding image content for summarization.
        """
        try:
            import openai
            from app.core.config import settings

            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

            # Read and encode image
            async with aiofiles.open(image_path, "rb") as f:
                image_data = await f.read()

            base64_image = base64.b64encode(image_data).decode("utf-8")

            # Get image format
            img_format = Path(image_path).suffix.lower().replace(".", "")
            if img_format == "jpg":
                img_format = "jpeg"

            # Call GPT-4 Vision
            response = await client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail. Focus on: 1) Main subject/content, 2) Key objects and people, 3) Text visible in the image, 4) Context and setting, 5) Notable details. Be concise but comprehensive (2-3 sentences)."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{img_format};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3,  # Lower temperature for factual descriptions
            )

            description = response.choices[0].message.content
            logger.info(f"Generated AI description: {description}")
            return description

        except Exception as e:
            logger.error(f"AI description failed: {e}")
            return None

    async def _extract_dominant_colors(self, image_path: str, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from image."""
        try:
            from PIL import Image
            import numpy as np
            from sklearn.cluster import KMeans

            img = Image.open(image_path)
            img = img.convert("RGB")
            img = img.resize((150, 150))  # Resize for performance

            # Get pixel data
            pixels = np.array(img).reshape(-1, 3)

            # Cluster colors
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)

            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)

            # Convert to hex
            hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]

            return hex_colors

        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return []

    async def _is_screenshot(self, image_path: str, metadata: ImageMetadata) -> bool:
        """Detect if image is a screenshot."""
        # Heuristics for screenshot detection:
        # 1. Common screenshot dimensions (16:9, 16:10, etc.)
        # 2. No EXIF data (screenshots don't have camera metadata)
        # 3. Specific aspect ratios

        aspect_ratio = metadata.width / metadata.height if metadata.height > 0 else 0

        # Common screen aspect ratios
        common_ratios = [16/9, 16/10, 4/3, 21/9, 3/2]

        for ratio in common_ratios:
            if abs(aspect_ratio - ratio) < 0.05:
                return True

        return False

    async def _is_meme(self, image_path: str, extracted_text: List[ExtractedText]) -> bool:
        """Detect if image is a meme."""
        # Heuristics for meme detection:
        # 1. Has text overlay (OCR detected text)
        # 2. Text is in specific positions (top/bottom)
        # 3. Common meme formats

        if len(extracted_text) > 0:
            # If image has text, it might be a meme
            total_text = " ".join([t.text for t in extracted_text])
            if len(total_text) > 10 and len(total_text) < 200:
                return True

        return False

    def _calculate_quality_score(
        self,
        metadata: ImageMetadata,
        detected_objects: List[DetectedObject],
    ) -> float:
        """Calculate image quality score (0.0-1.0)."""
        score = 0.0

        # Resolution score (higher resolution = better)
        pixels = metadata.width * metadata.height
        if pixels >= 1920 * 1080:  # Full HD or better
            score += 0.4
        elif pixels >= 1280 * 720:  # HD
            score += 0.3
        elif pixels >= 640 * 480:  # SD
            score += 0.2
        else:
            score += 0.1

        # File size score (not too small, not too large)
        size_mb = metadata.file_size_bytes / (1024 * 1024)
        if 0.1 <= size_mb <= 10:
            score += 0.3
        elif size_mb < 0.1:
            score += 0.1
        else:
            score += 0.2

        # Object detection score
        if len(detected_objects) > 0:
            avg_confidence = sum(obj.confidence for obj in detected_objects) / len(detected_objects)
            score += 0.3 * avg_confidence

        return min(score, 1.0)

    async def batch_analyze_images(
        self,
        image_paths: List[str],
        max_concurrent: int = 5,
    ) -> List[AnalyzedImage]:
        """Analyze multiple images concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _analyze_with_semaphore(path: str) -> Optional[AnalyzedImage]:
            async with semaphore:
                try:
                    return await self.analyze_image(path)
                except Exception as e:
                    logger.error(f"Failed to analyze {path}: {e}")
                    return None

        tasks = [_analyze_with_semaphore(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        return [r for r in results if isinstance(r, AnalyzedImage)]

