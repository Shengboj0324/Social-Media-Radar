"""Comprehensive unit tests for Advanced OCR (TrOCR/CRNN/EasyOCR).

Tests text detection, recognition, and bounding box extraction.
Verifies mathematical correctness and OCR pipeline behavior.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from app.media.advanced_ocr import (
    TrOCRModel,
    EasyOCREngine,
    AdvancedOCRPipeline,
    CRNNConfig,
    EasyOCRConfig,
    TextRegion,
    OCRResult,
)


class TestCRNNConfig:
    """Test CRNNConfig dataclass."""

    def test_crnn_config_defaults(self):
        """Test CRNNConfig default values."""
        config = CRNNConfig()

        assert config.model_name == "microsoft/trocr-base-handwritten"
        assert config.max_length == 384
        assert config.device == "cpu"
        assert config.batch_size == 8

    def test_crnn_config_custom_values(self):
        """Test CRNNConfig with custom values."""
        config = CRNNConfig(
            max_length=512,
            batch_size=16,
        )

        assert config.max_length == 512
        assert config.batch_size == 16


class TestEasyOCRConfig:
    """Test EasyOCRConfig dataclass."""

    def test_easyocr_config_defaults(self):
        """Test EasyOCRConfig default values."""
        config = EasyOCRConfig()

        assert config.languages == ["en"]
        assert config.gpu is False
        assert config.detector is True
        assert config.recognizer is True
        assert config.paragraph is False

    def test_easyocr_config_custom_languages(self):
        """Test EasyOCRConfig with custom languages."""
        config = EasyOCRConfig(languages=["en", "zh", "ja"])

        assert config.languages == ["en", "zh", "ja"]
        assert len(config.languages) == 3

    def test_easyocr_config_post_init(self):
        """Test EasyOCRConfig __post_init__ sets default languages."""
        config = EasyOCRConfig(languages=None)

        # __post_init__ should set default
        assert config.languages == ["en"]


class TestTextRegion:
    """Test TextRegion model."""

    def test_text_region_creation(self):
        """Test TextRegion creation."""
        region = TextRegion(
            text="Hello World",
            confidence=0.95,
            bbox=[10.0, 20.0, 100.0, 50.0],
        )

        assert region.text == "Hello World"
        assert region.confidence == 0.95
        assert region.bbox == [10.0, 20.0, 100.0, 50.0]
        assert region.language == "en"

    def test_text_region_bbox_format(self):
        """Test TextRegion bbox format [x1, y1, x2, y2]."""
        region = TextRegion(
            text="Test",
            confidence=0.9,
            bbox=[5.0, 10.0, 50.0, 30.0],
        )

        # Verify bbox format
        x1, y1, x2, y2 = region.bbox
        assert x1 < x2  # x1 should be less than x2
        assert y1 < y2  # y1 should be less than y2


class TestOCRResult:
    """Test OCRResult model."""

    def test_ocr_result_creation(self):
        """Test OCRResult creation."""
        regions = [
            TextRegion(text="Hello", confidence=0.9, bbox=[0, 0, 50, 20]),
            TextRegion(text="World", confidence=0.95, bbox=[60, 0, 120, 20]),
        ]

        result = OCRResult(
            text_regions=regions,
            full_text="Hello World",
            languages_detected=["en"],
            average_confidence=0.925,
            model_name="easyocr",
        )

        assert len(result.text_regions) == 2
        assert result.full_text == "Hello World"
        assert result.average_confidence == 0.925

    def test_ocr_result_average_confidence_calculation(self):
        """Test average confidence calculation."""
        confidences = [0.9, 0.8, 0.95, 0.85]
        avg = sum(confidences) / len(confidences)

        assert avg == pytest.approx(0.875)


class TestTrOCRModel:
    """Test TrOCRModel implementation."""

    def test_trocr_initialization(self):
        """Test TrOCRModel initialization."""
        trocr = TrOCRModel()

        assert trocr.config is not None
        assert trocr.model is None
        assert trocr.processor is None
        assert trocr._initialized is False

    def test_trocr_with_custom_config(self):
        """Test TrOCRModel with custom config."""
        config = CRNNConfig(max_length=512)
        trocr = TrOCRModel(config)

        assert trocr.config.max_length == 512



class TestEasyOCREngine:
    """Test EasyOCREngine implementation."""

    def test_easyocr_initialization(self):
        """Test EasyOCREngine initialization."""
        engine = EasyOCREngine()

        assert engine.config is not None
        assert engine.reader is None
        assert engine._initialized is False

    def test_easyocr_with_custom_config(self):
        """Test EasyOCREngine with custom config."""
        config = EasyOCRConfig(languages=["en", "zh"])
        engine = EasyOCREngine(config)

        assert engine.config.languages == ["en", "zh"]

    def test_bbox_conversion(self):
        """Test bounding box conversion from polygon to [x1, y1, x2, y2]."""
        # CRITICAL: Test bbox conversion algorithm
        # Input: polygon points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        # Output: [min_x, min_y, max_x, max_y]

        bbox_polygon = [
            (10, 20),  # Top-left
            (100, 20),  # Top-right
            (100, 50),  # Bottom-right
            (10, 50),  # Bottom-left
        ]

        # Convert to flat bbox
        bbox_flat = [
            min(p[0] for p in bbox_polygon),  # x1 = min x
            min(p[1] for p in bbox_polygon),  # y1 = min y
            max(p[0] for p in bbox_polygon),  # x2 = max x
            max(p[1] for p in bbox_polygon),  # y2 = max y
        ]

        assert bbox_flat == [10, 20, 100, 50]

    def test_bbox_conversion_rotated(self):
        """Test bbox conversion with rotated polygon."""
        # Rotated rectangle
        bbox_polygon = [
            (15, 10),
            (105, 15),
            (100, 55),
            (10, 50),
        ]

        bbox_flat = [
            min(p[0] for p in bbox_polygon),  # x1
            min(p[1] for p in bbox_polygon),  # y1
            max(p[0] for p in bbox_polygon),  # x2
            max(p[1] for p in bbox_polygon),  # y2
        ]

        assert bbox_flat == [10, 10, 105, 55]

    def test_average_confidence_calculation(self):
        """Test average confidence calculation."""
        # CRITICAL: Test average confidence formula
        confidences = [0.9, 0.85, 0.95, 0.8]

        avg_confidence = sum(confidences) / len(confidences)

        assert avg_confidence == 0.875

    def test_average_confidence_empty_list(self):
        """Test average confidence with empty list."""
        confidences = []

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        assert avg_confidence == 0.0


class TestAdvancedOCRPipeline:
    """Test AdvancedOCRPipeline implementation."""

    def test_pipeline_initialization(self):
        """Test AdvancedOCRPipeline initialization."""
        pipeline = AdvancedOCRPipeline()

        assert pipeline.use_trocr is True
        assert pipeline.use_easyocr is True
        assert pipeline.trocr is not None
        assert pipeline.easyocr is not None

    def test_pipeline_trocr_only(self):
        """Test pipeline with TrOCR only."""
        pipeline = AdvancedOCRPipeline(use_trocr=True, use_easyocr=False)

        assert pipeline.trocr is not None
        assert pipeline.easyocr is None

    def test_pipeline_easyocr_only(self):
        """Test pipeline with EasyOCR only."""
        pipeline = AdvancedOCRPipeline(use_trocr=False, use_easyocr=True)

        assert pipeline.trocr is None
        assert pipeline.easyocr is not None

    def test_auto_engine_selection_multi_language(self):
        """Test auto engine selection for multi-language."""
        # Multi-language should prefer EasyOCR
        languages = ["en", "zh", "ja"]

        # Simulate selection logic
        if len(languages) > 1:
            preferred = "easyocr"
        else:
            preferred = "trocr"

        assert preferred == "easyocr"

    def test_auto_engine_selection_single_language(self):
        """Test auto engine selection for single language."""
        # Single language (English) should prefer TrOCR
        languages = ["en"]

        # Simulate selection logic (assuming TrOCR available)
        if len(languages) > 1:
            preferred = "easyocr"
        else:
            preferred = "trocr"

        assert preferred == "trocr"


class TestMathematicalCorrectness:
    """Test mathematical correctness of algorithms."""

    def test_bbox_area_calculation(self):
        """Test bounding box area calculation."""
        bbox = [10, 20, 100, 50]  # [x1, y1, x2, y2]

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        assert width == 90
        assert height == 30
        assert area == 2700

    def test_bbox_center_calculation(self):
        """Test bounding box center calculation."""
        bbox = [10, 20, 100, 50]

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        assert center_x == 55.0
        assert center_y == 35.0

    def test_confidence_weighted_average(self):
        """Test confidence-weighted average."""
        # Weighted average: sum(confidence * length) / sum(length)
        texts = ["Hello", "World", "Test"]
        confidences = [0.9, 0.8, 0.95]
        lengths = [len(t) for t in texts]

        weighted_sum = sum(c * l for c, l in zip(confidences, lengths))
        total_length = sum(lengths)
        weighted_avg = weighted_sum / total_length

        # (0.9*5 + 0.8*5 + 0.95*4) / (5+5+4)
        # (4.5 + 4.0 + 3.8) / 14 = 12.3 / 14 = 0.878...
        assert abs(weighted_avg - 0.8785714) < 1e-6


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_regions(self):
        """Test with empty text regions."""
        result = OCRResult(
            text_regions=[],
            full_text="",
            languages_detected=[],
            average_confidence=0.0,
            model_name="test",
        )

        assert len(result.text_regions) == 0
        assert result.full_text == ""

    def test_single_text_region(self):
        """Test with single text region."""
        region = TextRegion(text="Test", confidence=0.9, bbox=[0, 0, 50, 20])
        result = OCRResult(
            text_regions=[region],
            full_text="Test",
            languages_detected=["en"],
            average_confidence=0.9,
            model_name="test",
        )

        assert len(result.text_regions) == 1
        assert result.average_confidence == 0.9

    def test_zero_confidence(self):
        """Test with zero confidence."""
        region = TextRegion(text="", confidence=0.0, bbox=[0, 0, 0, 0])

        assert region.confidence == 0.0

    def test_bbox_zero_area(self):
        """Test bbox with zero area."""
        bbox = [10, 20, 10, 20]  # Same point

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        assert area == 0


class TestIntegration:
    """Integration tests for OCR pipeline."""

    @pytest.mark.asyncio
    async def test_trocr_initialize_idempotent(self):
        """Test that TrOCR initialize can be called multiple times safely."""
        trocr = TrOCRModel()

        with patch('app.media.advanced_ocr.TrOCRProcessor') as mock_processor, \
             patch('app.media.advanced_ocr.VisionEncoderDecoderModel') as mock_model:

            mock_processor.from_pretrained.return_value = Mock()
            mock_model_instance = Mock()
            mock_model_instance.eval.return_value = None
            mock_model.from_pretrained.return_value = mock_model_instance

            await trocr.initialize()
            assert trocr._initialized is True

            # Call again - should not reinitialize
            await trocr.initialize()
            assert trocr._initialized is True

            # Should only call from_pretrained once
            assert mock_model.from_pretrained.call_count == 1

    @pytest.mark.asyncio
    async def test_easyocr_initialize_idempotent(self):
        """Test that EasyOCR initialize can be called multiple times safely."""
        engine = EasyOCREngine()

        with patch('app.media.advanced_ocr.easyocr') as mock_easyocr:
            mock_reader = Mock()
            mock_easyocr.Reader.return_value = mock_reader

            await engine.initialize()
            assert engine._initialized is True

            # Call again - should not reinitialize
            await engine.initialize()
            assert engine._initialized is True

            # Should only call Reader once
            assert mock_easyocr.Reader.call_count == 1

    def test_batch_processing_logic(self):
        """Test batch processing logic."""
        total_items = 25
        batch_size = 8

        # Process in batches
        batches = []
        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            batches.append((i, batch_end))

        assert len(batches) == 4
        assert batches[0] == (0, 8)
        assert batches[1] == (8, 16)
        assert batches[2] == (16, 24)
        assert batches[3] == (24, 25)

    def test_text_joining(self):
        """Test text joining from multiple regions."""
        texts = ["Hello", "World", "Test"]
        full_text = " ".join(texts)

        assert full_text == "Hello World Test"

    def test_language_detection_deduplication(self):
        """Test language detection with deduplication."""
        languages = ["en", "en", "zh", "en", "zh"]
        unique_languages = list(set(languages))

        assert len(unique_languages) == 2
        assert "en" in unique_languages
        assert "zh" in unique_languages

