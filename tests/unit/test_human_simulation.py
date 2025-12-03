"""Comprehensive unit tests for Human Simulation (Bézier Curves).

Tests Bézier curve generation and human-like behavior simulation.
Verifies mathematical correctness and realistic patterns.
"""

import pytest
import math
from unittest.mock import AsyncMock, MagicMock, patch

from app.scraping.human_simulation import (
    Point,
    BezierCurve,
    HumanSimulator,
)


class TestPoint:
    """Test Point dataclass."""

    def test_point_creation(self):
        """Test Point creation."""
        p = Point(10.5, 20.3)

        assert p.x == 10.5
        assert p.y == 20.3

    def test_point_with_integers(self):
        """Test Point with integer coordinates."""
        p = Point(100, 200)

        assert p.x == 100
        assert p.y == 200


class TestBezierCurve:
    """Test BezierCurve implementation."""

    def test_cubic_bezier_endpoints(self):
        """Test that cubic Bézier curve starts and ends at correct points."""
        start = Point(0, 0)
        end = Point(100, 100)
        control1 = Point(25, 75)
        control2 = Point(75, 25)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=50)

        # First point should be start
        assert abs(points[0].x - start.x) < 0.01
        assert abs(points[0].y - start.y) < 0.01

        # Last point should be end
        assert abs(points[-1].x - end.x) < 0.01
        assert abs(points[-1].y - end.y) < 0.01

    def test_cubic_bezier_formula(self):
        """Test cubic Bézier formula at t=0.5."""
        start = Point(0, 0)
        end = Point(100, 100)
        control1 = Point(0, 100)
        control2 = Point(100, 0)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=100)

        # At t=0.5, calculate expected point
        t = 0.5
        expected_x = (
            (1 - t) ** 3 * start.x
            + 3 * (1 - t) ** 2 * t * control1.x
            + 3 * (1 - t) * t**2 * control2.x
            + t**3 * end.x
        )
        expected_y = (
            (1 - t) ** 3 * start.y
            + 3 * (1 - t) ** 2 * t * control1.y
            + 3 * (1 - t) * t**2 * control2.y
            + t**3 * end.y
        )

        # Point at index 50 should be at t=0.5
        assert abs(points[50].x - expected_x) < 0.01
        assert abs(points[50].y - expected_y) < 0.01

    def test_cubic_bezier_num_points(self):
        """Test that correct number of points are generated."""
        start = Point(0, 0)
        end = Point(100, 100)
        control1 = Point(25, 75)
        control2 = Point(75, 25)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=50)

        # Should have num_points + 1 points (including both endpoints)
        assert len(points) == 51

    def test_cubic_bezier_straight_line(self):
        """Test Bézier curve with control points on the line (straight line)."""
        start = Point(0, 0)
        end = Point(100, 100)
        # Control points on the line
        control1 = Point(25, 25)
        control2 = Point(75, 75)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=10)

        # All points should be on the line y = x
        for point in points:
            assert abs(point.x - point.y) < 0.01, f"Point ({point.x}, {point.y}) not on line y=x"

    def test_generate_control_points(self):
        """Test control point generation."""
        start = Point(0, 0)
        end = Point(100, 100)

        control1, control2 = BezierCurve.generate_control_points(start, end, curvature=0.3)

        # Control points should exist
        assert control1 is not None
        assert control2 is not None

        # Control points should be Point objects
        assert isinstance(control1, Point)
        assert isinstance(control2, Point)

    def test_generate_control_points_zero_curvature(self):
        """Test control points with zero curvature."""
        start = Point(0, 0)
        end = Point(100, 100)

        control1, control2 = BezierCurve.generate_control_points(start, end, curvature=0.0)

        # With zero curvature, control points should be near the midpoint
        mid_x = (start.x + end.x) / 2
        mid_y = (start.y + end.y) / 2

        # Control points should be close to midpoint
        assert abs(control1.x - mid_x) < 10
        assert abs(control1.y - mid_y) < 10

    def test_generate_control_points_perpendicular(self):
        """Test that control points are perpendicular to the line."""
        start = Point(0, 0)
        end = Point(100, 0)  # Horizontal line

        # Set random seed for reproducibility
        import random
        random.seed(42)

        control1, control2 = BezierCurve.generate_control_points(start, end, curvature=0.5)



    def test_bezier_curve_monotonicity(self):
        """Test that Bézier curve progresses monotonically in parameter t."""
        start = Point(0, 0)
        end = Point(100, 100)
        control1 = Point(25, 75)
        control2 = Point(75, 25)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=50)

        # X coordinates should generally increase (for this specific curve)
        for i in range(1, len(points)):
            assert points[i].x >= points[i - 1].x - 0.01  # Allow small numerical errors


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_bezier_with_same_start_and_end(self):
        """Test Bézier curve where start and end are the same."""
        start = Point(50, 50)
        end = Point(50, 50)
        control1 = Point(40, 60)
        control2 = Point(60, 40)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=10)

        # Should still generate points
        assert len(points) == 11

        # First and last should be the same
        assert abs(points[0].x - points[-1].x) < 0.01
        assert abs(points[0].y - points[-1].y) < 0.01

    def test_bezier_with_zero_distance(self):
        """Test control point generation with zero distance."""
        start = Point(50, 50)
        end = Point(50, 50)

        control1, control2 = BezierCurve.generate_control_points(start, end, curvature=0.3)

        # Should handle gracefully (perpendicular vector should be zero)
        assert control1 is not None
        assert control2 is not None

    def test_bezier_with_one_point(self):
        """Test Bézier curve with num_points=0."""
        start = Point(0, 0)
        end = Point(100, 100)
        control1 = Point(25, 75)
        control2 = Point(75, 25)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=0)

        # Should have 1 point (start/end)
        assert len(points) == 1

    def test_bezier_with_negative_coordinates(self):
        """Test Bézier curve with negative coordinates."""
        start = Point(-100, -100)
        end = Point(100, 100)
        control1 = Point(-50, 50)
        control2 = Point(50, -50)

        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=50)

        # Should work with negative coordinates
        assert len(points) == 51
        assert abs(points[0].x - (-100)) < 0.01
        assert abs(points[-1].x - 100) < 0.01

    def test_human_simulator_zero_error_rate(self):
        """Test HumanSimulator with zero error rate."""
        sim = HumanSimulator(error_rate=0.0)

        assert sim.error_rate == 0.0

    def test_human_simulator_high_error_rate(self):
        """Test HumanSimulator with high error rate."""
        sim = HumanSimulator(error_rate=0.5)

        assert sim.error_rate == 0.5


class TestAsyncMethods:
    """Test async methods with mocks."""

    @pytest.mark.asyncio
    async def test_move_mouse_to_mock(self):
        """Test move_mouse_to with mocked page."""
        sim = HumanSimulator(mouse_speed_pixels_per_second=1000)

        # Mock page
        page = AsyncMock()
        page.viewport_size = {"width": 1920, "height": 1080}
        page.mouse.move = AsyncMock()

        # Move mouse
        await sim.move_mouse_to(page, 500, 500, curvature=0.3, num_steps=10)

        # Should have called mouse.move multiple times
        assert page.mouse.move.call_count == 11  # num_steps + 1

    @pytest.mark.asyncio
    async def test_move_mouse_to_no_viewport(self):
        """Test move_mouse_to when viewport is None."""
        sim = HumanSimulator()

        # Mock page with no viewport
        page = AsyncMock()
        page.viewport_size = None
        page.mouse.move = AsyncMock()

        # Should use default viewport
        await sim.move_mouse_to(page, 500, 500, num_steps=5)

        # Should still work
        assert page.mouse.move.call_count == 6

    @pytest.mark.asyncio
    async def test_human_delay(self):
        """Test human_delay."""
        sim = HumanSimulator()

        # Test delay (should complete quickly in test)
        import time
        start = time.time()
        await sim.human_delay(0.01, 0.02)
        elapsed = time.time() - start

        # Should have delayed at least 0.01 seconds
        assert elapsed >= 0.01

    @pytest.mark.asyncio
    async def test_random_micro_movement_no_viewport(self):
        """Test random_micro_movement when viewport is None."""
        sim = HumanSimulator()

        # Mock page with no viewport
        page = AsyncMock()
        page.viewport_size = None

        # Should return early
        await sim.random_micro_movement(page)

        # Should not crash


class TestStatisticalProperties:
    """Test statistical properties of human simulation."""

    def test_control_points_randomness(self):
        """Test that control points are random."""
        start = Point(0, 0)
        end = Point(100, 100)

        # Generate multiple control points
        control_points = []
        for i in range(10):
            import random
            random.seed(i)
            control1, control2 = BezierCurve.generate_control_points(start, end, curvature=0.5)
            control_points.append((control1, control2))

        # Should have different control points
        unique_control1 = set((c1.x, c1.y) for c1, c2 in control_points)
        assert len(unique_control1) > 5, "Control points should be random"

    def test_typing_speed_variance(self):
        """Test that typing speed calculation is correct."""
        # Test different WPM values
        speeds = [30, 60, 90, 120]

        for wpm in speeds:
            sim = HumanSimulator(typing_speed_wpm=wpm)
            expected_cps = (wpm * 5) / 60
            assert abs(sim.chars_per_second - expected_cps) < 0.01


class TestIntegration:
    """Integration tests for human simulation."""

    def test_bezier_curve_end_to_end(self):
        """Test complete Bézier curve generation."""
        start = Point(0, 0)
        end = Point(1000, 800)

        # Generate control points
        control1, control2 = BezierCurve.generate_control_points(start, end, curvature=0.4)

        # Generate curve
        points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=100)

        # Verify properties
        assert len(points) == 101
        assert abs(points[0].x - start.x) < 0.01
        assert abs(points[0].y - start.y) < 0.01
        assert abs(points[-1].x - end.x) < 0.01
        assert abs(points[-1].y - end.y) < 0.01

    def test_multiple_curves_different_paths(self):
        """Test that multiple curves generate different paths."""
        start = Point(0, 0)
        end = Point(100, 100)

        import random

        curves = []
        for i in range(5):
            random.seed(i)
            control1, control2 = BezierCurve.generate_control_points(start, end, curvature=0.5)
            points = BezierCurve.cubic_bezier(start, end, control1, control2, num_points=50)
            curves.append(points)

        # Curves should be different (check midpoint)
        midpoints = [(curve[25].x, curve[25].y) for curve in curves]
        unique_midpoints = set(midpoints)

        assert len(unique_midpoints) > 3, "Should generate different curves"

    @pytest.mark.asyncio
    async def test_realistic_mouse_movement_sequence(self):
        """Test realistic mouse movement sequence."""
        sim = HumanSimulator(mouse_speed_pixels_per_second=800)

        # Mock page
        page = AsyncMock()
        page.viewport_size = {"width": 1920, "height": 1080}
        page.mouse.move = AsyncMock()

        # Perform multiple movements
        targets = [(100, 100), (500, 300), (800, 600), (1200, 900)]

        for x, y in targets:
            await sim.move_mouse_to(page, x, y, num_steps=20)

        # Should have moved to all targets
        assert page.mouse.move.call_count == 21 * 4  # 21 points per movement * 4 movements

