"""Tests for core data models."""

from datetime import datetime
from uuid import uuid4

import pytest

from app.core.models import (
    ContentItem,
    MediaType,
    SourcePlatform,
    UserInterestProfile,
    Cluster,
)


def test_content_item_creation():
    """Test creating a ContentItem."""
    user_id = uuid4()
    item = ContentItem(
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="test123",
        source_url="https://reddit.com/r/test/comments/test123",
        title="Test Post",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )

    assert item.user_id == user_id
    assert item.source_platform == SourcePlatform.REDDIT
    assert item.title == "Test Post"
    assert item.media_type == MediaType.TEXT


def test_content_item_with_topics():
    """Test ContentItem with topics."""
    item = ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.YOUTUBE,
        source_id="video123",
        source_url="https://youtube.com/watch?v=video123",
        title="AI Tutorial",
        media_type=MediaType.VIDEO,
        published_at=datetime.utcnow(),
        topics=["AI", "machine learning", "tutorial"],
    )

    assert len(item.topics) == 3
    assert "AI" in item.topics


def test_user_interest_profile():
    """Test UserInterestProfile creation."""
    user_id = uuid4()
    profile = UserInterestProfile(
        user_id=user_id,
        interest_topics=["AI", "technology", "science"],
        negative_filters=["sports", "celebrity"],
    )

    assert profile.user_id == user_id
    assert len(profile.interest_topics) == 3
    assert len(profile.negative_filters) == 2


def test_cluster_creation():
    """Test Cluster creation."""
    user_id = uuid4()
    items = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id=f"post{i}",
            source_url=f"https://reddit.com/post{i}",
            title=f"Post {i}",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        )
        for i in range(3)
    ]

    cluster = Cluster(
        user_id=user_id,
        topic="AI News",
        summary="Latest developments in AI",
        items=items,
        item_ids=[item.id for item in items],
        relevance_score=0.85,
        platforms_represented=[SourcePlatform.REDDIT],
    )

    assert cluster.topic == "AI News"
    assert len(cluster.items) == 3
    assert cluster.relevance_score == 0.85
    assert SourcePlatform.REDDIT in cluster.platforms_represented



# ---------------------------------------------------------------------------
# DataResidencyGuard unit tests (Step 1 — competitive_analysis.md §5.1)
# ---------------------------------------------------------------------------

from app.core.data_residency import DataResidencyGuard
from app.core.errors import DataResidencyViolationError


def _make_item(**kwargs) -> ContentItem:
    """Helper: build a minimal ContentItem, overriding any field via kwargs."""
    defaults = dict(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="test123",
        source_url="https://reddit.com/r/test/comments/test123",
        title="Test Post",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return ContentItem(**defaults)


class TestDataResidencyGuard:
    """Unit tests for DataResidencyGuard — implements competitive_analysis.md §5.1."""

    def setup_method(self):
        self.guard = DataResidencyGuard()

    # (a) PII is stripped from author fields
    def test_author_is_pseudonymised(self):
        """Author real name is replaced with a stable anon_ prefix pseudonym."""
        item = _make_item(author="john.doe")
        clean = self.guard.redact(item)

        assert clean.author is not None
        assert clean.author.startswith("anon_"), f"Expected pseudonym, got: {clean.author}"
        assert "john" not in clean.author.lower()
        assert "doe" not in clean.author.lower()

    def test_author_pseudonym_is_stable(self):
        """Same author name always produces the same pseudonym (deterministic)."""
        item1 = _make_item(author="alice")
        item2 = _make_item(author="alice")
        g = DataResidencyGuard()
        assert g.redact(item1).author == g.redact(item2).author

    # (b) URLs are pseudonymised (PII query params stripped)
    def test_url_pii_query_params_redacted(self):
        """Profile PII query parameters in source_url are replaced (jane.smith removed)."""
        pii_url = "https://example.com/post?user=jane.smith&page=1"
        item = _make_item(source_url=pii_url)
        clean = self.guard.redact(item)

        assert "jane.smith" not in clean.source_url
        # After URL encoding, <redacted> may appear as %3Credacted%3E — either form is valid
        assert "redacted" in clean.source_url.lower()
        # Non-PII param should survive
        assert "page" in clean.source_url

    def test_url_without_pii_params_unchanged(self):
        """URLs without PII query parameters are not modified."""
        safe_url = "https://reddit.com/r/python/comments/abc123"
        item = _make_item(source_url=safe_url)
        clean = self.guard.redact(item)
        assert clean.source_url == safe_url

    # (c) DataResidencyViolationError is raised when bypass is detected
    def test_verify_clean_raises_on_email_in_text(self):
        """verify_clean raises DataResidencyViolationError if email found in raw_text."""
        item = _make_item(raw_text="Contact me at user@example.com for details")
        with pytest.raises(DataResidencyViolationError) as exc_info:
            self.guard.verify_clean(item)
        assert exc_info.value.field == "raw_text"
        assert "email" in exc_info.value.pattern

    def test_verify_clean_raises_on_phone_in_text(self):
        """verify_clean raises DataResidencyViolationError if phone number found in raw_text."""
        item = _make_item(raw_text="Call me at 555-867-5309 anytime")
        with pytest.raises(DataResidencyViolationError) as exc_info:
            self.guard.verify_clean(item)
        assert exc_info.value.field == "raw_text"

    def test_verify_clean_raises_on_pii_url(self):
        """verify_clean raises DataResidencyViolationError if source_url has PII params."""
        item = _make_item(source_url="https://example.com/?username=secret_user")
        with pytest.raises(DataResidencyViolationError) as exc_info:
            self.guard.verify_clean(item)
        assert exc_info.value.field == "source_url"

    # (d) Clean content passes through unchanged
    def test_clean_content_passes_unchanged(self):
        """Content with no PII passes through redact() with minimal changes."""
        item = _make_item(
            author=None,
            raw_text="This product is great, highly recommend it!",
            source_url="https://reddit.com/r/python/comments/clean123",
        )
        clean = self.guard.redact(item)

        assert clean.raw_text == item.raw_text
        assert clean.source_url == item.source_url
        assert clean.author is None

    def test_email_scrubbed_from_raw_text(self):
        """Email addresses in raw_text are replaced with <email_redacted>."""
        item = _make_item(raw_text="Reach out at support@company.com for help.")
        clean = self.guard.redact(item)

        assert "<email_redacted>" in clean.raw_text
        assert "support@company.com" not in clean.raw_text

    def test_redact_is_idempotent(self):
        """Calling redact() twice on the same item produces the same result."""
        item = _make_item(author="jane.doe", raw_text="Email: test@test.com")
        once = self.guard.redact(item)
        twice = self.guard.redact(once)
        assert once.author == twice.author
        assert once.raw_text == twice.raw_text

    def test_verify_clean_passes_on_redacted_item(self):
        """verify_clean does not raise after redact() has processed the item."""
        item = _make_item(
            author="john.doe",
            raw_text="Email me: user@example.com",
            source_url="https://example.com/?user=john",
        )
        clean = self.guard.redact(item)
        # Should NOT raise after full redaction
        self.guard.verify_clean(clean)



# ---------------------------------------------------------------------------
# TeamRole unit tests (Step 4 — competitive_analysis.md §5.5)
# ---------------------------------------------------------------------------

from app.core.models import TeamRole
from app.core.signal_models import TeamDigest


class TestTeamRole:
    """Unit tests for TeamRole privilege model — competitive_analysis.md §5.5."""

    # (a) VIEWER cannot assign (privilege too low)
    def test_viewer_does_not_have_manager_privilege(self):
        """VIEWER role has insufficient privilege for MANAGER-only actions."""
        assert not TeamRole.has_role_at_least(TeamRole.VIEWER, TeamRole.MANAGER)

    def test_analyst_does_not_have_manager_privilege(self):
        """ANALYST role has insufficient privilege for MANAGER-only actions."""
        assert not TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.MANAGER)

    # (b) MANAGER can assign
    def test_manager_has_manager_privilege(self):
        """MANAGER role satisfies MANAGER requirement."""
        assert TeamRole.has_role_at_least(TeamRole.MANAGER, TeamRole.MANAGER)

    def test_manager_also_has_viewer_and_analyst_privileges(self):
        """MANAGER satisfies both VIEWER and ANALYST requirements."""
        assert TeamRole.has_role_at_least(TeamRole.MANAGER, TeamRole.VIEWER)
        assert TeamRole.has_role_at_least(TeamRole.MANAGER, TeamRole.ANALYST)

    def test_viewer_has_viewer_privilege_only(self):
        """VIEWER satisfies VIEWER but not ANALYST or MANAGER."""
        assert TeamRole.has_role_at_least(TeamRole.VIEWER, TeamRole.VIEWER)
        assert not TeamRole.has_role_at_least(TeamRole.VIEWER, TeamRole.ANALYST)

    def test_analyst_has_viewer_and_analyst_privileges(self):
        """ANALYST satisfies VIEWER and ANALYST but not MANAGER."""
        assert TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.VIEWER)
        assert TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.ANALYST)
        assert not TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.MANAGER)

    def test_privilege_levels_are_ordered(self):
        """Privilege levels are strictly ordered: VIEWER < ANALYST < MANAGER."""
        assert TeamRole.privilege_level(TeamRole.VIEWER) < TeamRole.privilege_level(TeamRole.ANALYST)
        assert TeamRole.privilege_level(TeamRole.ANALYST) < TeamRole.privilege_level(TeamRole.MANAGER)

    # (c) TeamDigest returns correct counts
    def test_team_digest_instantiation(self):
        """TeamDigest instantiates correctly with default zero counts."""
        from uuid import uuid4
        from datetime import datetime
        now = datetime.utcnow()
        digest = TeamDigest(
            team_id=uuid4(),
            period_start=now,
            period_end=now,
        )
        assert digest.total_signals == 0
        assert digest.by_status == {}
        assert digest.by_type == {}
        assert digest.unassigned_count == 0
        assert digest.high_urgency_count == 0

    def test_team_digest_with_counts(self):
        """TeamDigest correctly stores by_status and by_type mappings."""
        from uuid import uuid4
        from datetime import datetime
        now = datetime.utcnow()
        digest = TeamDigest(
            team_id=uuid4(),
            period_start=now,
            period_end=now,
            total_signals=10,
            by_status={"new": 7, "acted": 3},
            by_type={"churn_risk": 4, "feature_request": 6},
            unassigned_count=2,
            high_urgency_count=4,
        )
        assert digest.total_signals == 10
        assert digest.by_status["new"] == 7
        assert digest.by_type["churn_risk"] == 4
        assert digest.unassigned_count == 2
        assert digest.high_urgency_count == 4
