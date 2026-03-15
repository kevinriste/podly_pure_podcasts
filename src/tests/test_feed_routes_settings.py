from types import SimpleNamespace
from typing import Any
from unittest import mock

from app.extensions import db
from app.models import Feed
from app.routes.feed_routes import feed_bp


def test_update_feed_settings_accepts_llm_chapter_fallback_override(app):
    app.testing = True
    app.register_blueprint(feed_bp)

    with app.app_context():
        feed = Feed(title="Settings Feed", rss_url="https://example.com/feed.xml")
        db.session.add(feed)
        db.session.commit()
        feed_id = feed.id

    client = app.test_client()

    def _writer_update_side_effect(
        model_name: str, model_id: int, updates: dict[str, Any], wait: bool = True
    ):
        assert model_name == "Feed"
        assert model_id == feed_id
        assert wait is True
        Feed.query.filter_by(id=model_id).update(updates)
        db.session.commit()
        return SimpleNamespace(success=True)

    with mock.patch("app.routes.feed_routes.writer_client") as mock_writer:
        mock_writer.update.side_effect = _writer_update_side_effect
        response = client.patch(
            f"/api/feeds/{feed_id}/settings",
            json={"enable_llm_chapter_fallback_tagging": False},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["enable_llm_chapter_fallback_tagging"] is False


def test_update_feed_settings_accepts_null_llm_chapter_fallback_override(app):
    app.testing = True
    app.register_blueprint(feed_bp)

    with app.app_context():
        feed = Feed(
            title="Settings Feed",
            rss_url="https://example.com/feed.xml",
            enable_llm_chapter_fallback_tagging=False,
        )
        db.session.add(feed)
        db.session.commit()
        feed_id = feed.id

    client = app.test_client()

    def _writer_update_side_effect(
        model_name: str, model_id: int, updates: dict[str, Any], wait: bool = True
    ):
        assert model_name == "Feed"
        assert model_id == feed_id
        assert wait is True
        assert updates["enable_llm_chapter_fallback_tagging"] is None
        Feed.query.filter_by(id=model_id).update(updates)
        db.session.commit()
        return SimpleNamespace(success=True)

    with mock.patch("app.routes.feed_routes.writer_client") as mock_writer:
        mock_writer.update.side_effect = _writer_update_side_effect
        response = client.patch(
            f"/api/feeds/{feed_id}/settings",
            json={"enable_llm_chapter_fallback_tagging": None},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["enable_llm_chapter_fallback_tagging"] is None


def test_update_feed_settings_forces_chapter_fallback_for_chapter_insert(app):
    app.testing = True
    app.register_blueprint(feed_bp)

    with app.app_context():
        feed = Feed(title="Settings Feed", rss_url="https://example.com/feed.xml")
        db.session.add(feed)
        db.session.commit()
        feed_id = feed.id

    client = app.test_client()

    def _writer_update_side_effect(
        model_name: str, model_id: int, updates: dict[str, Any], wait: bool = True
    ):
        assert model_name == "Feed"
        assert model_id == feed_id
        assert wait is True
        assert updates["ad_detection_strategy"] == "chapter_insert"
        assert updates["enable_llm_chapter_fallback_tagging"] is True
        Feed.query.filter_by(id=model_id).update(updates)
        db.session.commit()
        return SimpleNamespace(success=True)

    with mock.patch("app.routes.feed_routes.writer_client") as mock_writer:
        mock_writer.update.side_effect = _writer_update_side_effect
        response = client.patch(
            f"/api/feeds/{feed_id}/settings",
            json={
                "ad_detection_strategy": "chapter_insert",
                "enable_llm_chapter_fallback_tagging": False,
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["ad_detection_strategy"] == "chapter_insert"
    assert payload["enable_llm_chapter_fallback_tagging"] is True
