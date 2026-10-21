import pytest

from database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / 'test.sqlite3')
    database.init()
    database.upsert_user(42, 'synthetic', 'Test')
    return database


def test_quota_uses_daily_then_extra_and_rolls_over(db):
    db.add_extra_minutes(42, 2)
    db.deduct_seconds(42, 650, 600, '2026-01-01')
    assert db.remaining_seconds(42, 600, '2026-01-01') == 70
    assert db.balance_info(42, 600, '2026-01-01')['daily_used_seconds'] == 600
    assert db.remaining_seconds(42, 600, '2026-01-02') == 670


def test_insufficient_balance_preserves_balance(db):
    db.ensure_daily_quota(42, '2026-01-01')
    with pytest.raises(RuntimeError, match='Insufficient'):
        db.deduct_seconds(42, 601, 600, '2026-01-01')
    assert db.remaining_seconds(42, 600, '2026-01-01') == 600


def test_user_update_preserves_quota_and_extra_balance(db):
    db.set_extra_balance_minutes(42, 4)
    db.deduct_seconds(42, 100, 600, '2026-01-01')
    db.upsert_user(42, 'renamed', 'Changed')
    assert db.remaining_seconds(42, 600, '2026-01-01') == 740


def test_ban_lifecycle(db):
    assert not db.is_banned(42)
    db.ban_user(42, 'synthetic', 'test', 1)
    assert db.is_banned(42)
    assert db.unban_user(42)
    assert not db.unban_user(42)


def test_initialization_is_repeatable(db):
    db.init()
    assert db.get_user(42)['username'] == 'synthetic'
