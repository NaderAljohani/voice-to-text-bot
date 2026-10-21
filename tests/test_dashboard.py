import dashboard_server as dashboard


def test_dashboard_settings_never_returns_token_or_admin_ids(monkeypatch, tmp_path):
    (tmp_path / '.env').write_text('TELEGRAM_BOT_TOKEN=synthetic\nADMIN_TELEGRAM_IDS=42\nSTORE_AUDIO_FILES=false\n', encoding='utf-8')
    monkeypatch.setattr(dashboard, 'BASE_DIR', tmp_path)
    assert dashboard.api_settings_safe() == {'settings': {'STORE_AUDIO_FILES': 'false'}}


def test_media_access_restricted_to_retained_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, 'BASE_DIR', tmp_path)
    retained = tmp_path / 'data' / 'stored_audio'
    retained.mkdir(parents=True)
    audio = retained / 'synthetic.wav'
    audio.write_bytes(b'synthetic')
    outside = tmp_path / 'private.wav'
    outside.write_bytes(b'synthetic')
    secret = retained / 'private.env'
    secret.write_text('synthetic')
    assert dashboard.safe_stored_media_path(str(audio)) == audio.resolve()
    assert dashboard.safe_stored_media_path(str(outside)) is None
    assert dashboard.safe_stored_media_path(str(secret)) is None
