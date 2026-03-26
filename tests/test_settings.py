from video_autocut.settings import Settings, get_settings


def test_settings_defaults():
    settings = Settings(openai_api_key="test-key")
    assert settings.model_name == "openai:gpt-4o-mini"
    assert settings.log_level == "INFO"
    assert settings.ffmpeg_path is None


def test_get_settings_returns_cached_instance():
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
