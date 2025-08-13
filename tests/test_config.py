"""Tests for config module - Multi-provider support."""
import tempfile
from pathlib import Path

import pytest


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_provider_config_defaults(self):
        """ProviderConfig should have sensible defaults."""
        from tai.config import ProviderConfig

        provider = ProviderConfig()
        assert provider.api_key == ""
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.model == "gpt-4o-mini"


class TestConfigMultiProvider:
    """Tests for Config with multi-provider support."""

    def test_config_has_providers_dict(self):
        """Config should have providers dict field."""
        from tai.config import Config, ProviderConfig

        config = Config()
        assert hasattr(config, "providers")
        assert isinstance(config.providers, dict)

    def test_config_has_default_provider_field(self):
        """Config should have default_provider field."""
        from tai.config import Config

        config = Config()
        assert hasattr(config, "default_provider")
        assert config.default_provider == "openai"

    def test_config_get_provider_returns_default(self):
        """Config.get_provider() should return default provider when no name given."""
        from tai.config import Config, ProviderConfig

        config = Config(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(api_key="sk-test", model="gpt-4"),
            }
        )

        provider = config.get_provider()
        assert provider.model == "gpt-4"

    def test_config_get_provider_by_name(self):
        """Config.get_provider() should return named provider."""
        from tai.config import Config, ProviderConfig

        config = Config(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(api_key="sk-openai", model="gpt-4o-mini"),
                "anthropic": ProviderConfig(
                    api_key="sk-anthropic",
                    base_url="https://api.anthropic.com/v1",
                    model="claude-3-haiku"
                ),
            }
        )

        provider = config.get_provider("anthropic")
        assert provider.model == "claude-3-haiku"
        assert provider.base_url == "https://api.anthropic.com/v1"

    def test_config_get_provider_nonexistent_raises(self):
        """Config.get_provider() should raise ValueError for nonexistent provider."""
        from tai.config import Config, ProviderConfig

        config = Config(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(api_key="sk-test"),
            }
        )

        with pytest.raises(ValueError) as exc_info:
            config.get_provider("nonexistent")

        assert "nonexistent" in str(exc_info.value)


class TestLoadConfigMultiProvider:
    """Tests for load_config with multi-provider TOML."""

    def test_load_config_parses_providers_table(self, tmp_path):
        """load_config() should parse [providers.openai] and [providers.anthropic]."""
        from tai.config import load_config, CONFIG_FILE
        import tomli_w

        config_content = """
default_provider = "anthropic"

[providers.openai]
api_key = "sk-openai"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"

[providers.anthropic]
api_key = "sk-anthropic"
base_url = "https://api.anthropic.com/v1"
model = "claude-3-haiku"

[tools]
directory = "/tmp/tools"
"""

        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)

        # Temporarily patch CONFIG_FILE
        import tai.config as config_module
        original_config_file = config_module.CONFIG_FILE
        config_module.CONFIG_FILE = config_file

        try:
            config = load_config()
            assert config.default_provider == "anthropic"
            assert "openai" in config.providers
            assert "anthropic" in config.providers
            assert config.providers["openai"].model == "gpt-4o-mini"
            assert config.providers["anthropic"].model == "claude-3-haiku"
        finally:
            config_module.CONFIG_FILE = original_config_file

    def test_load_config_defaults_to_openai_if_missing(self, tmp_path):
        """load_config() should default to 'openai' if default_provider missing."""
        from tai.config import load_config
        import tomli_w

        config_content = """
[providers.openai]
api_key = "sk-test"

[tools]
directory = "/tmp/tools"
"""

        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)

        import tai.config as config_module
        original_config_file = config_module.CONFIG_FILE
        config_module.CONFIG_FILE = config_file

        try:
            config = load_config()
            assert config.default_provider == "openai"
        finally:
            config_module.CONFIG_FILE = original_config_file


class TestGetDefaultConfigDict:
    """Tests for get_default_config_dict with providers structure."""

    def test_get_default_config_dict_has_providers(self):
        """get_default_config_dict() should return providers table."""
        from tai.config import get_default_config_dict

        default = get_default_config_dict()

        assert "default_provider" in default
        assert default["default_provider"] == "openai"
        assert "providers" in default
        assert isinstance(default["providers"], dict)
        assert "openai" in default["providers"]

    def test_get_default_config_dict_provider_structure(self):
        """get_default_config_dict() providers should have correct structure."""
        from tai.config import get_default_config_dict

        default = get_default_config_dict()
        openai = default["providers"]["openai"]

        assert "api_key" in openai
        assert "base_url" in openai
        assert "model" in openai
        assert openai["base_url"] == "https://api.openai.com/v1"
        assert openai["model"] == "gpt-4o-mini"
