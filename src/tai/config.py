"""Configuration management for oh-my-tai."""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Conditional import for TOML support
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

# Config file locations
CONFIG_DIR = Path.home() / ".tai"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class ProviderConfig:
    """LLM provider configuration."""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"


@dataclass
class ToolsConfig:
    """Tools directory configuration."""
    directory: str = field(default_factory=lambda: str(CONFIG_DIR / "tools"))


@dataclass
class Config:
    """Main configuration container."""
    default_provider: str = "openai"
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    def get_provider(self, name: Optional[str] = None) -> ProviderConfig:
        """Get provider config by name or default.

        Args:
            name: Provider name to get, or None for default provider.

        Returns:
            ProviderConfig for the requested provider.

        Raises:
            ValueError: If the provider is not configured.
        """
        provider_name = name or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not configured")
        return self.providers[provider_name]


def get_default_config_dict() -> dict:
    """Return default configuration as dictionary."""
    return {
        "default_provider": "openai",
        "providers": {
            "openai": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
            }
        },
        "tools": {
            "directory": str(CONFIG_DIR / "tools"),
        },
    }


def ensure_config_exists() -> Path:
    """Create config directory and file if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "tools").mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        default_config = get_default_config_dict()
        with open(CONFIG_FILE, "wb") as f:
            tomli_w.dump(default_config, f)
        print(f"Created default config at {CONFIG_FILE}")

    return CONFIG_FILE


def load_config() -> Config:
    """Load configuration from file."""
    config_path = ensure_config_exists()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    default_provider = data.get("default_provider", "openai")
    providers_data = data.get("providers", {})
    tools_data = data.get("tools", {})

    # Build providers dict
    providers = {}
    for name, provider_config in providers_data.items():
        providers[name] = ProviderConfig(**provider_config)

    return Config(
        default_provider=default_provider,
        providers=providers,
        tools=ToolsConfig(**tools_data),
    )


def get_tools_directory() -> Path:
    """Get expanded tools directory path."""
    config = load_config()
    return Path(config.tools.directory).expanduser()
