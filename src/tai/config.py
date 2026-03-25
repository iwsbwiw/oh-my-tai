"""oh-my-tai 的配置管理。"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 根据 Python 版本选择 TOML 解析实现
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

# 配置文件路径
CONFIG_DIR = Path.home() / ".tai"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class ProviderConfig:
    """LLM 提供方配置。"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"


@dataclass
class ToolsConfig:
    """工具目录配置。"""
    directory: str = field(default_factory=lambda: str(CONFIG_DIR / "tools"))


@dataclass
class Config:
    """主配置对象。"""
    default_provider: str = "openai"
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    def get_provider(self, name: Optional[str] = None) -> ProviderConfig:
        """按名称或默认项获取 provider 配置。

        Args:
            name: 要获取的 provider 名称；为 ``None`` 时使用默认 provider。

        Returns:
            对应的 ``ProviderConfig``。

        Raises:
            ValueError: 指定的 provider 未配置时抛出。
        """
        provider_name = name or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not configured")
        return self.providers[provider_name]


def get_default_config_dict() -> dict:
    """以字典形式返回默认配置。"""
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
    """在配置目录或配置文件不存在时自动创建。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "tools").mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        default_config = get_default_config_dict()
        with open(CONFIG_FILE, "wb") as f:
            tomli_w.dump(default_config, f)
        print(f"Created default config at {CONFIG_FILE}")

    return CONFIG_FILE


def load_config() -> Config:
    """从文件加载配置。"""
    config_path = ensure_config_exists()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    default_provider = data.get("default_provider", "openai")
    providers_data = data.get("providers", {})
    tools_data = data.get("tools", {})

    # 组装 providers 配置字典
    providers = {}
    for name, provider_config in providers_data.items():
        providers[name] = ProviderConfig(**provider_config)

    return Config(
        default_provider=default_provider,
        providers=providers,
        tools=ToolsConfig(**tools_data),
    )


def get_tools_directory() -> Path:
    """获取展开后的工具目录路径。"""
    config = load_config()
    return Path(config.tools.directory).expanduser()
