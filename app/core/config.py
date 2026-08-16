from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "POLARIS API"
    app_version: str = "1.3.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8081,http://127.0.0.1:8081"
    )

    supabase_url: str = ""
    supabase_jwt_secret: str = ""
    supabase_jwt_audience: str = "authenticated"
    auth_disabled: bool = True

    database_url: str = "sqlite+aiosqlite:///./data/polaris.db"
    polaris_db_path: str = ""

    llm_provider: str = "qwen"
    huggingface_api_key: str = ""

    literature_mcp_endpoint: str = "http://127.0.0.1:8000/mcp"
    watcher_port: int = 8765

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
