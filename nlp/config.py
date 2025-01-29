from pathlib import Path
from pydantic import Extra
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource


class NlpSettings(BaseSettings):

    HUGGINGFACE_CACHE_DIR: str

    class Config:
        extra = Extra.allow

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            JsonConfigSettingsSource(settings_cls, json_file=Path(__file__).parent / "config.json"),
            dotenv_settings,
            env_settings,
        )


settings = NlpSettings()
