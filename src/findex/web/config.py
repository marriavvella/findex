from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    index_path: str = "index.pkl"
    host: str = "0.0.0.0"
    port: int = 8000

settings = Settings()
