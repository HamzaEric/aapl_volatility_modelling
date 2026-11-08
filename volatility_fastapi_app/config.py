from __future__ import annotations

class Settings:
    """Project-wide configuration container."""
    db_name: str = "stocks.sqlite"
    model_directory: str = "models"

settings = Settings()

