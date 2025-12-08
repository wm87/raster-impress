import pytest
from pathlib import Path

@pytest.fixture
def data_dir() -> Path:
    """Verzeichnis für dauerhaft gespeicherte Testdaten."""
    path = Path(__file__).parent / "test_data"
    path.mkdir(exist_ok=True)
    return path