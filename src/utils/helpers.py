"""
Cascadia Scorer - Utility Functions

Common utility functions used across the pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def load_config() -> Dict[str, Any]:
    """Load environment variables and configuration"""
    load_dotenv()
    
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "default_vlm": os.getenv("DEFAULT_VLM_PROVIDER", "openai")
    }


def load_scoring_rules() -> Dict[str, Any]:
    """Load scoring rules from JSON configuration"""
    config_path = Path(__file__).parent.parent.parent / "config" / "scoring_rules.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


def get_output_dir() -> Path:
    """Get the output directory, creating if necessary"""
    output_dir = get_project_root() / "data" / "output"
    return ensure_dir(output_dir)


def get_input_dir() -> Path:
    """Get the input directory"""
    return get_project_root() / "data" / "input"


def generate_game_id() -> str:
    """Generate a unique game ID based on timestamp"""
    return f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_json(data: Dict, filepath: Path) -> None:
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict:
    """Load JSON file to dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)


class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger("Timer")
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"Completed: {self.name} in {duration:.2f}s")
