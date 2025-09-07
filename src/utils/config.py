import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
import yaml

class Config(BaseSettings):
    """Application configuration"""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8003, env="API_PORT") 
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database Configuration
    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Model Configuration
    model_path: str = Field(default="./models/saved/", env="MODEL_PATH")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    
    # Bayesian Sampling Configuration
    n_samples: int = Field(default=2000, env="N_SAMPLES")
    n_tune: int = Field(default=1000, env="N_TUNE")
    n_cores: int = Field(default=2, env="N_CORES")
    
    # Survival Analysis Configuration
    survival_timepoints: list = Field(default=[30, 60, 90, 180, 365])
    significance_level: float = Field(default=0.05)
    
    # Risk Scoring Configuration
    risk_thresholds: Dict[str, float] = Field(default={
        'low': 0.3,
        'medium': 0.6, 
        'high': 0.8,
        'critical': 0.95
    })
    
    # Security Configuration
    secret_key: str = Field(env="SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # Performance Configuration
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def get_config() -> Config:
    """Get application configuration"""
    return Config()

def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model-specific configuration from YAML"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}
