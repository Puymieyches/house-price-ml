import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///mlflow/mlflow.db')
    API_KEY = os.getenv('API_KEY')
    MODEL_REGISTRY_URI = os.getenv('MODEL_REGISTRY_URI', './models')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
    @classmethod
    def validate(cls):
        """Validate that all required config is present."""
        required = ['DATABASE_URL']
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing required config: {var}")
            
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'

config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

def get_config():
    env = os.getenv('ENVIRONMENT', 'development')
    return config_map[env]
