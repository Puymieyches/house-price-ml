'''Example Logging usage in code'''
from src.utils.logging import setup_logging

logger = setup_logging(__name__)

def train_model(data):
    logger.info("Starting model training")
    try:
        # Training code here
        logger.info(f"Model training completed successfully")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise
    