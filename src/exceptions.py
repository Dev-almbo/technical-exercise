# Best practice is custom exceptions to really trace what went wrong were


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelError(Exception):
    """Custom exception for model-related errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
