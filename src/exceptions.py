class ConfigurationError(BaseException):
    """Custom exception for configuration errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelError(BaseException):
    """Custom exception for model-related errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
