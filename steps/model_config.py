class ModelConfig:
    """
    Configuration class for setting and retrieving the model name.

    Attributes:
        model_name (str): Default model name, accessible as a class attribute.

    Methods:
        set_model_name (str): Updates the class-level model name.
    """

    model_name = "LinearRegressionModel"  # Default class attribute

    def __init__(self):
        # Initializes instance with the current class-level model name
        self.model_name = ModelConfig.model_name

    @classmethod
    def set_model_name(cls, name):
        # Updates the class-level model name
        cls.model_name = name
