class DataLoadError(Exception):
     """Raised when a CSV file cannot be loaded or has an invalid format."""
     pass

class MappingError(Exception):
     """Raised when a test point cannot be mapped to an ideal function."""
     pass

