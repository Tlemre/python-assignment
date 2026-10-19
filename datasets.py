import pandas as pd
from exceptions import DataLoadError


class Dataset:
    """Base class for loading data from a CSV file."""

    def __init__(self, file_path):
        """Store the path of the CSV file."""
        self.file_path = file_path
        self.data = None

    def load(self):
        """Load the CSV file into a pandas DataFrame."""
        try:
            self.data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise DataLoadError(f"File not found: {self.file_path}")
        return self.data


class TrainingData(Dataset):
      """Training data with four training functions."""
      pass


class IdealFunctions(Dataset):
      """Data of the 50 ideal functions."""
      pass


class TestData(Dataset):
      """Test data that will be mapped to the ideal functions."""
      pass
