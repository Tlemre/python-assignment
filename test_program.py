import os
import tempfile
import unittest

import pandas as pd

from database import DatabaseManager
from datasets import Dataset, TrainingData
from exceptions import DataLoadError, MappingError
from mapper import TestDataMapper
from selector import FunctionSelector


class TestDataset(unittest.TestCase):
    """Tests for loading CSV files."""

    def test_load_training_data(self):
        """The training data should have 400 rows and 5 columns."""
        data = TrainingData("data/train.csv").load()
        self.assertEqual(data.shape, (400, 5))

    def test_inheritance(self):
        """TrainingData should be a child class of Dataset."""
        self.assertIsInstance(TrainingData("data/train.csv"), Dataset)

    def test_missing_file_raises_error(self):
        """Loading a missing file should raise a DataLoadError."""
        with self.assertRaises(DataLoadError):
            Dataset("data/does_not_exist.csv").load()


class TestFunctionSelector(unittest.TestCase):
    """Tests for the least squares selection."""

    def setUp(self):
        """Create small example data where the correct answer is known."""
        self.training = pd.DataFrame({"x": [1, 2, 3], "y1": [2.1, 3.9, 6.0]})
        self.ideal = pd.DataFrame({"x": [1, 2, 3], "y1": [1, 2, 3], "y2": [2, 4, 6]})
        self.selector = FunctionSelector(self.training, self.ideal)

    def test_find_best_function(self):
        """y2 (= 2x) should fit the training data best."""
        self.assertEqual(self.selector.find_best_function("y1"), "y2")

    def test_max_deviation(self):
        """The largest deviation between training y1 and ideal y2 should be 0.1."""
        self.assertAlmostEqual(self.selector.max_deviation("y1", "y2"), 0.1)


class TestTestDataMapper(unittest.TestCase):
    """Tests for the sqrt(2) mapping criterion."""

    def setUp(self):
        """Create an ideal function and a selection with a maximum deviation of 0.5."""
        self.ideal = pd.DataFrame({"x": [1.0, 2.0], "y5": [10.0, 20.0]})
        self.mapper = TestDataMapper(self.ideal, {"y1": ("y5", 0.5)})

    def test_point_inside_limit_is_mapped(self):
        """A deviation of 0.6 is below 0.5 * sqrt(2) and should be mapped."""
        function, deviation = self.mapper.map_point(1.0, 10.6)
        self.assertEqual(function, "y5")
        self.assertAlmostEqual(deviation, 0.6)

    def test_point_outside_limit_is_not_mapped(self):
        """A deviation of 3.0 is above the limit and should not be mapped."""
        function, _ = self.mapper.map_point(1.0, 13.0)
        self.assertIsNone(function)

    def test_unknown_x_raises_error(self):
        """An x-value that does not exist should raise a MappingError."""
        with self.assertRaises(MappingError):
            self.mapper.map_point(99.0, 1.0)


class TestDatabaseManager(unittest.TestCase):
    """Tests for storing data in SQLite."""

    def test_save_and_read_table(self):
        """A saved table should be readable with the same content."""
        with tempfile.TemporaryDirectory() as folder:
            database = DatabaseManager(os.path.join(folder, "test.db"))
            data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
            database.save_table(data, "example")
            result = pd.read_sql("example", database.engine)
            self.assertEqual(result.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()