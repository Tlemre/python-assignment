import pandas as pd
import math
from exceptions import MappingError


class TestDataMapper:
    """Maps test points to the chosen ideal functions using the sqrt(2) criterion."""

    def __init__(self, ideal_functions, selection):
        """Store the ideal functions and the result of the function selection."""
        self.ideal_functions = ideal_functions
        self.selection = selection

    def map_point(self, x, y):
        """Map a single test point to the best matching ideal function."""
        row = self.ideal_functions[(self.ideal_functions["x"] - x).abs() < 1e-9]
        if row.empty:
            raise MappingError(f"x-value {x} not found in the ideal functions.")
        best_function = None
        best_deviation = None
        for ideal_column, max_dev in self.selection.values():
            deviation = abs(y - row[ideal_column].iloc[0])
            if deviation <= max_dev * math.sqrt(2):
                if best_deviation is None or deviation < best_deviation:
                    best_deviation = deviation
                    best_function = ideal_column
        return best_function, best_deviation

    def map_all(self, test_data):
        """Map all test points line by line and return the results as a DataFrame."""
        results = []
        for _, point in test_data.iterrows():
            function, deviation = self.map_point(point["x"], point["y"])
            if function is not None:
                results.append({
                    "X (test func)": point["x"],
                    "Y (test func)": point["y"],
                    "Delta Y (test func)": deviation,
                    "No. of ideal func": function,
                })
        return pd.DataFrame(results)