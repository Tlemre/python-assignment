class FunctionSelector:
    """Selects the best-fitting ideal functions using the least squares method."""

    def __init__(self, training_data, ideal_functions):
        """Store the training data and the ideal functions as DataFrames."""
        self.training_data = training_data
        self.ideal_functions = ideal_functions

    def find_best_function(self, training_column):
        """Return the ideal function with the smallest sum of squared deviations."""
        best_function = None
        best_sum = float("inf")
        for column in self.ideal_functions.columns[1:]:
            current_sum = ((self.training_data[training_column] - self.ideal_functions[column]) ** 2).sum()
            if current_sum < best_sum:
                best_sum = current_sum
                best_function = column
        return best_function

    def max_deviation(self, training_column, ideal_column):
        """Return the largest absolute deviation between training data and ideal function."""
        return (self.training_data[training_column] - self.ideal_functions[ideal_column]).abs().max()

    def select_all(self):
        """Select the best ideal function and its maximum deviation for each training column."""
        results = {}
        for column in self.training_data.columns[1:]:
            best = self.find_best_function(column)
            results[column] = (best, self.max_deviation(column, best))
        return results