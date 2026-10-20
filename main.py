from database import DatabaseManager
from datasets import IdealFunctions, TestData, TrainingData
from exceptions import DataLoadError, MappingError
from mapper import TestDataMapper
from selector import FunctionSelector
from visualizer import Visualizer


def main():
    """Run the complete program: load, store, select, map and visualise the data."""
    try:
        training = TrainingData("data/train.csv").load()
        ideal = IdealFunctions("data/ideal.csv").load()
        test = TestData("data/test.csv").load()
    except DataLoadError as error:
        print(f"Error while loading the data: {error}")
        return

    database = DatabaseManager("assignment.db")
    database.save_table(training, "training_data")
    database.save_table(ideal, "ideal_functions")

    selection = FunctionSelector(training, ideal).select_all()
    for train_col, (ideal_col, max_dev) in selection.items():
        print(f"{train_col} -> {ideal_col} (max deviation: {max_dev:.4f})")

    try:
        mapping = TestDataMapper(ideal, selection).map_all(test)
    except MappingError as error:
        print(f"Error while mapping the test data: {error}")
        return
    database.save_table(mapping, "test_mapping")
    print(f"{len(mapping)} of {len(test)} test points assigned")

    Visualizer("plots.html").plot(training, ideal, test, selection, mapping)
    print("Plots saved to plots.html")


if __name__ == "__main__":
    main()