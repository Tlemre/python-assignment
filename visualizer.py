from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


class Visualizer:
    """Creates interactive plots of the data and the mapping results with Bokeh."""

    def __init__(self, output_path):
        """Store the path of the HTML file for the plots."""
        self.output_path = output_path

    def _ideal_value(self, ideal_functions, column, x):
        """Return the value of an ideal function at a given x-value."""
        row = ideal_functions[(ideal_functions["x"] - x).abs() < 1e-9]
        return row[column].iloc[0]

    def _function_plot(self, training_data, ideal_functions, train_col, ideal_col, mapping, color):
        """Create one plot for a training function, its ideal function and its mapped test points."""
        plot = figure(width=550, height=400, title=f"{train_col} -> ideal function {ideal_col}",
                      x_axis_label="x", y_axis_label="y")
        plot.scatter(training_data["x"], training_data[train_col], size=3, color="lightgray",
                     legend_label="Training data")
        plot.line(ideal_functions["x"], ideal_functions[ideal_col], line_width=2, color=color,
                  legend_label=f"Ideal function {ideal_col}")
        points = mapping[mapping["No. of ideal func"] == ideal_col]
        source = ColumnDataSource(data={
            "x": points["X (test func)"],
            "y": points["Y (test func)"],
            "ideal_y": [self._ideal_value(ideal_functions, ideal_col, x) for x in points["X (test func)"]],
            "deviation": points["Delta Y (test func)"],
        })
        plot.segment(x0="x", y0="y", x1="x", y1="ideal_y", source=source, color=color, line_dash="dashed")
        renderer = plot.scatter(x="x", y="y", source=source, size=9, color=color,
                                legend_label="Mapped test points")
        plot.add_tools(HoverTool(renderers=[renderer],
                                 tooltips=[("x", "@x"), ("y", "@y"), ("Deviation", "@deviation")]))
        plot.legend.location = "top_left"
        return plot

    def _overview_plot(self, test_data, mapping, colors):
        """Create an overview plot of all test points, coloured by their assigned function."""
        plot = figure(width=550, height=400, title="All test points", x_axis_label="x", y_axis_label="y")
        merged = test_data.merge(mapping, left_on=["x", "y"], right_on=["X (test func)", "Y (test func)"],
                                 how="left")
        unassigned = merged[merged["No. of ideal func"].isna()]
        plot.scatter(unassigned["x"], unassigned["y"], size=6, color="gray", legend_label="Not assigned")
        for ideal_col, color in colors.items():
            assigned = merged[merged["No. of ideal func"] == ideal_col]
            plot.scatter(assigned["x"], assigned["y"], size=6, color=color, legend_label=ideal_col)
        plot.legend.location = "top_left"
        return plot

    def plot(self, training_data, ideal_functions, test_data, selection, mapping):
        """Create all plots and save them as an HTML file."""
        output_file(self.output_path, title="Ideal function mapping")
        plots = []
        colors = {}
        for i, (train_col, (ideal_col, _)) in enumerate(selection.items()):
            colors[ideal_col] = COLORS[i]
            plots.append(self._function_plot(training_data, ideal_functions, train_col, ideal_col,
                                             mapping, COLORS[i]))
        plots.append(self._overview_plot(test_data, mapping, colors))
        save(gridplot(plots, ncols=2))