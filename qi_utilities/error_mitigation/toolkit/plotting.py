"""
Utility functions for plotting.

Authors: Jan Hemink
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.axes

COLOURS10 = plt.get_cmap("tab10").colors  # pyright: ignore[reportAttributeAccessIssue]
COLOURS20 = plt.get_cmap("tab20").colors  # pyright: ignore[reportAttributeAccessIssue]


def set_colour_cycle_10(ax: matplotlib.axes.Axes) -> None:
    ax.set_prop_cycle("color", COLOURS10)


def set_colour_cycle_20(ax: matplotlib.axes.Axes) -> None:
    ax.set_prop_cycle("color", COLOURS20)


def plot_multi_bar(
    ax: matplotlib.axes.Axes,
    x_labels: Sequence[str],
    data: dict[str, Sequence[float | int]],
    yerr: dict[str, Sequence[float | int]] = {},
    width: float = 0.7,
    bar_label: bool = False,
    bar_label_kwargs: dict = {"padding": 3},
    **kwargs,
) -> None:
    """
    Makes a bar plot with multiple bars next to each other for each x-axis label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes on which to plot the bar chart.
    x_labels : Sequence[str | int]
        The (x-axis) labels of the different bars.
    data : dict[str, Sequence[float | int]]
        The data for the multiple bars. Dictionary from which the keys are used as the legend labels
        and the values a `Sequence` containing the plotted data for each of the `x_labels`.
    yerr : dict[str, Sequence[float | int]]
        Dictionary containing vertical error bar values. Keys used for the `yerr` dictionary should
        also be present in the `data` dictionary. When present in the `data` dictionary but not the
        `yerr` dictionary, that specific bar will be plotted without error bars. By default an empty
        dictionary, which results in no error bars being plotted.
    width : float, optional
        Width of the bars, how much of the x-axis is covered by the bars. Should be larger than 0
        (otherwise bars won't be plotted), and at most equal to 1 (bars will overlap when width > 1).
        When equal to 1 there won't be any whitespace between the bars. Defaults to 0.7
    bar_label : bool, optional
        Flag to plot labels with the value above each bar, by default False
    bar_label_kwargs : dict, optional
        Keyword arguments passed along to `matplotlib.pyplot.bar_label`, by default {"padding": 3}
    kwargs : optional
        Keyword arguments passed along to `matplotlib.pyplot.bar`.

    See Also
    --------
    `matplotlib.pyplot.bar` : For how the plotting works and optional keyword arguments.
    """
    num_bars = len(data)
    width = width / num_bars
    x = np.arange(len(x_labels))
    for idx, (label, values) in enumerate(data.items()):
        offset = width * idx
        bar = ax.bar(x + offset, values, width, label=label, yerr=yerr.get(label), **kwargs)
        if bar_label:
            ax.bar_label(bar, **bar_label_kwargs)
    ax.set_xticks(x + width * (0.5 * (num_bars - 1)), x_labels)
