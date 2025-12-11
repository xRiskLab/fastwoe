"""
Model performance metrics and visualization.

Implements CAP curves (binary) and Power curves (continuous targets like LGD).
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import expit, logit

from .fast_somersd import somersd_yx


def plot_performance(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series, list],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 5),
    dpi: int = 100,
    show_plot: bool = True,
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
) -> tuple:
    """
    Plot model performance curve (CAP for binary, Power curve for continuous).

    Automatically detects target type and creates appropriate visualization.
    Supports multiple predictions for model comparison.

    Args:
        y_true: True target values (binary 0/1 or continuous)
        y_pred: Predicted scores/probabilities. Can be:
            - Single array: one model
            - List of arrays: multiple models for comparison
        weights: Optional weights (e.g., EAD for LGD models)
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size as (width, height). Only used if ax is None.
        dpi: Figure resolution. Only used if ax is None.
        show_plot: Whether to display the plot. Only used if ax is None.
        labels: Optional labels for multiple predictions. If None, uses "Model 1", "Model 2", etc.
        colors: Optional colors for multiple predictions. If None, uses default colormap.

    Returns:
        Tuple of (figure, axes, gini_coefficient(s))
        - gini is a single float if y_pred is single array
        - gini is a list of floats if y_pred is a list of arrays

    Examples:
        >>> # Single model
        >>> fig, ax, gini = plot_performance(y_true, y_pred)

        >>> # Compare multiple models
        >>> fig, ax, ginis = plot_performance(
        ...     y_true,
        ...     [y_pred1, y_pred2, y_pred3],
        ...     labels=['Model A', 'Model B', 'Model C']
        ... )

        >>> # Side-by-side comparison with custom grid
        >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        >>> _, _, gini1 = plot_performance(y_test, y_pred, ax=ax1)
        >>> _, _, gini2 = plot_performance(y_test, y_pred, weights=ead, ax=ax2)
        >>> plt.tight_layout()
        >>> plt.show()
    """
    # Convert to numpy
    y_true = np.asarray(y_true)

    # Handle multiple predictions
    if isinstance(y_pred, list):
        y_preds = [np.asarray(yp) for yp in y_pred]
        is_multiple = True
    else:
        y_preds = [np.asarray(y_pred)]
        is_multiple = False

    # Setup labels and colors
    if labels is None:
        if is_multiple:
            labels = [f"Model {i + 1}" for i in range(len(y_preds))]
        else:
            labels = ["Model"]

    if colors is None:
        # Default colormap
        default_colors = [
            "#69db7c",
            "#55d3ed",
            "#ffa94d",
            "#c430c1",
            "#ff6b6b",
            "#4dabf7",
        ]
        colors = default_colors[: len(y_preds)]

    # Keep track of whether weights were provided
    has_weights = weights is not None
    weights = np.asarray(weights) if weights is not None else np.ones_like(y_true)

    # Auto-detect binary vs continuous
    is_binary = np.all(np.isin(y_true, [0, 1]))

    # Calculate Gini for all predictions using fast Somers' D
    ginis = []
    for yp in y_preds:
        g = somersd_yx(y_true, yp, weights if has_weights else None).statistic
        ginis.append(g)

    # Create figure or use provided axes
    if ax is None:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        ax1 = ax
        fig = ax1.get_figure()

    # CAP curve for binary and continuous targets
    n = len(y_true)
    n_events = y_true.sum()

    # Plot random line
    ax1.plot(
        [0, 1],
        [0, 1],
        linestyle="dotted",
        color="black",
        alpha=0.5,
    )

    # Plot perfect/ideal line
    if is_binary:
        # Binary: perfect line is step function at bad_rate
        bad_rate = n_events / n
        ax1.plot(
            [0, bad_rate, 1],
            [0, 1, 1],
            color="dodgerblue",
            label="Crystal Ball",
        )
    else:
        # Continuous: perfect line is sorting by TRUE target values
        perfect_idx = np.argsort(y_true)[::-1]
        y_perfect = y_true[perfect_idx]
        cum_pop_perfect = np.concatenate([[0], np.arange(1, n + 1) / n])
        cum_events_perfect = np.concatenate([[0], np.cumsum(y_perfect) / n_events])
        ax1.plot(
            cum_pop_perfect,
            cum_events_perfect,
            color="dodgerblue",
            label="Crystal Ball",
        )

    # Plot each model (same for both binary and continuous)
    for yp, label, color, g in zip(y_preds, labels, colors, ginis):
        sort_idx = np.argsort(yp)[::-1]
        y_sorted = y_true[sort_idx]

        cum_pop = np.concatenate([[0], np.arange(1, n + 1) / n])
        cum_events = np.concatenate([[0], np.cumsum(y_sorted) / n_events])

        ax1.plot(
            cum_pop,
            cum_events,
            color=color,
            label=f"{label} AR: {g:.2%}",
        )

    # Styling (same for both)
    ax1.set_xlabel("Fraction of population", fontfamily="Arial", fontsize=12)
    ax1.set_ylabel("Fraction of target", fontfamily="Arial", fontsize=12)
    ax1.set_title("Cumulative Accuracy Profile (CAP)", fontsize=14, fontfamily="Arial")
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, which="both", linestyle="dotted", linewidth=0.7, alpha=0.6)
    # Only apply tight_layout and show if we created the figure
    if ax is None:
        plt.tight_layout()
        if show_plot:
            plt.show()

    # Return single gini or list depending on input
    gini_out = ginis if is_multiple else ginis[0]
    return fig, ax1, gini_out


def visualize_woe(
    woe_encoder,
    feature_name: str,
    mode: Literal["probability", "log_odds"] = "probability",
    figsize: tuple = (10, None),
    color_positive: str = "#F783AC",
    color_negative: str = "#A4D8FF",
    show_plot: bool = True,
) -> pd.DataFrame:
    """Visualize Weight of Evidence (WOE) transformation for a feature."""
    # Get WOE mappings
    try:
        mappings = woe_encoder.get_all_mappings()[feature_name]
    except (AttributeError, KeyError) as e:
        raise ValueError(
            f"Could not get WOE mappings for feature '{feature_name}'. Error: {e}"
        ) from e

    frame = pd.DataFrame(mappings)[["category", "woe"]]

    try:
        prior = woe_encoder.y_prior_
    except AttributeError as exc:
        raise ValueError("WOE encoder must have 'y_prior_' attribute.") from exc

    if mode == "probability":
        frame["proba"] = expit(logit(prior) + frame["woe"])
        frame["proba_delta"] = frame["proba"] - prior
        value_col = "proba_delta"
        baseline = prior
        xlabel = "Default Probability"

        def value_formatter(x, p):
            """Value formatter for probability."""
            return f"{x:.0%}"
    else:
        frame["log_odds"] = logit(prior) + frame["woe"]
        frame["log_odds_delta"] = frame["woe"]
        value_col = "log_odds_delta"
        baseline = logit(prior)
        xlabel = "Log-Odds"

        def value_formatter(x, p):
            """Value formatter for log-odds."""
            return f"{x:.2f}"

    try:
        frame = frame.sort_values("category")
    except TypeError:
        frame = frame.sort_values(value_col)

    width, height = figsize
    if height is None:
        height = max(3, len(frame) * 0.3)

    fig, ax = plt.subplots(figsize=(width, height))
    colors = [color_positive if x >= 0 else color_negative for x in frame[value_col]]

    if mode == "probability":
        ax.barh(
            range(len(frame)),
            frame[value_col],
            left=baseline,
            color=colors,
            height=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
    else:
        ax.barh(
            range(len(frame)),
            frame[value_col],
            color=colors,
            height=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["category"])

    if mode == "probability":
        ax.axvline(
            baseline,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"Average: {baseline:.1%}",
            zorder=10,
        )
    else:
        ax.axvline(
            0, color="black", linewidth=1.5, linestyle="--", label="Baseline", zorder=10
        )

    ax.xaxis.set_major_formatter(plt.FuncFormatter(value_formatter))
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(f"WOE: {feature_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    plt.tight_layout()
    if show_plot:
        plt.show()

    if mode == "probability":
        return frame[["category", "woe", "proba", "proba_delta"]]
    else:
        return frame[["category", "woe", "log_odds", "log_odds_delta"]]
