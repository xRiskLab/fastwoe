"""
Plotting functionality for FastWoe.

This module provides visualization functions for model performance and WOE analysis.
Requires matplotlib (install with: pip install fastwoe[plotting]).
"""

from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.special import expit, logit

from .metrics import somersd_yx

# Set Arial as global font
rcParams["font.family"] = "Arial"


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
    ax1.set_xlabel("Fraction of population", fontsize=12)
    ax1.set_ylabel("Fraction of target", fontsize=12)
    ax1.set_title("Cumulative Accuracy Profile (CAP)", fontsize=14)
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
    feature_name: Optional[str] = None,
    explanation: Optional[dict] = None,
    mode: Literal["proba", "logit"] = "proba",
    figsize: tuple = (10, None),
    show_plot: bool = True,
) -> pd.DataFrame:
    """Visualize Weight of Evidence (WOE) transformation effects.

    Creates either a feature-level WOE curve (showing WOE values across bins) or
    a prediction-level waterfall chart (showing feature contributions to a single prediction).

    Args:
        woe_encoder: WOE encoder with get_all_mappings() and y_prior_ attributes.
            Typically a FastWoe instance.
        feature_name: Feature to visualize for feature-level WOE curve.
            Required when explanation is None. Ignored when explanation is provided.
        explanation: Prediction-level explanation dictionary with 'feature_contributions' key.
            When provided, creates waterfall chart and ignores feature_name.
        mode: Output scale for values. Options:
            - "proba": Probability scale (0-1)
            - "logit": Log-odds scale
        figsize: Figure dimensions as (width, height) tuple. Default is (10, 6).
        show_plot: If True, displays the plot immediately. Default is True.

    Returns:
        pd.DataFrame: Visualization data with columns depending on mode:
            - Feature-level: bin labels, WOE values, frequencies
            - Prediction-level: feature names, contribution values

    Raises:
        ValueError: If neither feature_name nor explanation is provided, or if
            feature_name not found in encoder mappings.

    Examples:
        >>> # Feature-level visualization
        >>> visualize_woe(encoder, feature_name="channel", mode="proba")

        >>> # Prediction-level visualization
        >>> from fastwoe import WeightOfEvidence
        >>> explainer = WeightOfEvidence(encoder, X_train, y_train)
        >>> explanation = explainer.explain(X_test.iloc[0])
        >>> visualize_woe(encoder, explanation=explanation, mode="proba")
    """
    # Determine if we're doing feature-level or prediction-level
    is_prediction_level = explanation is not None

    if is_prediction_level:
        # Prediction-level visualization
        if "feature_contributions" not in explanation:
            raise ValueError("explanation dict must contain 'feature_contributions' key")

        feature_contributions = explanation["feature_contributions"]
        frame = pd.DataFrame(
            [{"feature": feat, "woe": woe_val} for feat, woe_val in feature_contributions.items()]
        )

        try:
            prior = woe_encoder.y_prior_
        except AttributeError as exc:
            raise ValueError("WOE encoder must have 'y_prior_' attribute.") from exc

        if mode == "proba":
            frame["proba"] = expit(logit(prior) + frame["woe"])
            frame["proba_delta"] = frame["proba"] - prior
            value_col = "proba_delta"
            baseline = prior
            xlabel = "Probability of default"
            title = "Feature contributions to prediction"

            def value_formatter(x, p):
                """Value formatter for probability deltas (bar labels)."""
                return f"{x:+.0%}"

            def axis_formatter(x, p):
                """Axis formatter for absolute probability values."""
                return f"{x:.0%}"
        else:  # mode == "logit"
            frame["logit"] = logit(prior) + frame["woe"]
            frame["logit_delta"] = frame["woe"]
            value_col = "logit_delta"
            baseline = logit(prior)
            xlabel = "Logit"
            title = "Feature contributions to prediction"

            def value_formatter(x, p):
                """Value formatter for logit deltas (bar labels)."""
                return f"{x:+.2f}"

            def axis_formatter(x, p):
                """Axis formatter for absolute logit values."""
                return f"{x:.2f}"

        # Sort by absolute value for better visualization
        frame = frame.reindex(frame[value_col].abs().sort_values(ascending=False).index)
        label_col = "feature"

    else:
        # Feature-level visualization (original functionality)
        if feature_name is None:
            raise ValueError("feature_name is required when explanation is not provided")

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

        if mode == "proba":
            frame["proba"] = expit(logit(prior) + frame["woe"])
            frame["proba_delta"] = frame["proba"] - prior
            value_col = "proba_delta"
            baseline = prior
            xlabel = "Default Probability"
            title = f"WOE: {feature_name}"

            def value_formatter(x, p):
                """Value formatter for probability deltas (bar labels)."""
                return f"{x:.0%}"

            def axis_formatter(x, p):
                """Axis formatter for absolute probability values."""
                return f"{x:.0%}"
        else:  # mode == "logit"
            frame["logit"] = logit(prior) + frame["woe"]
            frame["logit_delta"] = frame["woe"]
            value_col = "logit_delta"
            baseline = logit(prior)
            xlabel = "Logit"
            title = f"WOE: {feature_name}"

            def value_formatter(x, p):
                """Value formatter for logit deltas (bar labels)."""
                return f"{x:.2f}"

            def axis_formatter(x, p):
                """Axis formatter for absolute logit values."""
                return f"{x:.2f}"

        try:
            frame = frame.sort_values("category")
        except TypeError:
            frame = frame.sort_values(value_col)

        label_col = "category"

    # Define colormap
    colormap = [
        "#CFD4D9",
        "#87D886",
        "#E889AB",
        "#AFD7FB",
        "#F5A623",
        "#9B59B6",
        "#1ABC9C",
        "#E74C3C",
    ]

    width, height = figsize
    if height is None:
        height = max(3, len(frame) * 0.3)

    fig, ax = plt.subplots(figsize=(width, height))

    # Assign colors from colormap
    n_bars = len(frame)
    colors = [colormap[i % len(colormap)] for i in range(n_bars)]

    # Draw bars - handle positive and negative separately for proper rendering
    pos_mask = frame[value_col] >= 0
    neg_mask = ~pos_mask
    y_positions = list[int](range(len(frame)))

    if mode == "proba":
        # For probability mode, bars extend from baseline
        # Draw positive bars (extend right from baseline)
        if pos_mask.any():
            pos_y = [i for i, mask in enumerate(pos_mask) if mask]
            pos_values = frame.loc[pos_mask, value_col].values
            pos_colors_list = [colors[i] for i, mask in enumerate(pos_mask) if mask]

            ax.barh(
                pos_y,
                pos_values,
                left=baseline,
                color=pos_colors_list,
                height=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

        # Draw negative bars (extend left from baseline)
        if neg_mask.any():
            neg_y = [i for i, mask in enumerate[Any](neg_mask) if mask]
            neg_values = frame.loc[neg_mask, value_col].values
            neg_colors_list = [colors[i] for i, mask in enumerate[Any](neg_mask) if mask]

            ax.barh(
                neg_y,
                abs(neg_values),
                left=baseline + neg_values,  # Start at baseline + negative value
                color=neg_colors_list,
                height=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

        # Add baseline line
        ax.axvline(
            baseline,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"Average default rate: {baseline:.1%}",
            zorder=10,
        )
    else:
        # For log_odds mode, bars centered at 0
        # Draw positive bars (extend right from 0)
        if pos_mask.any():
            pos_y = [i for i, mask in enumerate[Any](pos_mask) if mask]
            pos_values = frame.loc[pos_mask, value_col].values
            pos_colors_list = [colors[i] for i, mask in enumerate[Any](pos_mask) if mask]

            ax.barh(
                pos_y,
                pos_values,
                left=0,
                color=pos_colors_list,
                height=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

        # Draw negative bars (extend left from 0)
        if neg_mask.any():
            neg_y = [i for i, mask in enumerate[Any](neg_mask) if mask]
            neg_values = frame.loc[neg_mask, value_col].values
            neg_colors_list = [colors[i] for i, mask in enumerate[Any](neg_mask) if mask]

            ax.barh(
                neg_y,
                abs(neg_values),
                left=neg_values,  # Start at negative value
                color=neg_colors_list,
                height=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

        # Add baseline line at 0
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Baseline", zorder=10)

    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame[label_col])

    # Collect label data first
    label_data = []
    for i, (_idx, row) in enumerate(frame.iterrows()):
        val = row[value_col]

        # Skip labels for values very close to zero
        if abs(val) < 0.0001:
            continue

        # Format the label
        label_text = value_formatter(val, None)
        if label_text in ["-0%", "+0%", "-0.00", "+0.00"]:
            continue

        bar_end = baseline + val if mode == "proba" else val
        ha = "left" if val >= 0 else "right"
        label_data.append((i, bar_end, label_text, ha))

    # Calculate initial xlim based on data range
    if mode == "proba":
        data_min = baseline + frame[value_col].min()
        data_max = baseline + frame[value_col].max()
        # Add padding: 8% on each side, but ensure baseline is visible
        padding = max((data_max - data_min) * 0.08, baseline * 0.02)
        x_min = min(data_min - padding, baseline * 0.90)
        x_max = max(data_max + padding, baseline * 1.10)
    else:  # mode == "logit"
        data_min = frame[value_col].min()
        data_max = frame[value_col].max()
        padding = (data_max - data_min) * 0.08 if (data_max - data_min) > 0 else 0.1
        x_min = data_min - padding
        x_max = data_max + padding
        # Ensure 0 is included if range spans it
        if x_min < 0 < x_max:
            x_min = min(x_min, -padding)
            x_max = max(x_max, padding)

    # Adjust xlim to accommodate labels (estimate ~10% of range for label width)
    if label_data:
        range_padding = (x_max - x_min) * 0.10
        for _i, bar_end, _label_text, ha in label_data:
            if ha == "left":
                # Label extends right
                if bar_end + range_padding > x_max:
                    x_max = bar_end + range_padding
            else:  # ha == "right"
                # Label extends left
                if bar_end - range_padding < x_min:
                    x_min = bar_end - range_padding

    ax.set_xlim(x_min, x_max)

    # Set explicit tick locations using same approach as plot_performance
    if mode == "proba":
        # For probability mode, use evenly spaced ticks like CAP curves
        # Calculate step size: use 0.01 (1%) for small ranges, 0.02 (2%) for larger
        range_size = x_max - x_min
        if range_size <= 0.05:  # Very small range (< 5%)
            step = 0.01  # 1% steps
        elif range_size <= 0.10:  # Small range (< 10%)
            step = 0.02  # 2% steps
        else:  # Normal range
            step = 0.05  # 5% steps

        # Create ticks from min to max with step, rounded to avoid floating point issues
        tick_start = np.floor(x_min / step) * step
        tick_end = np.ceil(x_max / step) * step
        tick_locations = np.arange(tick_start, tick_end + step / 2, step)
        # Filter to only include ticks within our range
        tick_locations = tick_locations[(tick_locations >= x_min) & (tick_locations <= x_max)]
        ax.set_xticks(tick_locations)
    else:  # mode == "logit"
        # For logit mode, use evenly spaced ticks
        range_size = abs(x_max - x_min)
        if range_size <= 1.0:
            step = 0.5
        elif range_size <= 2.0:
            step = 1.0
        else:
            step = 2.0

        # Create ticks from min to max with step
        tick_start = np.floor(x_min / step) * step
        tick_end = np.ceil(x_max / step) * step
        tick_locations = np.arange(tick_start, tick_end + step / 2, step)
        # Filter to only include ticks within our range
        tick_locations = tick_locations[(tick_locations >= x_min) & (tick_locations <= x_max)]
        ax.set_xticks(tick_locations)

    # Add labels with small offset (1.5% of final range)
    final_range = x_max - x_min
    offset = final_range * 0.015

    for i, bar_end, label_text, ha in label_data:
        label_x = bar_end + offset if ha == "left" else bar_end - offset
        ax.text(
            label_x,
            i,
            label_text,
            va="center",
            ha=ha,
            fontsize=10,
        )

    ax.xaxis.set_major_formatter(plt.FuncFormatter(axis_formatter))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    plt.tight_layout()
    if show_plot:
        plt.show()

    if mode == "proba":
        return (
            frame[["feature", "woe", "proba", "proba_delta"]]
            if is_prediction_level
            else frame[["category", "woe", "proba", "proba_delta"]]
        )
    if is_prediction_level:
        return frame[["feature", "woe", "logit", "logit_delta"]]
    else:
        return frame[["category", "woe", "logit", "logit_delta"]]
