"""
Display utilities for rich HTML rendering of DataFrames in Jupyter notebooks.

Provides styled HTML output similar to scikit-learn's estimator HTML representation.
"""

from functools import wraps
from typing import Callable, Optional

import pandas as pd


def _get_light_mode_css() -> str:
    """Get light mode CSS based on baseline foundation design."""
    return """
    <style>
    .fastwoe-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        margin: 20px 0;
        padding: 0;
        background: #FFFFFF;
    }
    .fastwoe-title {
        font-size: 18px;
        font-weight: 500;
        color: #202020;
        margin-bottom: 8px;
        padding: 12px 16px;
        background: #FCFCFC;
        border-left: 3px solid #646464;
        border-radius: 0;
    }
    .fastwoe-subtitle {
        font-size: 13px;
        color: #838383;
        margin-bottom: 12px;
        padding: 0 16px;
        font-style: italic;
    }
    .fastwoe-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
        border: 1px solid #E8E8E8;
        border-radius: 0;
        background: #FFFFFF;
    }
    .fastwoe-table thead {
        background: #FCFCFC;
        border-bottom: 2px solid #E8E8E8;
    }
    .fastwoe-table th {
        padding: 12px 16px;
        text-align: left;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
        color: #646464;
    }
    .fastwoe-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #E8E8E8;
        color: #444444;
        background: #FFFFFF;
    }
    .fastwoe-table tbody tr:nth-child(even) td {
        background-color: #FCFCFC;
    }
    .fastwoe-table tbody tr:hover td {
        background-color: #F0F0F0;
        transition: background-color 0.2s cubic-bezier(0.32, 0.72, 0, 1);
    }
    .fastwoe-table tbody tr:last-child td {
        border-bottom: none;
    }
    .numeric-cell {
        font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
        text-align: right;
    }
    .highlight-high {
        background: rgba(76, 175, 80, 0.12) !important;
        font-weight: 500;
    }
    .highlight-medium {
        background: rgba(255, 193, 7, 0.12) !important;
    }
    .highlight-low {
        background: rgba(244, 67, 54, 0.12) !important;
    }
    .significance-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 500;
    }
    .sig-yes {
        background-color: rgba(76, 175, 80, 0.15);
        color: #2d5016;
    }
    .sig-no {
        background-color: rgba(244, 67, 54, 0.15);
        color: #7f1d1d;
    }
    </style>
    """


def _get_dark_mode_css() -> str:
    """Get dark mode CSS based on baseline foundation design."""
    return """
    <style>
    .fastwoe-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        margin: 20px 0;
        padding: 0;
        background: #0B0B0B;
    }
    .fastwoe-title {
        font-size: 18px;
        font-weight: 500;
        color: #EEEEEE;
        margin-bottom: 8px;
        padding: 12px 16px;
        background: #161616;
        border-left: 3px solid #B5B5B5;
        border-radius: 0;
    }
    .fastwoe-subtitle {
        font-size: 13px;
        color: #7B7B7B;
        margin-bottom: 12px;
        padding: 0 16px;
        font-style: italic;
    }
    .fastwoe-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
        border: 1px solid #222222;
        border-radius: 0;
        background: #0B0B0B;
    }
    .fastwoe-table thead {
        background: #161616;
        border-bottom: 2px solid #222222;
    }
    .fastwoe-table th {
        padding: 12px 16px;
        text-align: left;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
        color: #B5B5B5;
    }
    .fastwoe-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #222222;
        color: #C8C8C8;
        background: #0B0B0B;
    }
    .fastwoe-table tbody tr:nth-child(even) td {
        background-color: #0F0F0F;
    }
    .fastwoe-table tbody tr:hover td {
        background-color: #222222;
        transition: background-color 0.2s cubic-bezier(0.32, 0.72, 0, 1);
    }
    .fastwoe-table tbody tr:last-child td {
        border-bottom: none;
    }
    .numeric-cell {
        font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
        text-align: right;
    }
    .highlight-high {
        background: rgba(76, 175, 80, 0.2) !important;
        font-weight: 500;
    }
    .highlight-medium {
        background: rgba(255, 193, 7, 0.2) !important;
    }
    .highlight-low {
        background: rgba(244, 67, 54, 0.2) !important;
    }
    .significance-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 500;
    }
    .sig-yes {
        background-color: rgba(76, 175, 80, 0.25);
        color: #69db7c;
    }
    .sig-no {
        background-color: rgba(244, 67, 54, 0.25);
        color: #ff6b6b;
    }
    </style>
    """


class StyledDataFrame:
    """
    Wrapper for pandas DataFrame with rich HTML representation.

    Provides styled HTML output in Jupyter notebooks with:
    - Color-coded cells based on values
    - Gradient backgrounds for numeric columns
    - Significance highlighting
    - Clean, professional styling similar to scikit-learn

    Example:
        >>> df = pd.DataFrame({'iv': [0.5, 0.3], 'gini': [0.6, 0.4]})
        >>> styled = StyledDataFrame(df, title="IV Analysis")
        >>> styled  # In Jupyter, displays with rich HTML
    """

    def __init__(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        highlight_cols: Optional[list] = None,
        precision: int = 4,
        theme: str = "light",
    ):
        """
        Initialize styled DataFrame wrapper.

        Args:
            df: DataFrame to style
            title: Optional title to display above the table
            subtitle: Optional subtitle/description
            highlight_cols: Columns to apply gradient highlighting (default: numeric columns)
            precision: Number of decimal places for floating point numbers
            theme: Color theme - "light" or "dark" (default: "light")
        """
        self.df = df.copy()
        self.title = title
        self.subtitle = subtitle
        self.highlight_cols = highlight_cols
        self.precision = precision
        self.theme = theme

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter notebooks."""
        return render_dataframe_html(
            self.df,
            title=self.title,
            subtitle=self.subtitle,
            highlight_cols=self.highlight_cols,
            precision=self.precision,
            theme=self.theme,
        )

    def __repr__(self) -> str:
        """Plain text representation for console."""
        return repr(self.df)

    def __str__(self) -> str:
        """String representation."""
        return str(self.df)


def render_dataframe_html(
    df: pd.DataFrame,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    highlight_cols: Optional[list] = None,
    precision: int = 4,
    theme: str = "light",
) -> str:
    """
    Render DataFrame as styled HTML for Jupyter notebooks.

    Args:
        df: DataFrame to render
        title: Optional title
        subtitle: Optional subtitle
        highlight_cols: Columns to highlight with gradients
        precision: Decimal precision
        theme: Color theme - "light" or "dark"

    Returns:
        HTML string with styled table
    """
    # CSS styling inspired by baseline foundation design system
    if theme == "dark":
        css = _get_dark_mode_css()
    else:
        css = _get_light_mode_css()

    # Start HTML
    html_parts = [css, '<div class="fastwoe-container">']

    # Add title if provided
    if title:
        html_parts.append(f'<div class="fastwoe-title">{title}</div>')

    # Add subtitle if provided
    if subtitle:
        html_parts.append(f'<div class="fastwoe-subtitle">{subtitle}</div>')

    html_parts.extend(('<table class="fastwoe-table">', "<thead><tr>"))
    html_parts.extend(f"<th>{col}</th>" for col in df.columns)
    html_parts.append("</tr></thead>")

    # Determine which columns to highlight
    if highlight_cols is None:
        highlight_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Calculate value ranges for gradient highlighting
    col_ranges = {}
    for col in highlight_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            col_ranges[col] = (col_min, col_max)

    # Table body
    html_parts.append("<tbody>")
    for _, row in df.iterrows():
        html_parts.append("<tr>")
        for col in df.columns:
            value = row[col]

            # Determine cell class and formatting
            cell_class = ""
            formatted_value = value

            # Format numeric values
            if pd.api.types.is_numeric_dtype(df[col]):
                cell_class = "numeric-cell"
                if isinstance(value, float):
                    formatted_value = f"{value:.{precision}f}"

                # Add gradient highlighting for specified columns
                if col in col_ranges:
                    col_min, col_max = col_ranges[col]
                    if col_max > col_min:
                        normalized = (value - col_min) / (col_max - col_min)
                        if normalized >= 0.66:
                            cell_class += " highlight-high"
                        elif normalized >= 0.33:
                            cell_class += " highlight-medium"
                        else:
                            cell_class += " highlight-low"

            # Special handling for significance column
            if col.lower() in ["iv_significance", "significance"]:
                if "significant" in str(value).lower() and "not" not in str(value).lower():
                    formatted_value = f'<span class="significance-badge sig-yes">{value}</span>'
                else:
                    formatted_value = f'<span class="significance-badge sig-no">{value}</span>'

            html_parts.append(f'<td class="{cell_class}">{formatted_value}</td>')
        html_parts.append("</tr>")
    html_parts.extend(("</tbody>", "</table>", "</div>"))
    return "".join(html_parts)


def style_iv_analysis(df: pd.DataFrame, theme: str = "light") -> StyledDataFrame:
    """
    Create styled HTML representation for IV analysis DataFrame.

    Args:
        df: IV analysis DataFrame from FastWoe.get_iv_analysis()
        theme: Color theme - "light" or "dark" (default: "light")

    Returns:
        StyledDataFrame with rich HTML representation

    Example:
        >>> woe = FastWoe()
        >>> woe.fit(X, y)
        >>> styled_df = style_iv_analysis(woe.get_iv_analysis(), theme="dark")
        >>> styled_df  # Displays with rich HTML in Jupyter
    """
    return StyledDataFrame(
        df,
        title="Information Value Analysis",
        subtitle="Feature importance and statistical significance for WOE encoding",
        highlight_cols=["iv", "gini", "iv_se"],
        precision=4,
        theme=theme,
    )


def style_woe_mapping(df: pd.DataFrame, feature_name: str, theme: str = "light") -> StyledDataFrame:
    """
    Create styled HTML representation for WOE mapping DataFrame.

    Args:
        df: WOE mapping DataFrame
        feature_name: Name of the feature
        theme: Color theme - "light" or "dark" (default: "light")

    Returns:
        StyledDataFrame with rich HTML representation
    """
    return StyledDataFrame(
        df,
        title=f"WOE Mapping: {feature_name}",
        subtitle="Weight of Evidence transformation for each category",
        highlight_cols=["woe", "iv"],
        precision=4,
        theme=theme,
    )


def styled(
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    highlight_cols: Optional[list] = None,
    precision: int = 4,
    theme: str = "light",
) -> Callable:
    """
    Decorator to automatically style DataFrame outputs from functions.

    This decorator wraps functions that return DataFrames and automatically
    applies StyledDataFrame formatting for rich HTML display in Jupyter notebooks.

    Args:
        title: Optional title to display above the table
        subtitle: Optional subtitle/description
        highlight_cols: Columns to apply gradient highlighting
        precision: Number of decimal places for floating point numbers
        theme: Color theme - "light" or "dark" (default: "light")

    Returns:
        Decorator function

    Example:
        >>> @styled(title="My Analysis", precision=3, theme="dark")
        >>> def get_results():
        >>>     return pd.DataFrame({'metric': [0.5, 0.7], 'value': [100, 200]})
        >>>
        >>> get_results()  # Returns StyledDataFrame in Jupyter
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return StyledDataFrame(
                    result,
                    title=title,
                    subtitle=subtitle,
                    highlight_cols=highlight_cols,
                    precision=precision,
                    theme=theme,
                )
            return result

        return wrapper

    return decorator


def iv_styled(func: Callable) -> Callable:
    """
    Decorator for IV analysis functions that return DataFrames.

    Automatically applies IV analysis styling to the returned DataFrame.

    Example:
        >>> @iv_styled
        >>> def get_iv_analysis():
        >>>     return woe.get_iv_analysis()
        >>>
        >>> get_iv_analysis()  # Returns styled IV analysis
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            return style_iv_analysis(result)
        return result

    return wrapper
