"""Various palettes for charts."""


def generate_sepia_palette(n=40):
    colors = []
    # Base sepia colors from light to dark
    sepia_base = [
        '#fff4e6',  # Very light sepia
        '#ffe4c4',  # Bisque
        '#deb887',  # Burlywood
        '#d2b48c',  # Tan
        '#bc8f8f',  # Rosy brown
        '#a0522d',  # Sienna
        '#8b4513',  # Saddle brown
        '#654321',  # Dark brown
    ]

    # Create variations by interpolating between base colors
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    import numpy as np

    # Create a colormap from our sepia colors
    sepia_cmap = LinearSegmentedColormap.from_list('sepia', sepia_base)

    # Generate n colors
    values = np.linspace(0, 1, n)
    colors = [to_hex(sepia_cmap(v)) for v in values]

    return colors


def hacker_crt_green_palette(num_colors):
    """
    Generates a Plotly color palette in hacker CRT green style with varying brightness.

    Parameters:
    num_colors (int): Number of colors to generate.

    Returns:
    list: A list of color strings in hexadecimal format.
    """
    palette = []

    for i in range(num_colors):
        if i % 2 == 0:
            # Brighter green with varying intensity
            intensity = 255 - int((i / num_colors) * 100)  # Adjust intensity based on position
            brighter_green = f"#00{intensity:02X}00"
            palette.append(brighter_green)
        else:
            # Darker green (fixed for contrast)
            darker_green = "#007700"
            palette.append(darker_green)

    return palette


def format_blog_post_figure(fig, show_legend=True, font_color="#888"):
    """Better formatting for blog post images from charts.

    Example:

    .. code-block:: python

        import pandas as pd
        from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data
        from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark
        from tradeexecutor.utils.notebook import set_large_plotly_chart_font
        from tradeexecutor.visual.palette import hacker_crt_green_palette, format_blog_post_figure


        benchmark_indexes = get_benchmark_data(
            strategy_universe,
            cumulative_with_initial_cash=state.portfolio.get_initial_cash(),
            max_count=4,
            start_at=state.get_trading_time_range()[0],
            interesting_assets=["WETH", "PEPE"],
        )

        set_large_plotly_chart_font(
            base_template="plotly_dark",
        )

        fig = visualise_equity_curve_benchmark(
            name="Strategy (Buy all time high)",
            state=state,
            benchmark_indexes=benchmark_indexes,
            height=800,
            log_y=True,

        )

        format_blog_post_figure(fig, show_legend=True)

        fig.show()

    """

    num_columns = 10
    fig.update_layout(
        # Move legend to bottom
        legend=dict(
            yanchor="top",
            y=-0.1,  # Adjust this value to move legend up/down
            xanchor="center",
            x=0.5,
            # Arrange items in 4 rows
            orientation="h",
            traceorder="normal",
            # nrows=4
            itemwidth=40,  # Adjust the multiplier as needed
            title_text="",
            font=dict(
                size=30,  # Adjust this value to make legend text bigger/smaller
                color=font_color,
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis=dict(
            title=None,
            # other x-axis properties...
            nticks=4,
            # Increase font size (default is usually 12)
            tickfont=dict(
                size=30, # Adjust this value to make font bigger/smaller
                color=font_color,
            ),
            anchor='free',
            side='bottom',
            position=0,
            automargin=True,
            ticklabelposition='outside left',  # Move tick labels to the left
            showgrid=False,
        ),
        yaxis=dict(
            title=None,
            # other y-axis properties...
            nticks=5,
            # Optionally specify tick labels
            # ticktext=['0%', '50%', '100%'],
            tickfont=dict(
                size=30,
                color=font_color,
            ),
            tickprefix='$',
            showgrid=False,
        )
    )

    # Adjust figure size to make it more portrait-like
    fig.update_layout(
        autosize=False,
        width=200,  # Decrease width
        height=200,  # Increase height for portrait aspect
        scene=dict(
            aspectratio=dict(x=0.5, y=1, z=0.5),  # Adjust x and y to change the aspect
            aspectmode='manual'
        )
    )

    fig.update_layout(
        showlegend=show_legend
    )