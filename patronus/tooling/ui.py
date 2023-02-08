import os
from pathlib import Path
from typing import Union

import plotly.io as pio


def save_figure(fig, dpi=2048, data_dir: Union[str, Path] = None, filename: Union[str, Path] = "plotly_figure.png"):
    fig = fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Lucida Grande",
        font_size=22,
        title_font_color="black",
        legend_title_font_color="black",
        xaxis_title="x",
        yaxis_title="y",
    )

    data_dir = Path(os.getcwd()) if data_dir is None else Path(data_dir)

    pio.write_image(fig, str(data_dir / str(filename)), width=1.5 * dpi, height=0.75 * dpi, engine="kaleido")


__all__ = ["save_figure"]
