"""Manage the application html layout."""

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html
from plotly import graph_objects as go


def build_components(df: pd.DataFrame, tags: list):
    """Instantiate interactive dash-core-components."""

    def _create_tag_slider(df, tag, name, step=0.1):
        df_min = round(df[tag].min(), 1)
        df_max = round(df[tag].max(), 1)
        df_mu = round(df[tag].mean(), 1)
        marks = {x: {"label": f"{x:.1f}"} for x in np.arange(df_min, df_max, 2)}
        slider = dcc.Slider(
            id=name,
            min=df_min,
            max=df_max,
            step=step,
            value=df_mu,
            dots=False,
            marks={**marks},
            included=False,
            updatemode="drag",
        )
        return slider

    # alpha selector
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_dropdown = dcc.Dropdown(
        id="alpha_selection",
        options=[{"value": label, "label": label} for label in thresholds],
        value=0.05,  # default value = 0.05
        multi=False,
        style=dict(
            width="50%",  # width of the widget (size)
            verticalAlign="middle",
            horizontalalign="middle",
            marginLeft=75,
        ),  # move dropdown into center of column
    )

    # number of contours selector
    n_contours_dropdown = dcc.Dropdown(
        id="number_of_contours",
        options=[{"value": label, "label": label} for label in range(2, 16)],
        value=10,
        multi=False,
        style=dict(
            width="50%",  # width of the widget (size)
            verticalAlign="middle",
            horizontalalign="middle",
            marginLeft=75,
            marginBottom=20,
        ),  # move dropdown into center of column
    )

    # value sliders
    tag_x_slider = _create_tag_slider(df, tags[0], "tag_x_slider")
    tag_y_slider = _create_tag_slider(df, tags[1], "tag_y_slider")

    components = {
        "alpha": alpha_dropdown,
        "n_contours": n_contours_dropdown,
        "tag_x": tag_x_slider,
        "tag_y": tag_y_slider,
    }

    return components


def build_dash_layout(components: dict):
    """Generate the app layout."""
    # instantiate figures
    figs = [go.Figure() for x in range(0, 4)]
    graph_2d = dcc.Graph(id="graph_2d", figure=figs[0])
    graph_x = dcc.Graph(id="graph_x", figure=figs[1])
    graph_y = dcc.Graph(id="graph_y", figure=figs[2])
    graph_score = dcc.Graph(id="graph_score", figure=figs[3])

    html_layout = html.Div(
        [
            html.H3(
                children="Demo - Process Health Modelling in 2D",
                style=dict(textAlign="center", verticalAlign="center"),
                className="h-50",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            children=[graph_x, graph_2d],
                            style=dict(verticalAlign="bottom"),
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(
                            children=[
                                html.Label("Tag X:"),
                                components["tag_x"],
                                html.Label("Tag Y:"),
                                components["tag_y"],
                                html.Label("Health Threshold (alpha)"),
                                components["alpha"],
                                html.Label("Contour levels (n):"),
                                components["n_contours"],
                                graph_y,
                            ],
                            style=dict(verticalAlign="bottom", lineheight="normal"),
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            children=[graph_score], style=dict(verticalAlign="center")
                        ),
                        width=3,
                        align="end",
                    ),
                ],
                style={"height": "100%", "width": "100%"},
                align="end",
            ),
        ]
    )

    return html_layout
