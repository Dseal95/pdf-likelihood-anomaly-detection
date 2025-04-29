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
        marks = {int(x): {"label": f"{x:.1f}"} for x in np.arange(df_min, df_max, 1)}
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

    # number of surfaces selector
    n_surfaces_dropdown = dcc.Dropdown(
        id="number_of_surfaces",
        options=[{"value": label, "label": label} for label in range(2, 11)],
        value=2,
        multi=False,
        style=dict(
            width="50%",  # width of the widget (size)
            verticalAlign="middle",
            horizontalalign="middle",
            marginLeft=75,
        ),  # move dropdown into center of column
    )

    # value sliders
    tag_x_slider = _create_tag_slider(df, tags[0], "tag_x_slider")
    tag_y_slider = _create_tag_slider(df, tags[1], "tag_y_slider")
    tag_z_slider = _create_tag_slider(df, tags[2], "tag_z_slider")

    components = {
        "alpha": alpha_dropdown,
        "n_surfaces": n_surfaces_dropdown,
        "tag_x": tag_x_slider,
        "tag_y": tag_y_slider,
        "tag_z": tag_z_slider,
    }

    return components


def build_dash_layout(components: dict):
    """Generate the app layout."""
    # instantiate figure
    graph = dcc.Graph(id="graph", figure=go.Figure(layout={"height": 850}))
    # html layout
    html_layout = dbc.Container(
        [
            html.H3(
                children="Demo - Process Health Modelling in 3D",
                style=dict(textAlign="center", verticalAlign="center"),
            ),
            dbc.Row(
                [dbc.Col(html.Div(children=[graph]), className="h-100", width=12)],
                className="h-80",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            children=[
                                html.Label("Health Threshold (alpha)"),
                                components["alpha"],
                            ]
                        ),
                        className="h-100",
                        width=True,
                    ),
                    dbc.Col(
                        html.Div(
                            children=[
                                html.Label("Surface levels (n):"),
                                components["n_surfaces"],
                            ]
                        ),
                        className="h-100",
                        width=True,
                    ),
                    dbc.Col(
                        html.Div(children=[html.Label("Tag X:"), components["tag_x"]]),
                        className="h-100",
                        width=True,
                    ),
                    dbc.Col(
                        html.Div(children=[html.Label("Tag Y:"), components["tag_y"]]),
                        className="h-100",
                        width=True,
                    ),
                    dbc.Col(
                        html.Div(children=[html.Label("Tag Z:"), components["tag_z"]]),
                        className="h-100",
                        width=True,
                    ),
                ],
                className="h-20",
            ),
        ],
        className="vh-100",
        fluid=True,
    )

    return html_layout
