"""Manage callback functions."""

import dash
import numpy as np
from dash.dependencies import Input, Output
from plotly import graph_objects as go

from inputs import Inputs


def assign_global_vars(estimators: Inputs):
    """Assign global variables that are required by the application callbacks.

    Here they are provided by our Estimators class.
    """
    global kde, pdf_y, pdf_x
    global hist_x, bins_x, hist_y, bins_y
    global p_theta, cumpr_theta, p_theta_max
    global x_range, y_range
    global model_type

    # estimators
    model_type = estimators.model_type
    kde = estimators.kde
    pdf_y = estimators.pdf_y
    pdf_x = estimators.pdf_x
    hist_x, bins_x = np.histogram(
        estimators.df[estimators.tags[0]], bins="stone", density=True
    )
    hist_y, bins_y = np.histogram(
        estimators.df[estimators.tags[1]], bins="stone", density=True
    )

    # properties used to set plot ranges
    p_theta, cumpr_theta = estimators.kde.pr_interval(kde.pdf)


def _get_marginal_traces(x, axes, pdf, hist, bins, orientation=None):
    """Parameterised generation of traces for marginal plots."""
    if orientation == "h":
        axis_1 = "y"
        axis_2 = "x"
    else:
        axis_1 = "x"
        axis_2 = "y"

    # marginalised pdf
    ax = {f"{axis_1}": axes, f"{axis_2}": pdf, "line": dict(color="black", width=2)}
    trace_pdf = go.Scattergl(**ax)

    # histogram
    ax = {
        f"{axis_2}": hist,
        f"{axis_1}": bins[0:-1],
        "width": bins[1] - bins[0],
        "orientation": orientation,
        "marker_color": "rgb(211,211,211)",
    }
    trace_hist = go.Bar(**ax)

    # plot a line at x
    y2 = np.interp(x, axes, pdf)
    ax = {
        f"{axis_1}": [x, x],
        f"{axis_2}": [0, y2],
        "mode": "lines",
        "line": dict(color="black", width=2, dash="dash"),
    }
    trace_line = go.Scattergl(**ax)

    return [trace_pdf, trace_hist, trace_line]


def register_callbacks(app, layouts):
    """Registers the callback functions that enable user interaction."""

    # Figure callbacks - these update plots during user interaction
    # Widget sync
    @app.callback(
        [Output("tag_x_slider", "value"), Output("tag_y_slider", "value")],
        [
            Input("tag_x_slider", "value"),
            Input("tag_y_slider", "value"),
            Input("graph_2d", "clickData"),
        ],
    )
    def sync_widgets(x, y, clickData):
        """Sync the value of tag_x and tag_y sliders if click functionality used."""
        # identify which callback to use (click or slider)
        if dash.callback_context.triggered:
            if dash.callback_context.triggered[0]["prop_id"] == "graph_2d.clickData":
                # use clicked value to update tag_x and tag_y sliders
                x = clickData["points"][0]["x"]
                y = clickData["points"][0]["y"]

        return x, y

    # 2D plot
    @app.callback(
        Output("graph_2d", "figure"),
        [
            Input("alpha_selection", "value"),
            Input("number_of_contours", "value"),
            Input("tag_x_slider", "value"),
            Input("tag_y_slider", "value"),
        ],
    )
    def update_contour_plot(alpha, n_contours, x, y):
        """Update the contour plot traces.

        Enables dynamic adjustment of contour lines, the health envelope
        (lower contour bound) and the current position [x,y]

        Returns: 2D contour plot plotly figure object
        """
        pdf_masked = kde.pdf_masked
        tags = kde.taglist

        # calculate contours
        ll_threshold = p_theta[cumpr_theta >= alpha][0]
        levels = np.linspace(ll_threshold, p_theta[-1], n_contours)
        level_spacing = (levels[-1] - levels[0]) / (len(levels) - 1)

        health_envelope = np.linspace(
            ll_threshold, p_theta[-1], 2
        )  # n_contours = 2 for health envelope
        health_envelope_spacing = (health_envelope[-1] - health_envelope[0]) / (
            len(health_envelope) - 1
        )

        # score the current position
        score, _ = kde.score([[x, y]])

        if model_type == "BP11":
            # modifying output for plotting. BP11 returns score and ll in arrays, MLCV in np.floats
            score = score[0]

        # classify
        c = "green" if score >= alpha else "red"

        # Plotly figure:
        # 2D density distribution
        trace1 = go.Contour(
            z=pdf_masked,
            contours={
                "start": levels[0],
                "end": levels[-1],
                "size": level_spacing,
                "coloring": "lines",
            },
            colorscale="viridis",
            showscale=False,
            showlegend=True,
            line_width=2,
            x=kde.axes[0],
            y=kde.axes[1],
            hovertemplate="<b>Density Function</b><br>"
            + "X,Y: (%{x:,.2f},%{y:,.2f})<br>"
            + "p_theta: %{z:.4e}"
            + "<extra></extra>",
        )

        # plot health envelope (outer contour bound)
        trace_env = go.Contour(
            z=pdf_masked,
            contours={
                "start": health_envelope[0],
                "end": health_envelope[-1],
                "size": health_envelope_spacing,
                "coloring": "lines",
            },
            colorscale=[[0, "rgb(250,0,0)"], [1.0, "rgb(250,0,0)"]],
            showscale=False,
            showlegend=True,
            line_width=2,
            x=kde.axes[0],
            y=kde.axes[1],
            hoverinfo="skip",
        )

        # Raw data scatter
        trace2 = go.Scattergl(
            x=kde.data[tags[0]].values,
            y=kde.data[tags[1]].values,
            mode="markers",
            showlegend=False,
            opacity=0.5,
            marker=dict(color="grey", size=3),
            hoverinfo="skip",
        )

        # Current position scatter
        trace_pt = go.Scattergl(
            x=[x],
            y=[y],
            xaxis="x",
            yaxis="y",
            mode="markers",
            marker=dict(color=c, size=16, line=dict(width=2, color="DarkSlateGrey")),
        )

        layouts.layout_2d["annotations"] = [
            dict(
                font=dict(size=12, color=c, family="Verdana"),
                text=f"Health Score: {100 * score:,.1f}%",
                align="left",
                xanchor="left",
                x="0.01",
                y="0.01",
                showarrow=False,
                xref="paper",
                yref="paper",
            )
        ]

        data = [trace1, trace_env, trace2, trace_pt]

        return go.Figure(data=data, layout=layouts.layout_2d)

    # x marginal plot
    @app.callback(Output("graph_x", "figure"), [Input("tag_x_slider", "value")])
    def update_x_plot(x):
        """Update marginal plot in x."""
        traces = _get_marginal_traces(x, kde.axes[0], pdf_x, hist_x, bins_x)

        return go.Figure(data=traces, layout=layouts.layout_x)

    # y marginal plot
    @app.callback(Output("graph_y", "figure"), [Input("tag_y_slider", "value")])
    def update_y_plot(y):
        """Update marginal plot in y."""
        traces = _get_marginal_traces(y, kde.axes[1], pdf_y, hist_y, bins_y, "h")

        return go.Figure(data=traces, layout=layouts.layout_y)

    # scoring function
    @app.callback(
        Output("graph_score", "figure"),
        [
            Input("alpha_selection", "value"),
            Input("tag_x_slider", "value"),
            Input("tag_y_slider", "value"),
        ],
    )
    def update_scoring_plot(alpha, x, y):
        """Updates the scoring function plot and traces.

        Plots the health threshold on scoring function and the healthscore
        of [x, y]

        Arguments:
            alpha: (float) - alpha slider value
            x: (float) - slider value of tagA
            y: (float) - slider value of tagB

        Returns: scoring function plotly figure object
        """
        # score the current position
        score, likelihood = kde.score([[x, y]])

        # classify
        c = "green" if score >= alpha else "red"

        # threshold
        ll_threshold = p_theta[cumpr_theta >= alpha][0]

        # scoring function
        trace1 = go.Scattergl(
            x=p_theta, y=cumpr_theta, mode="lines", line=dict(color="black")
        )

        if model_type == "MLCV":
            # modifying output for plotting. BP11 returns score and ll in arrays, MLCV in np.floats
            score = [score]
            likelihood = [likelihood]

        # score for the current position
        trace_pt = go.Scattergl(
            x=likelihood,
            y=score,
            mode="markers",
            marker=dict(color=c, size=16, line=dict(width=2, color="DarkSlateGrey")),
        )
        layouts.layout_scoring["shapes"] = [
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=ll_threshold,
                x1=ll_threshold,
                y0=0,
                y1=1,
                line=dict(color="red", width=2),
            ),
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                x1=ll_threshold,
                y0=0,
                y1=1,
                line=dict(width=0),
                fillcolor="red",
                opacity=0.2,
            ),
        ]
        layouts.layout_scoring["annotations"] = [
            dict(
                font=dict(size=12, color="red", family="Verdana"),
                text="Health Threshold",
                align="left",
                xanchor="left",
                textangle=90,
                x=ll_threshold,
                y="0.5",
                showarrow=False,
                xref="x",
                yref="paper",
            )
        ]

        traces = [trace1, trace_pt]
        return go.Figure(data=traces, layout=layouts.layout_scoring)
