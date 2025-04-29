"""Manage callback functions."""

from math import exp, pi, sqrt

import dash
import numpy as np
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal

from inputs import Inputs


def _probability_density(mu, coordinates):
    """Calculate the probability density of a 3D normal gaussian distribution.
    3 variables uncorrelated so covariance matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]"""

    scipyMNdist = multivariate_normal(
        mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    pdf = []
    for X in coordinates:
        # pX = (1 / sqrt(2 * pi)) * exp(-(1 / 2) * (X[0] - mu[0]) ** 2) * (1 / sqrt(2 * pi)) * exp(
        #     -(1 / 2) * (X[1] - mu[1]) ** 2) * (1 / sqrt(2 * pi)) * exp(-(1 / 2) * (X[2] - mu[2]) ** 2)
        pX = scipyMNdist.pdf(X)
        pdf.append(pX)

    return pdf


def assign_global_vars(estimators: Inputs):
    """Assign global variables that are required by the application callbacks.

    Here they are provided by our Estimators class.
    """
    global kde, pdf_y, pdf_x, pdf_z
    global x_, y_, z_, pdf_values  # , pdf_analytical
    global hist_x, bins_x, hist_y, bins_y, hist_z, bins_z
    global p_theta, cumpr_theta, p_theta_max
    global x_range, y_range, z_range
    global model_type

    # estimators
    model_type = estimators.model_type
    kde = estimators.kde
    pdf_values = estimators.pdf_values
    x_, y_, z_ = estimators.grid_coords.T

    pdf_y = estimators.pdf_y
    pdf_x = estimators.pdf_x
    pdf_z = estimators.pdf_z
    hist_x, bins_x = np.histogram(
        estimators.df[estimators.tags[0]], bins="stone", density=True
    )
    hist_y, bins_y = np.histogram(
        estimators.df[estimators.tags[1]], bins="stone", density=True
    )
    hist_z, bins_z = np.histogram(
        estimators.df[estimators.tags[2]], bins="stone", density=True
    )

    # properties used to set plot ranges
    p_theta, cumpr_theta = estimators.kde.pr_interval(kde.pdf)

    # analytical pdf for analytical plot
    # pdf_analytical = _probability_density(mu=[0, 0, 0], coordinates=estimators.grid_coords)


def _get_marginal_traces(x, axes, pdf, hist, bins, orientation=None):
    """Parameterised generation of traces for marginal plots."""

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
        [
            Output("tag_x_slider", "value"),
            Output("tag_y_slider", "value"),
            Output("tag_z_slider", "value"),
        ],
        [
            Input("tag_x_slider", "value"),
            Input("tag_y_slider", "value"),
            Input("tag_z_slider", "value"),
            Input("graph", "clickData"),
        ],
    )
    def sync_3d_widgets(x, y, z, clickData):
        """Sync the value of tag_x and tag_y sliders if click functionality used."""
        # identify which callback to use (click or slider)
        if dash.callback_context.triggered:
            if dash.callback_context.triggered[0]["prop_id"] == "graph.clickData":
                # use clicked value to update tag_x, tag_y and tag_z sliders
                x = clickData["points"][0]["x"]
                y = clickData["points"][0]["y"]
                z = clickData["points"][0]["z"]

        return x, y, z

    # 3D plot
    @app.callback(
        Output("graph", "figure"),
        [
            Input("alpha_selection", "value"),
            Input("number_of_surfaces", "value"),
            Input("tag_x_slider", "value"),
            Input("tag_y_slider", "value"),
            Input("tag_z_slider", "value"),
        ],
    )
    def update_figure(alpha, n_surfaces, x, y, z):
        """Update the contour plot traces.

        Enables dynamic adjustment of figure including: isosurfaces on 3D density plot, current position on marginal
        x,y and z and health region on scoring function.

        Returns: plotly figure with subplots
        """
        tags = kde.taglist

        # calculate health envelope
        ll_threshold = p_theta[cumpr_theta >= alpha][0]
        levels = np.linspace(ll_threshold, p_theta[-1], n_surfaces)

        # score the current position
        score, likelihood = kde.score([[x, y, z]])

        if model_type == "BP11":
            # modifying output for plotting. BP11 returns score and ll in arrays, MLCV in np.floats
            score = score[0]
            likelihood = likelihood[0]

        # classify
        score_color = "green" if score >= alpha else "red"

        # instantiate plotly figure
        fig = make_subplots(
            rows=3,
            cols=3,
            column_widths=[0.55, 0.25, 0.2],
            specs=[
                [{"type": "scene", "rowspan": 3}, {}, {"rowspan": 3}],
                [None, {}, None],
                [None, {}, None],
            ],
        )

        # Create data traces:
        # 3D kde pdf plot
        isosurface_kde = go.Isosurface(
            x=x_,
            y=y_,
            z=z_,
            value=pdf_values,
            opacity=0.1,
            isomin=levels[0],
            isomax=levels[-1],
            surface=dict(count=len(levels), fill=1, pattern="all", show=True),
            colorscale="viridis",
            caps=dict(x_show=True, y_show=True, z_show=True),
            showscale=False,
            showlegend=False,
            hovertemplate="X,Y,Z: (%{x:,.2f},%{y:,.2f},%{z:,.2f})<br>"
            + "<extra></extra>",
        )
        ## 3D Analytical pdf plot
        # isosurface_analytical = go.Isosurface(x=x_,
        #                                       y=y_,
        #                                       z=z_,
        #                                       value=pdf_analytical,
        #                                       opacity=0.1,
        #                                       isomin=levels[0],
        #                                       isomax=levels[0],
        #                                       surface=dict(count=1,
        #                                                    fill=1,
        #                                                    pattern="all",
        #                                                    show=True
        #                                                    ),
        #                                       colorscale=[[0, "rgb(220,20,60)"], [1.0, "rgb(220,20,60)"]],
        #                                       caps=dict(x_show=True,
        #                                                 y_show=True,
        #                                                 z_show=True),
        #                                       showscale=False,
        #                                       showlegend=False
        #                                       )
        # Raw data scatter
        raw_data = go.Scatter3d(
            x=kde.data[tags[0]].values,
            y=kde.data[tags[1]].values,
            z=kde.data[tags[2]].values,
            mode="markers",
            showlegend=False,
            opacity=0.5,
            marker=dict(color="grey", size=1),
            hovertemplate="X,Y,Z: (%{x:,.2f},%{y:,.2f},%{z:,.2f})<br>"
            + "<extra></extra>",
        )
        # plotting invisible pdf grid
        grid_data = go.Scatter3d(
            x=x_,
            y=y_,
            z=z_,
            mode="markers",
            showlegend=False,
            opacity=0,
            marker=dict(color="white", size=1),
            hovertemplate="X,Y,Z: (%{x:,.2f},%{y:,.2f},%{z:,.2f})<br>"
            + "<extra></extra>",
        )
        # Current position scatter
        trace_pt = go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers",
            marker=dict(
                color=score_color, size=4, line=dict(width=2, color="DarkSlateGrey")
            ),
            hovertemplate="X,Y,Z: (%{x:,.2f},%{y:,.2f},%{z:,.2f})<br>"
            "<extra></extra>",
        )
        # generate marginal x, y and z traces
        x_traces = _get_marginal_traces(x, kde.axes[0], pdf_x, hist_x, bins_x)
        y_traces = _get_marginal_traces(y, kde.axes[1], pdf_y, hist_y, bins_y)
        z_traces = _get_marginal_traces(z, kde.axes[2], pdf_z, hist_z, bins_z)

        # scoring function
        scoring = go.Scattergl(
            x=p_theta, y=cumpr_theta, mode="lines", line=dict(color="black")
        )
        # score for the current position
        scoring_pt = go.Scattergl(
            x=[likelihood],
            y=[score],
            mode="markers",
            marker=dict(
                color=score_color, size=10, line=dict(width=2, color="DarkSlateGrey")
            ),
        )

        # Adding traces to subplot figure - Add Analytical PDf for debugging
        fig.add_traces(
            [
                isosurface_kde,
                raw_data,
                grid_data,
                trace_pt,  # isosurface_analytical
                x_traces[0],
                x_traces[1],
                x_traces[2],
                y_traces[0],
                y_traces[1],
                y_traces[2],
                z_traces[0],
                z_traces[1],
                z_traces[2],
                scoring,
                scoring_pt,
            ],
            rows=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1],
            cols=[1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
        )

        # Update scoring plot
        layouts.layout_3d["shapes"] = [
            dict(
                type="line",
                xref="x2",
                yref="y2",
                x0=ll_threshold,
                x1=ll_threshold,
                y0=0,
                y1=1,
                line=dict(color="red", width=2),
            ),
            dict(
                type="rect",
                xref="x2",
                yref="y2",
                x0=0,
                x1=ll_threshold,
                y0=0,
                y1=1,
                line=dict(width=0),
                fillcolor="red",
                opacity=0.2,
            ),
        ]
        layouts.layout_3d["annotations"] = [
            dict(
                font=dict(size=12, color=score_color, family="Verdana"),
                text="Health Threshold",
                align="left",
                xanchor="left",
                textangle=90,
                x=ll_threshold,
                y="0.5",
                showarrow=False,
                xref="x2",
                yref="y2",
            )
        ]
        layouts.scene_3d["annotations"] = [
            dict(
                showarrow=False,
                x=max(x_) - 1,
                y=max(y_) - 1,
                z=min(z_) + 1,
                text=f"Health Score: {100 * score:.1f}%",
                textangle=0,
                font=dict(family="Verdana", color=score_color, size=12),
            ),
            dict(
                showarrow=False,
                x=0,
                y=0,
                z=max(z_),
                text="Density Function",
                textangle=0,
                font=dict(family="Verdana", color="black", size=16),
            ),
        ]

        # Update figure with new layouts
        fig.update_scenes(layouts.scene_3d)
        fig.update_layout(layouts.layout_3d)

        return fig
