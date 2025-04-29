"""2D Health Modelling Dash App."""

import dash_bootstrap_components as dbc
from dash import dash

import figure_layouts as figure_layouts
from callbacks import assign_global_vars, register_callbacks
from inputs import Inputs
from layout import build_components, build_dash_layout


def update_figure_layouts(layouts: dict, kde):
    """Dynamically modify figure layout.

    We set figure ranges here because we are not using subplots.
    """
    # set figure ranges
    p_theta, _ = kde.pr_interval(kde.pdf)
    p_theta_max = p_theta[-1]
    x_range = [kde.axes[0].min(), kde.axes[0].max()]
    y_range = [kde.axes[1].min(), kde.axes[1].max()]

    layouts.layout_2d["yaxis_range"] = y_range
    layouts.layout_2d["xaxis_range"] = x_range

    layouts.layout_x["xaxis_range"] = x_range
    layouts.layout_y["yaxis_range"] = y_range

    layouts.layout_scoring["yaxis_range"] = [0, 1]
    layouts.layout_scoring["xaxis_range"] = [0, p_theta_max]


def build_app(config):
    """Assemble components and instantiate the application."""
    # generate raw data and fit estimators
    models = Inputs(config)

    # expose global variables for callbacks
    assign_global_vars(models)

    # dynamically update figure range
    update_figure_layouts(figure_layouts, models.kde)

    # instantiate the app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    )

    # build the application layout (models, figures, widgets)
    dcc_objects = build_components(models.df, models.tags)
    html_layout = build_dash_layout(dcc_objects)
    app.layout = html_layout

    # register callbacks
    register_callbacks(app, figure_layouts)

    return app


if __name__ == "__main__":
    config = {
        "model_type": "MLCV",
        "num_points": 64,  # MLCV 2**n | BP11 required 2**n +1
        "n_samples": 1000,
        # Mixture of 2D Gaussians [mu_x, mu_y, sigma_x, sigma_y, rho]
        "mixture_params": [  # Mixture 1: Realistic 2D joint probability distribution
            [-0.5, -4.5, 1.0, 1.0, 0.6],
            [-3.0, 0.0, 1.0, 1.0, 0.8],
            [1.0, -2.0, 1.0, 1.0, -0.8],
            [1, -4.0, 1.5, 1.5, 0.0],
        ],
        # 'mixture_params': [
        #     [1, 3, 1.0, 1.0, 0.6],  # Mixture 2: exaggerated joint distribution with a single high concentration of density
        #     [2, -2, 0.1, 0.1, 0.0]],
        "max_nbytes": "1M",
    }

    # load the app
    app = build_app(config)

    # run
    app.run(debug=False)
