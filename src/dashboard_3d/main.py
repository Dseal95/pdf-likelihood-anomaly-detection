"""3D Health Modelling Dash App."""

import dash_bootstrap_components as dbc
from dash import dash

import figure_layouts
from callbacks import assign_global_vars, register_callbacks
from inputs import Inputs
from layout import build_components, build_dash_layout


def build_app(config):
    """Assemble components and instantiate the application."""
    # generate raw data and fit estimators
    models = Inputs(config)

    # expose global variables for callbacks
    assign_global_vars(models)

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
        "model_type": "BP11",
        "num_points": 65,  # 2^n + 1: 33, 65
        "n_samples": 1000,
        # Mixture of 3D Gaussians [mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, rho]
        "mixture_params": [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [6.0, 0.0, -2.0, 1.0, 1.0, 1.0, 0.0],
        ],
        "max_nbytes": "1M",
    }
    # load the app
    app = build_app(config)

    # run
    app.run(debug=False)
