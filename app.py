import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import flask
import os


print(dcc.__version__) # 0.6.0 or above is required

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    # these meta_tags ensure content is scaled correctly on different devices
    # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

server = app.server
app.config.suppress_callback_exceptions = True
