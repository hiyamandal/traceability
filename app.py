import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import os

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    # these meta_tags ensure content is scaled correctly on different devices
    # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

app.config.suppress_callback_exceptions = True
server = app.server

server.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')

if __name__ == '__main__':
    app.run_server(debug=True)
