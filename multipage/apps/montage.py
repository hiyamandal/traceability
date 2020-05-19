import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from apps import commonmodules
from app import app


layout = html.Div([
    commonmodules.get_header(),
    commonmodules.get_menu(),
    html.Br(),
    html.H3('Station Montage', style={'font-weight': 'bold'}),
    html.Hr(),
    html.H4('Montage des Bauteils erfolgreich abgeschlossen!'),
    html.Br(),
    html.Div(
            [
                dbc.Alert(
                    [
                        # "Zur ",
                        html.A("Startseite", href="/", className="alert-link"),
                    ],
                    color="primary",
                ),
            ]
        ),
])
