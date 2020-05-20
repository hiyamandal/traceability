import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from apps import commonmodules
from app import app


layout = html.Div([
    commonmodules.get_header(),
    html.Br(),
    commonmodules.get_menu(),
    html.Br(),
    html.Div([
        html.H3('Station Anmeldung', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
        html.Hr(),
        html.H4('Bauteil wurde erfolgreich angemeldet!'),
        html.Br(),
        html.Div(
            [
                dbc.Alert(
                    [
                        "Zur Station ",
                        html.A("Spanen", href="/spanen", className="alert-link"),
                    ],
                    color="primary",
                ),
            ]
        ),
    ], className='twelve columns'),

])


