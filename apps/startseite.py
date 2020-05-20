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
        html.H3('Startseite', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
        html.Hr(),
        html.H4('Bitte neues Bauteil anmelden!'),
        html.Br(),
        html.Div(
            [
                dbc.Alert(
                    [
                        "Zur Station ",
                        html.A("Anmeldung", href="/anmeldung", className="alert-link"),
                    ],
                    color="primary",
                ),
            ]
        ),
        html.Br(),
        html.P(
            'Diese Applikation verwendet Machine Learning Klassifizierungsalgorithmen zur automatisierten Qualitätssicherung.'),
    ], className='twelve columns'),

])

