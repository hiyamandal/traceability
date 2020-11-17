import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from apps import commonmodules
from app import app


# layout for page anmeldung
layout = html.Div([
    commonmodules.get_header(),
    html.Br(),
    html.Div([
        html.H3('Station Anmeldung', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
        html.Hr(),
        html.Div(
        [
            html.A(dbc.Button("Anmeldung", color="primary", className="mr-1", id="button-anmeldung",
                              style={'font-size': '100%'},), href="/spanen"),
        ]),
        html.Br(),
        html.Div(id="anmeldung"),
    ], className='twelve columns'),

])

# callback for button
@app.callback([
          Output('anmeldung', 'children'),
          Output('anmeldung', 'style'),
      ],[
          Input('button-anmeldung','n_clicks')
       ])
def update_anmeldung(n_clicks):
    if n_clicks == None:
        n_clicks = 0
    if n_clicks % 2 ==0:
        child = html.H4('Bitte neues Bauteil anmelden!'),
    else:
        child = html.Div(
            [
                html.H4('Bauteil ist angemeldet!'),
                html.Br(),
                dbc.Alert(
                    [
                        "Zur Station ",
                        html.A("Spanen", href="/spanen", className="alert-link"),
                    ],
                    color="primary",
                    style={'font-size': '150%'},
                ),
            ])
    return [child, None]