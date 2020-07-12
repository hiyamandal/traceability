import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from apps import commonmodules
from app import app


layout = html.Div([
    commonmodules.get_header(),
    html.Br(),
    html.H3('Station Montage', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
    html.Hr(),
    html.Div(
            [   dbc.ButtonGroup(
                    [dbc.Button("Bauteil iO", color="success", className="mr-5", id="button_io", style={'font-size': '100%'}, n_clicks_timestamp='0'),
                     dbc.Button("Bauteil niO", color="danger", className="mr-6", id="button_nio", style={'font-size': '100%'}, n_clicks_timestamp='0')
                    ]
                ),], id="buttons_montage"),
    html.Div( id="montage-ende"),
])


@app.callback([
          Output('montage-ende', 'children'),
      ],[
          Input('button_io','n_clicks_timestamp'),
          Input('button_nio', 'n_clicks_timestamp'),
       ])
def update_montage_io(button_io,button_nio):
    if int(button_io) == 0 and int(button_nio) == 0:
        child = html.Div(
            [
            html.Br(),
            html.H4('Bitte Montage des Bauteils durchführen!'),
            html.Br(),
            ]
        )
    elif int(button_io) > int(button_nio):
        child = html.Div(
        [
            html.Br(),
            html.H4('Die Montage des Bauteils war erfolgreich!'),
            html.Br(),
            dbc.Alert(
                [
                    html.A("Neues Bauteil anmelden.", href="/", className="alert-link"),
                ],
                color="primary",
                style={'font-size': '150%'},
            ),
        ])
    else:
        child = html.Div(
            [
                html.Br(),
                html.H4('Die Montage des Bauteils ist fehlgeschlagen!'),
                html.Br(),
                dbc.Alert(
                    [
                        html.A("Neues Bauteil anmelden.", href="/", className="alert-link"),
                    ],
                    color="primary",
                    style={'font-size': '150%'},
                ),
            ])
    return [child]



