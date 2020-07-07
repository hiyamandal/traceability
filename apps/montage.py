import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from apps import commonmodules
from app import app


layout = html.Div([
    commonmodules.get_header(),
    # html.Br(),
    # commonmodules.get_menu(),
    html.Br(),
    html.H3('Station Montage', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
    html.Hr(),
    html.H4('Bitte Montage des Bauteils durchführen!'),
    html.Br(),
    html.Div(
            [   dbc.ButtonGroup(
                    [dbc.Button("Bauteil iO", color="success", className="mr-5", id="button-io", style={'font-size': '100%'}),
                     dbc.Button("Bauteil niO", color="danger", className="mr-6", id="button-nio", style={'font-size': '100%'},)
                    ]
                ),], id="buttons_montage"),
    html.Div( id="montage-ende"),
])


@app.callback([
          Output('montage-ende', 'children'),
      ],[
          Input('button-io','n_clicks'),
       ])
def update_montage_io(n_clicks):
    if n_clicks != None :
        child = html.Div(
        [
            html.Br(),
            html.H4('Montage des Bauteils war erfolgreich!'),
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
        child = None
    return [child]

# @app.callback([
#           Output('montage-ende', 'children'),
#       ],[
#           Input('button-nio','n_clicks'),
#        ])
# def update_montage_nio(n_clicks):
#     child = html.Div(
#     [
#         html.Br(),
#         html.H4('Montage des Bauteils hat fehlgeschlagen!'),
#         html.Br(),
#         dbc.Alert(
#             [
#                 html.A("Neues Bauteil anmelden.", href="/", className="alert-link"),
#             ],
#             color="primary",
#             style={'font-size': '150%'},
#         ),
#     ])
#     return [child]

