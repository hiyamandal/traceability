import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_flexbox_grid as dfx

import numpy as np
import plotly.graph_objs as go

from app import app
from apps import commonmodules


N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

tabs_styles = {
    'height': '54px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold'
}

layout = html.Div([
    commonmodules.get_header(),
    html.Br(),
    commonmodules.get_menu(),
    html.Br(),
    html.H3('Station Lackieren', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
    html.Hr(),
    html.Div(
        [
            dbc.Alert(
                [
                    "Handlungsempfehlung: Weiter zur Station ",
                    html.A("Montage", href="/montage", className="alert-link"),
                ],
                color="success",
            ),
            dbc.Alert(
                [
                    "Handlungsempfehlung: Nachbearbeitung an Station  ",
                    html.A("Lackieren", href="/lackieren", className="alert-link"),
                ],
                color="warning",
            ),
            dbc.Alert(
                [
                    "Handlungsempfehlung: Klassifiziere Bauteil als ",
                    html.A("Ausschuss", href="/", className="alert-link"),
                ],
                color="danger",
            ),
        ]
    ),
    dfx.Grid(id='grid', fluid=True, children=[
        dfx.Row(children=[
            dfx.Col(xs=12, lg=3, children=[
                html.Div('Hello,'), html.Div('World!')
            ])
        ]),
        dfx.Row(id='row', children=[
            dfx.Col(id='col', xs=6, lg=2, children=html.Div('Hello, World!')),
            dfx.Col(xs=6, lg=2, children=html.Div('Hello, World!'))
        ])
    ])
])

# @app.callback(Output('tabs-lackieren-content', 'children'),
#               [Input('tabs-lackieren', 'value')])
# def render_content(tab):
#     if tab == 'tab-1':
#         return html.Div([
#             html.H3('Tab content 1')
#         ])
#     elif tab == 'tab-2':
#         return html.Div([
#             html.H3('Tab content 2')
#         ])