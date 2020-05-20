import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_flexbox_grid as dfx

import plotly.graph_objects as go

from apps import commonmodules
from app import app

# tab styles
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

# plot für prozessgrößen
fig1 = go.Figure()
fig1.add_trace(go.Indicator(
    mode = "number+gauge", value = 3,
    delta = {'reference': 200},
    domain = {'x': [0.25, 1], 'y': [0.08, 0.25]},
    title = {'text': "Kraft in kN"},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [0, 7]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': 5},
        'steps': [
            {'range': [0, 0.5], 'color': "lightgray"},
            {'range': [0.5, 5], 'color': "lightgreen"} ,
            {'range': [5, 7], 'color': "lightgray"}],
        'bar': {'color': "black"}}),
)
fig1.update_layout(grid = {'rows': 2, 'columns': 1, 'pattern': "independent"}, height = 125,  margin = {'t':0, 'b':0, 'l':0, 'r':0}, )

fig2 = go.Figure()
fig2.add_trace(go.Indicator(
    mode = "number+gauge", value = 2,
    delta = {'reference': 200},
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': "Leistung in kW"},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [0, 5]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': 3},
        'steps': [
            {'range': [0, 3], 'color': "lightgreen"},
            {'range': [3, 5], 'color': "lightgray"}],
        'bar': {'color': "black"}})
)
fig2.update_layout(grid = {'rows': 2, 'columns': 1, 'pattern': "independent"}, height = 125,  margin = {'t':0, 'b':0, 'l':0, 'r':0}, ) # margin = {'t':0, 'b':0, 'l':0},  ,title="Prozessdaten an Station Spanen"

# page content
layout = html.Div([
    commonmodules.get_header(),
    # html.Br(),
    # commonmodules.get_menu(),
    html.Br(),
    html.H3('Station Spanen', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
    html.Hr(),
    html.H4('Klassifizierungsergebnisse', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    html.Div([dcc.Graph(figure=fig1),]),
                    html.Div([dcc.Graph(figure=fig2),]),
                ]),
                dbc.Row(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P("No. of Wells"),
                                            html.H6(
                                                id="well_text",
                                                className="info_text"
                                            )
                                        ],
                                        id="wells",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        [
                                            html.P("Gas"),
                                            html.H6(
                                                id="gasText",
                                                className="info_text"
                                            )
                                        ],
                                        id="gas",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        [
                                            html.P("Oil"),
                                            html.H6(
                                                id="oilText",
                                                className="info_text"
                                            )
                                        ],
                                        id="oil",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        [
                                            html.P("Water"),
                                            html.H6(
                                                id="waterText",
                                                className="info_text"
                                            )
                                        ],
                                        id="water",
                                        className="pretty_container"
                                    ),
                                ],
                                id="tripleContainer",
                            )
                        ],
                        id="infoContainer",
                        # className="row",
                        # style={"height": "21vh"}
                ),
            ],),
            dbc.Col([

                html.Img(src=app.get_asset_url('test_spanen.png'))
            ],),
        ], align="center" ),
    ], className="flex-hd-row, flex-column align-items-center p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"), # d-flex
    html.Hr(style={'height': '30px', 'font-weight': 'bold'}),
    html.H4('Handlungsempfehlung', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div(
        [
            dbc.Alert(
                [
                    "Handlungsempfehlung: Weiter zur Station ",
                    html.A("Lackieren", href="/lackieren", className="alert-link"),
                ],
                color="success",
            ),
            dbc.Alert(
                [
                    "Handlungsempfehlung: Nachbearbeitung an Station  ",
                    html.A("Spanen", href="/spanen", className="alert-link"),
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
    ], className="flex-hd-row, flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"), # d-flex
    html.Br(),
    html.Div(
    [
        dbc.Button(
            "Weitere Handlungsoptionen",
            id="collapse-button-options",
            className="mb-3",
            color="primary",
        ),
        dbc.Collapse(
            html.Div([
                html.Div(
                    [
                        dbc.Alert(
                            [
                                "Weiter zur Station ",
                                html.A("Lackieren", href="/lackieren", className="alert-link"),
                            ],
                            color="success",
                        ),
                        dbc.Alert(
                            [
                                "Nachbearbeitung an Station  ",
                                html.A("Spanen", href="/spanen", className="alert-link"),
                            ],
                            color="warning",
                        ),
                        dbc.Alert(
                            [
                                "Klassifiziere Bauteil als ",
                                html.A("Ausschuss", href="/", className="alert-link"),
                            ],
                            color="danger",
                        ),
                    ]
                ),
            ], className="flex-hd-row, flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  #  d-flex
            id="collapse-options",

        ),
    ],
    ),
    html.Hr(),
    html.H4('Detailinformationen', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div(
        [
            dbc.Button(
                "Details ein-/ausblenden",
                id="collapse-button-details",
                className="mb-3",
                color="primary",
            ),
            dbc.Collapse(
                html.Div([
                    dcc.Tabs(id='tabs-spanen', value='tab-1', children=[
                        dcc.Tab(label='Konfusionsmatrix', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col(
                                        html.Img(src=app.get_asset_url('confusion_absolute.png')),
                                        align="center",
                                    ),
                                    dbc.Col(
                                        html.Img(src=app.get_asset_url('confusion_normalised.png')),
                                        align="center",
                                    ),
                                ], align="center",)
                            ], className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                        ]),
                        dcc.Tab(label='Wirtschaftliche Bewertung', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                html.H5("Wirtschaftliche Bewertung einfügen.")
                            ], className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                        ]),
                    ], style=tabs_styles),
                ], className="flex-column, flex-hd-row p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                id="collapse-details",
            ),
        ],
    ),
    html.Hr(),
    # html.Div(
    #     [
    #         dbc.Button(
    #             "Konfusionsmatrix",
    #             id="collapse-button-confusion",
    #             className="mb-3",
    #             color="primary",
    #         ),
    #         dbc.Collapse(
    #                 html.Div([
    #                     dbc.Row([
    #                         dbc.Col(
    #                             html.Img(src=app.get_asset_url('confusion_absolute.png')),
    #                             align="center",
    #                         ),
    #                         dbc.Col(
    #                             html.Img(src=app.get_asset_url('confusion_normalised.png')),
    #                             align="center",
    #                         ),
    #                     ],
    #                         align="center",
    #                     )
    #                 ], className="flex-column, flex-hd-row p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"), # d-flex
    #             id="collapse-confusion",
    #
    #         ),
    #     ],
    # ),
    # html.Div(
    #     [
    #         dbc.Button(
    #             "Wirtschaftliche Bewertung",
    #             id="collapse-button-commercial",
    #             className="mb-3",
    #             color="primary",
    #         ),
    #         dbc.Collapse(
    #             dbc.Card(
    #                 dbc.CardBody("Füge wirtschaftliche Bewertung hinzu."),
    #             ),
    #             id="collapse-commercial",
    #
    #         ),
    #     ],
    # ),
    # html.Hr(),
])

# callbacks
@app.callback(
    Output("collapse-options", "is_open"),
    [Input("collapse-button-options", "n_clicks")],
    [State("collapse-options", "is_open")],
)
def toggle_collapse_options(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-details", "is_open"),
    [Input("collapse-button-details", "n_clicks")],
    [State("collapse-details", "is_open")],
)
def toggle_collapse_options(n, is_open):
    if n:
        return not is_open
    return is_open

# @app.callback(
#     Output("collapse-confusion", "is_open"),
#     [Input("collapse-button-confusion", "n_clicks")],
#     [State("collapse-confusion", "is_open")],
# )
# def toggle_collapse_details(n, is_open):
#     if n:
#         return not is_open
#     return is_open
#
# @app.callback(
#     Output("collapse-commercial", "is_open"),
#     [Input("collapse-button-commercial", "n_clicks")],
#     [State("collapse-commercial", "is_open")],
# )
# def toggle_collapse_commercial(n, is_open):
#     if n:
#         return not is_open
#     return is_open


