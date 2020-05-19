import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from apps import commonmodules
from app import app


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
    commonmodules.get_menu(),
    html.Br(),
    html.H3('Station Spanen', style={'font-weight': 'bold'}),
    html.Hr(),
    dcc.Tabs(id='tabs-spanen', value='tab-1', children=[
        dcc.Tab(label='Übersicht', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Details', value='tab-2', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-spanen-content'),
    html.Br(),
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
        className="row"
    ),
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
        ]
    ),
    html.Br(),
    html.Div(
    [
        dbc.Button(
            "Weitere Optionen",
            id="collapse-button",
            className="mb-3",
            color="primary",
        ),
        dbc.Collapse(
            dbc.Card(
                # dbc.CardBody("This content is hidden in the collapse"),
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
            ),
            id="collapse",

        ),
    ],
    ),
    html.Br(),
    html.Div(
        [
            dbc.Row(dbc.Col(html.Div("A single column"))),
            dbc.Row(
                [
                    dbc.Col(html.Div("One of three columns")),
                    dbc.Col(html.Div("One of three columns")),
                    dbc.Col(html.Div("One of three columns")),
                ]
            ),
        ]
    )
])

@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# @app.callback(Output('tabs-spanen-content', 'children'),
#               [Input('tabs-spanen', 'value')])
# def render_content(tab):
#     if tab == 'tab-1':
#         return html.Div([
#             html.H3('Tab content 1')
#         ])
#     elif tab == 'tab-2':
#         return html.Div([
#             html.H3('Tab content 2')
#         ])