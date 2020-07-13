import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pandas

import plotly.graph_objects as go

from apps import commonmodules
from app import app

from scipy.stats import multinomial, uniform, expon
import numpy as np

global_list_lackieren = [0]

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

config = dict({'scrollZoom': True})

# page content
layout = html.Div([
    commonmodules.get_header(),
    html.Br(),
    html.H3('Station Lackieren', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
    html.Hr(),
    html.H4('Klassifizierungsergebnisse', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        html.Div([dcc.Graph(id='fig1_callback_lackieren'),], style={'textAlign': 'center', 'width': '100%'}),
                    ]),
                    dbc.Row([
                        html.Div([dcc.Graph(id='fig2_callback_lackieren'),], style={'textAlign': 'center', 'width': '100%'}),
                    ]),
                    dbc.Row([
                        html.Div([dcc.Graph(id='fig3_callback_lackieren'), ],
                                 style={'textAlign': 'center', 'width': '100%'}),
                    ]),
                ],
                className = "pretty_container",),
                dbc.Row(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        id="category_lackieren",
                                        className="pretty_container",
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Genauigkeit", style={"text-align": "center",'font-weight': 'bold'}),
                                            html.Br(),
                                            html.H2("0.95", style={"text-align": "center",'font-weight': 'bold'}, ),
                                            html.Br(),
                                        ],
                                        id="accuracy",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        [
                                            html.H4("F1-Score",
                                                    style={"text-align": "center", 'font-weight': 'bold'}),
                                            html.Br(),
                                            html.H2("0.93", style={"text-align": "center", 'font-weight': 'bold'}, ),
                                            html.Br(),
                                        ],
                                        id="f1score",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Präzision", style={"text-align": "center",'font-weight': 'bold'}),
                                            html.Br(),
                                            html.H5("Gutteil: 0.97", style={"text-align": "center"}, ),
                                            html.H5("Nachbearbeiten: 0.90", style={"text-align": "center"}, ),
                                            html.H5("Ausschuss: 0.95", style={"text-align": "center"}, ),
                                        ],
                                        id="precision",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Sensitivität", style={"text-align": "center",'font-weight': 'bold'}),
                                            html.Br(),
                                            html.H5("Gutteil: 0.98", style={"text-align": "center"}, ),
                                            html.H5("Nachbearbeiten: 0.83", style={"text-align": "center"}, ),
                                            html.H5("Ausschuss: 0.95", style={"text-align": "center"}, ),
                                        ],
                                        id="sensitivity",
                                        className="pretty_container"
                                    ),
                                ],
                                id="fiveContainer",
                            )
                        ],
                        id="infoContainer",
                ),
            ],
            width=7),
            dbc.Col([
                html.Div([
                    html.H5("Testdaten", style={"text-align": "center",'font-weight': 'bold'}),
                    html.Div([dcc.Graph(id='fig4_callback_lackieren'),], style={'textAlign': 'center', 'width': '100%'}),
                ],
                className = "pretty_container",),
            ],
            width=5),
        ],
        align="center",
        ),
    ], className="flex-hd-row, flex-column align-items-center p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"), # d-flex
    html.Hr(style={'height': '30px', 'font-weight': 'bold'}),
    html.H4('Handlungsempfehlung', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div(
        className="flex-hd-row, flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm", # d-flex
        id="handlungsempfehlung_lackieren"),
    html.Br(),
    html.Div(
    [
        dbc.Button(
            "Weitere Handlungsoptionen",
            id="collapse-button-options_lackieren",
            className="mb-3",
            color="primary",
            style={'font-size': '100%'},
        ),
        dbc.Collapse(
            html.Div([
                html.Div(
                    [
                        dbc.Alert(
                            [
                                "Weiter zur Station ",
                                html.A("Montage", href="/montage", className="alert-link"),
                            ],
                            color="success",
                            style={'font-size': '150%'},
                        ),
                        dbc.Alert(
                            [
                                "Nachbearbeitung an Station  ",
                                html.A("Lackieren", href="/lackieren", className="alert-link"),
                            ],
                            color="warning",
                            style={'font-size': '150%'},
                        ),
                        dbc.Alert(
                            [
                                "Klassifiziere Bauteil als ",
                                html.A("Ausschuss", href="/", className="alert-link"),
                            ],
                            color="danger",
                            style={'font-size': '150%'},
                        ),
                    ]
                ),
            ], className="flex-hd-row, flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  #  d-flex
            id="collapse-options_lackieren",
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
                id="collapse-button-details_lackieren",
                className="mb-3",
                color="primary",
                style={'font-size': '100%'},
            ),
            dbc.Collapse(
                html.Div([
                    dcc.Tabs(id='tabs-lackieren', value='tab-1', children=[
                    dcc.Tab(label='Schnittansicht Klassifizierungsergebnisse', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col(
                                        html.Img(src=app.get_asset_url('lackieren/train_lackieren_power.png')),
                                        align="center",
                                    ),
                                    dbc.Col(
                                        html.Img(src=app.get_asset_url('lackieren/train_lackieren_pressure.png')),
                                        align="center",
                                    ),
                                    dbc.Col(
                                        html.Img(src=app.get_asset_url('lackieren/train_lackieren_temp.png')),
                                        align="center",
                                    ),
                                ], align="center",)
                            ], className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                        ]),
                        dcc.Tab(label='Konfusionsmatrix', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                            html.Img(src=app.get_asset_url('lackieren/confusion_absolut_lackieren.png'))],
                                            style={'width': '100 %', "text-align": "center", 'diplay': "flex"},
                                        ),
                                    ),

                                    dbc.Col(
                                        html.Div([
                                            html.Img(src=app.get_asset_url('confusion_normalised_lackieren.png'))],
                                            style={'width': '100 %', "text-align": "center", 'diplay': "flex"},
                                        ),
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
                id="collapse-details_lackieren",
            ),
        ],
    ),
    html.Hr(),
])

# callbacks
@app.callback(
    Output("collapse-options_lackieren", "is_open"),
    [Input("collapse-button-options_lackieren", "n_clicks")],
    [State("collapse-options_lackieren", "is_open")],
)
def toggle_collapse_options(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-details_lackieren", "is_open"),
    [Input("collapse-button-details_lackieren", "n_clicks")],
    [State("collapse-details_lackieren", "is_open")],
)
def toggle_collapse_options(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback([
          Output('fig1_callback_lackieren', 'figure'),
          Output('fig2_callback_lackieren', 'figure'),
          Output('fig3_callback_lackieren', 'figure'),
          Output('fig4_callback_lackieren', 'figure'),
          Output('category_lackieren', 'children'),
          Output('category_lackieren', 'style'),
          Output('handlungsempfehlung_lackieren', 'children')
      ],[
          Input('url','pathname')
       ])
def update_inputs(pathname):

    if pathname == '/lackieren':

        # append global list
        index = global_list_lackieren[-1]
        while global_list_lackieren[-1] == index:
            # erzeuge randomisierte prozessgrößen
            draw = multinomial.rvs(1, [0.4, 0.3, 0.3])  # 0.5, 0.3, 0.2
            index = int(np.where(draw == 1)[0])
        global_list_lackieren.append(index)

        # gutteil, leisung P und kraft F sind OK
        if index == 0:

            mean = [1.5, 6, 32]
            cov = [[1, 0, 0], [0, 2, 0], [0, 0, 75]]

            bool = True
            while bool:
                L, D, T = np.random.multivariate_normal(mean, cov, 1).T
                if L > 0 and L < 3 and D > 4 and D < 8 and T > 18 and T < 45:
                    bool = False

        # # nacharbeit
        # elif index == 1:
        #     draw2 = multinomial.rvs(1, [0.5, 0.5])
        #     index2 = np.where(draw2 == 1)
        #
        #     # Leistung zu hoch
        #     if index2[0] == 0:
        #         P = expon.rvs(3.5, 0.3, size=1)
        #         F = uniform.rvs(3.5, 1.5, size=1)
        #
        #     # Kraft zu niedrig oder zu hoch
        #     elif index2[0] == 1:
        #
        #         draw3 = multinomial.rvs(1, [0.5, 0.5])
        #         index3 = np.where(draw3 == 1)
        #
        #         # Kraft zu niedrig
        #         if index3[0] == 0:
        #             P = uniform.rvs(0.5, 1, size=1)
        #             F = uniform.rvs(0, 0.25, size=1)
        #
        #         # Kraft zu hoch
        #         elif index3[0] == 1:
        #             P = uniform.rvs(2, 0.5, size=1)
        #             F = expon.rvs(5.5, 0.2, size=1)
        #
        # # ausschuss: leistung und kraft zu hoch
        # elif index == 2:
        #     P = expon.rvs(3.5, 0.3, size=1)  # loc, scale, size
        #     F = expon.rvs(5.5, 0.2, size=1)

        # preliminary values
        D = 6
        T = 30
        P = 1.5

        # plot für prozessgrößen
        fig1_callback = go.Figure()

        fig1_callback.add_trace(go.Indicator(
            mode="number+gauge", value=D, number={'font': {'size': 30}},
            # delta = {'reference': 200},
            domain={'x': [0.25, 1], 'y': [0.3, 0.7]},
            title={'text': "Druck in bar", 'font': {'size': 20}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 12]},
                'threshold': {
                    'line': {'color': 'black', 'width': 5},
                    'thickness': 0.75,
                    'value': 8},
                'steps': [
                    {'range': [0, 4], 'color': "lightgray"},
                    {'range': [4, 8], 'color': "green"},
                    {'range': [8, 12], 'color': "lightgray"}],
                'bar': {
                    'color': 'black'}
                        },
        ),
        )

        fig1_callback.update_layout(autosize=True, height=100, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                           paper_bgcolor="#f9f9f9", )

        fig2_callback = go.Figure()

        fig2_callback.add_trace(go.Indicator(
            mode="number+gauge", value=T, number={'font': {'size': 30}},
            # delta = {'reference': 200},
            domain={'x': [0.25, 1], 'y': [0.3, 0.7]},
            title={'text': "Temperatur in °C", 'font': {'size': 20}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 60]},
                'threshold': {
                    'line': {'color': 'black', 'width': 5},
                    'thickness': 0.75,
                    'value': 45},
                'steps': [
                    {'range': [0, 18], 'color': "lightgray"},
                    {'range': [18, 45], 'color': "green"},
                    {'range': [45, 60], 'color': "lightgray"}],
                'bar': {
                    'color': 'black'}
                        },
        ),
        )

        fig2_callback.update_layout(autosize=True, height=100, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                           paper_bgcolor="#f9f9f9", )

        fig3_callback = go.Figure()
        fig3_callback.add_trace(go.Indicator(
            mode="number+gauge", value=P, number={'font': {'size': 30}},
            # delta = {'reference': 200},
            domain={'x': [0.25, 1], 'y': [0.3, 0.7]},
            title={'text': "Leistung in kW", 'font': {'size': 20}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 6]},
                'threshold': {
                    'line': {'color': 'black', 'width': 5},
                    'thickness': 0.75,
                    'value': 3},
                'steps': [
                    {'range': [0, 3], 'color': "green"},
                    {'range': [3, 6], 'color': "lightgray"}],
                'bar': {'color': 'black'},
            },
        )
        )
        fig3_callback.update_layout(autosize=True, height=100, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                                    paper_bgcolor="#f9f9f9", )
        n_train = 3000
        with open("assets/lackieren/lackieren_knn_data_"+str(n_train)+".csv") as mycsv:
            count = 0
            for line in mycsv:
                if count == 0:
                    cat = line
                if count == 1:
                    L = line
                if count == 2:
                    D = line
                if count == 3:
                    T = line
                if count == 4:
                    break
                count += 1

        # transform strings to numpy lists, while conserving np.array dimensions
        import re, ast

        cat = re.sub('\s+', '', cat)
        z_scatter = ast.literal_eval(cat)

        L = re.sub('\s+', '', L)
        L = ast.literal_eval(L)

        D = re.sub('\s+', '', D)
        D = ast.literal_eval(D)

        T = re.sub('\s+', '', T)
        T = ast.literal_eval(T)


        # dataframe for scatter data
        df = pandas.DataFrame({'Leistung in kW': L, 'Druck in bar': D, 'Temperatur in °C': T, 'z_scatter': z_scatter})

        # define marker colors
        marker_color = {
            2.0: 'red',
            1.0: 'orange',
            0.0: 'green'
        }

        marker_colors = [marker_color[k] for k in df['z_scatter'].values]

        fig4_callback = go.Figure()

        # scatter plot of training data
        fig4_callback.add_trace(go.Scatter3d(
            x=df['Leistung in kW'],
            y=df['Druck in bar'],
            z=df['Temperatur in °C'],
            mode='markers',
            name='Trainingsdaten',
            marker_color=marker_colors,
            marker=dict(
                opacity = 1,
                # symbol = 'x',
                # color='rgb(255, 178, 102)',
                size=4,
                line=dict(
                    color='DarkSlateGrey',
                    width=0.1
                )
            )
        ))

        # # scatter plot of single new test point
        # fig4_callback.add_trace(go.Scatter(
        #     x=np.asarray(F[0]),
        #     y=np.asarray(P[0]),
        #     mode='markers',
        #     name='Prozesskräfte',
        #     marker_color='magenta',
        #     marker=dict(
        #         symbol='cross',
        #         # color='rgb(255, 178, 102)',
        #         size=30,
        #         line=dict(
        #             color='DarkSlateGrey',
        #             width=2
        #         )
        #     )
        # ))

        fig4_callback.update_layout(
            margin = {'l': 0, 'r': 0, 't': 0, 'b': 0},
            scene=dict(
                xaxis=dict(
                    title='Leistung in kW'),
                yaxis=dict(
                    title='Druck in bar'),
                zaxis=dict(
                    title='Temperatur in °C'), ),
            # xaxis_title='Leistung in kN',
            # yaxis_title='Druck in bar',
            # zaxis_title='Temperatur in °C',
            # font=dict(
            #     size=18,
            # ),
        )


        # # load model from pickle
        # import pickle
        #
        # clf = pickle.load(open("assets/lackieren_knn_model_"+str(n_train)+".sav", 'rb'))
        # z = clf.predict(np.c_[F[0], P[0]])
        # if z[0] == 0:
        cat_string = "Gutteil"
        cat_color = "green"
        empfehlung_alert = dbc.Alert(
                            [
                                "Handlungsempfehlung: Weiter zur Station ",
                                html.A("Montage", href="/montage", className="alert-link"),
                            ],
                            color="success",
                            style={'font-size': '150%'},
                            ),
        # elif z[0] == 1:
        #     cat_string = "Nachbearbeiten"
        #     cat_color = "orange"
        #     empfehlung_alert = dbc.Alert(
        #                         [
        #                             "Handlungsempfehlung: Nachbearbeitung an Station ",
        #                             html.A("Lackieren", href="lackieren", className="alert-link"),
        #                         ],
        #                         color="warning",
        #                         ),
        #
        # elif z[0] == 2:
        #     cat_string = "Ausschuss"
        #     cat_color = "darkred"
        #     empfehlung_alert = dbc.Alert(
        #                         [
        #                             "Handlungsempfehlung: Klassifiziere Bauteil als ",
        #                             html.A("Ausschuss", href="/", className="alert-link"),
        #                         ],
        #                         color="danger",
        #                         ),
        #
        category = html.Div(
                        [
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H4(cat_string, style={"text-align": "center", "color": "white",
                                                   'font-weight': 'bold', 'vertical-align': 'middle'}, id="3cat"),
                         ],
                    )

        style = {"background-color": cat_color, 'display': 'inline-block', "text-align": "center", 'vertical-align': 'middle'}

        return [fig1_callback, fig2_callback, fig3_callback, fig4_callback, category, style, empfehlung_alert]

    else:
        fig1_callback = go.Figure()
        fig2_callback = go.Figure()
        fig3_callback = go.Figure()
        fig4_callback = go.Figure()
        z = 0
        z = np.asarray(z)
        category= None
        style = None
        empfehlung_alert = None

        return [fig1_callback, fig2_callback, fig3_callback, fig4_callback, category, style, empfehlung_alert]
