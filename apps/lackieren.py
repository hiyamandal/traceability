import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pandas

import plotly.graph_objects as go

from apps import commonmodules
from app import app

import numpy as np
import pickle

global_index_lackieren = [0]
global_slider_lackieren = ['2000']
global_L_list_lackieren = []
global_D_list_lackieren = []
global_T_list_lackieren = []

# tab styles
tabs_styles = {
    'height': '54px',
    'font-size': '120%',
    'text-align':"center",
    'display': 'inline-block',
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'text-align': "center",
    'display': 'inline-block',
}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
    'text-align':"center",
    'display': 'inline-block',
}

config = dict({'scrollZoom': True})

# page content
layout = html.Div([
    commonmodules.get_header(),
    html.Br(),
    html.H4('Station Lackieren', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
    html.Hr(),
    html.H4('Klassifizierungsergebnisse', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        html.Div([
                            dcc.Graph(id='fig1_callback_lackieren'),
                        ], style={'textAlign': 'center', 'width': '100%'}),
                    ]),

                    dbc.Row([
                        html.Div(
                            [
                                dcc.Graph(id='fig2_callback_lackieren'),
                             ], style={'textAlign': 'center', 'width': '100%'}),

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
                                        id="accuracy_lackieren",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        id="f1score_lackieren",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        id="precision_lackieren",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        id="sensitivity_lackieren",
                                        className="pretty_container"
                                    ),
                                ],
                                id="fiveContainer",
                            )
                        ],
                        id="infoContainer",
                ),
            ],
            width=6),
            dbc.Col([
                html.Div([
                    html.H5("Vorhergesagte Klasse für Testdaten", style={"text-align": "center",'font-weight': 'bold'}),
                    html.Br(),
                    html.Div([
                        dcc.Graph(id='fig4_callback_lackieren'),
                        html.H6("Anzahl an Datenpunkten (Training + Test):", style={"text-align": "center",'font-weight': 'bold'}),
                        dcc.Slider(
                            id='dataset-slider_lackieren',
                            min=2000,
                            max=10000,
                            value=2000,
                            marks={
                                2000: '# data: 2000 ',
                                6000: '6000',
                                10000: '10000',
                            },
                            step=4000,
                        )
                    ], style={'textAlign': 'center', 'width': '100%'}),
                ],
                className = "pretty_container",),
            ],
            width=6),
        ],
        align="center",
        ),#flex-hd-row, flex-column
    ], className="flex-hd-row, flex-column align-items-center p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"), # d-flex
    html.Hr(style={'height': '30px', 'font-weight': 'bold'}),
    html.H5('Handlungsempfehlung', style={'font-weight': 'bold'}),
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
                            style={'font-size': '130%'},
                        ),
                        dbc.Alert(
                            [
                                "Nachbearbeitung an Station  ",
                                html.A("Lackieren", href="/lackieren", className="alert-link"),
                            ],
                            color="warning",
                            style={'font-size': '130%'},
                        ),
                        dbc.Alert(
                            [
                                "Klassifiziere Bauteil als ",
                                html.A("Ausschuss", href="/", className="alert-link"),
                            ],
                            color="danger",
                            style={'font-size': '130%'},
                        ),
                    ]
                ),
            ], className="flex-hd-row, flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
            id="collapse-options_lackieren",
        ),
    ],
    ),
    html.Hr(),
    html.H5('Detailinformationen', style={'font-weight': 'bold'}),
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
                        dcc.Tab(label='Schnittansicht Klassifizierungsergebnisse', style=tab_style,
                                selected_style=tab_selected_style, children=[
                                html.Div([
                                    dbc.Row([
                                        dbc.Col(
                                            html.Div([
                                            ],
                                                style={'width': '100 %', "text-align": "center", 'diplay': "flex"},
                                                id='lackieren_train_power',
                                            ),
                                        ),
                                        dbc.Col(
                                            html.Div([
                                            ],
                                                style={'width': '100 %', "text-align": "center", 'diplay': "flex"},
                                                id='lackieren_train_pressure',
                                            ),
                                        ),
                                        dbc.Col(
                                            html.Div([
                                            ],
                                                style={'width': '100 %', "text-align": "center", 'diplay': "flex"},
                                                id='lackieren_train_temp',
                                            ),
                                        ),
                                    ], align="center", )
                                ],
                                    className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),
                                # d-flex
                            ]),
                        dcc.Tab(label='Konfusionsmatrix', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                        ],
                                            style={'width': '100 %', "text-align": "center", 'diplay': "flex"},
                                            id='lackieren_confusion_absolute',
                                        ),
                                    ),
                                    dbc.Col(
                                        html.Div([
                                        ],
                                            style={'width': '100 %', "text-align": "center", 'diplay': "flex"},
                                            id='lackieren_confusion_normalised',
                                        ),
                                    ),
                                ], align="center", )
                            ], className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),
                            # d-flex
                        ]),
                        dcc.Tab(label='Wirtschaftliche Bewertung', style=tab_style, selected_style=tab_selected_style,
                                children=[
                                    html.Div([
                                        html.H5("Wirtschaftliche Bewertung einfügen.")
                                    ],
                                        className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),
                                    # d-flex
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
          Output('handlungsempfehlung_lackieren', 'children'),
          Output('accuracy_lackieren', 'children'),
          Output('f1score_lackieren', 'children'),
          Output('precision_lackieren', 'children'),
          Output('sensitivity_lackieren', 'children'),
          Output('lackieren_confusion_absolute', 'children'),
          Output('lackieren_confusion_normalised', 'children'),
          Output('lackieren_train_power', 'children'),
          Output('lackieren_train_pressure', 'children'),
          Output('lackieren_train_temp', 'children'),
      ],[
          Input('url','pathname'),
          Input('dataset-slider_lackieren','value')
       ])
def update_inputs(pathname, value):

    n_train = value
    # load model from pickle
    clf = pickle.load(open("assets/lackieren/lackieren_knn_model_" + str(n_train) + ".sav", 'rb'))

    print()
    if pathname == '/lackieren':

        #   append global url list
        old_slider_status = global_slider_lackieren[-1]
        global_slider_lackieren.append(str(value))
        new_slider_status = global_slider_lackieren[-1]
        reload_datapoint = True
        if new_slider_status != old_slider_status:
            reload_datapoint = False

        if reload_datapoint == True:
            # append global index list
            index = global_index_lackieren[-1]
            while global_index_lackieren[-1] == index:

                mean = [1.5, 6, 32]
                cov = [[1.5, 0, 0], [0, 5, 0], [0, 0, 120]] # 1,3,100

                L, D, T = np.random.multivariate_normal(mean, cov, 1).T
                L = np.abs(L)
                D = np.abs(D)
                T = np.abs(T)

                z = clf.predict(np.c_[L, D, T])
                index = int(z)

            global_index_lackieren.append(index)

            global_L_list_lackieren.append(L)
            global_D_list_lackieren.append(D)
            global_T_list_lackieren.append(T)

        # do prediction
        z = clf.predict(
            np.c_[global_L_list_lackieren[-1][0], global_D_list_lackieren[-1][0], global_T_list_lackieren[-1][0]])

        # load confusion matrix
        lackieren_confusion_absolute_callback = html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_confusion_absolute_' + str(n_train) + '.png'))
        ],)
        lackieren_confusion_normalised_callback = html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_confusion_normalised_' + str(n_train) + '.png'))
        ],)

        # load predictions along planes
        lackieren_train_power= html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_train_power_' + str(n_train) + '.png'))
        ],)
        lackieren_train_pressure = html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_train_pressure_' + str(n_train) + '.png'))
        ],)
        lackieren_train_temp= html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_train_temp_' + str(n_train) + '.png'))
        ],)

        # plot für prozessgrößen
        fig1_callback = go.Figure()

        fig1_callback.add_trace(go.Indicator(
            mode="number+gauge", value=global_D_list_lackieren[-1][0], number={'font': {'size': 30}},
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
            mode="number+gauge", value=global_T_list_lackieren[-1][0], number={'font': {'size': 30}},
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
        )
        )
        fig2_callback.update_layout(autosize=True, height=100, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                                    paper_bgcolor="#f9f9f9", )

        fig3_callback = go.Figure()
        fig3_callback.add_trace(go.Indicator(
            mode="number+gauge", value=global_L_list_lackieren[-1][0], number={'font': {'size': 30}},
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

        with open("assets/lackieren/lackieren_knn_data_"+str(n_train)+".csv") as mycsv:
            count = 0
            for line in mycsv:
                if count == 0:
                    z_test_load = line
                if count == 1:
                    data_test_load = line
                if count == 2:
                    z_train_load = line
                if count == 3:
                    data_train_load = line
                if count == 4:
                    report = line
                if count == 5:
                    break
                count += 1

        # transform strings to numpy lists, while conserving np.array dimensions
        import re, ast

        z_test_load = re.sub('\s+', '', z_test_load)
        z_test_scatter = ast.literal_eval(z_test_load)

        # z_train_load = re.sub('\s+', '', z_train_load)
        # z_train_scatter = ast.literal_eval(z_train_load)

        data_test_load = re.sub('\s+', '', data_test_load)
        data_test_load = np.asarray(ast.literal_eval(data_test_load))
        L_test_scatter = np.round(data_test_load[:, 0],2).tolist()
        D_test_scatter = np.round(data_test_load[:, 1],2).tolist()
        T_test_scatter = np.round(data_test_load[:, 2],2).tolist()

        # data_train_load = re.sub('\s+', '', data_train_load)
        # data_train_load = np.asarray(ast.literal_eval(data_train_load))
        # x_train_scatter = np.round(data_train_load[:, 0],2).tolist()
        # y_train_scatter = np.round(data_train_load[:, 1],2).tolist()

        report = re.sub('\s+', '', report)
        report = ast.literal_eval(report)

        # get accuracy metrics from report
        accuracy = np.round(report['accuracy'], 2)
        f1_score = np.round(report['macroavg']['f1-score'], 2)

        precision_gutteil = np.round(report['0.0']['precision'], 2)
        sensitivity_gutteil = np.round(report['0.0']['recall'], 2)
        precision_nachbearbeiten = np.round(report['1.0']['precision'], 2)
        sensitivity_nachbearbeiten = np.round(report['1.0']['recall'], 2)
        precision_ausschuss = np.round(report['2.0']['precision'], 2)
        sensitivity_ausschuss = np.round(report['2.0']['recall'], 2)

        # update info boxes
        accuracy_callback = html.Div(
            [
                html.H6("Genauigkeit", style={"text-align": "center", 'font-weight': 'bold'}),
                html.Br(),
                html.H4(str(accuracy), style={"text-align": "center", 'font-weight': 'bold'}, ),
                html.Br(),
            ],
        ),
        f1_score_callback = html.Div(
            [
                html.H6("F1-Score",
                        style={"text-align": "center", 'font-weight': 'bold'}),
                html.Br(),
                html.H4(str(f1_score), style={"text-align": "center", 'font-weight': 'bold'}, ),
                html.Br(),
            ],
        ),
        precision_callback = html.Div(
            [
                html.H6("Präzision", style={"text-align": "center", 'font-weight': 'bold'}),
                html.Br(),
                html.P("Gutteil: " + str(precision_gutteil), style={"text-align": "center"}, ),
                html.P("Nachbearb.: " + str(precision_nachbearbeiten), style={"text-align": "center"}, ),
                html.P("Ausschuss: " + str(precision_ausschuss), style={"text-align": "center"}, ),
            ],
        ),
        sensitivity_callback = html.Div(
            [
                html.H6("Sensitivität", style={"text-align": "center", 'font-weight': 'bold'}),
                html.Br(),
                html.P("Gutteil: " + str(sensitivity_gutteil), style={"text-align": "center"}, ),
                html.P("Nachbearb.: " + str(sensitivity_nachbearbeiten), style={"text-align": "center"}, ),
                html.P("Ausschuss: " + str(sensitivity_ausschuss), style={"text-align": "center"}, ),
            ],
        ),

        # dataframe for scatter data
        df_test = pandas.DataFrame({'L_test_scatter': L_test_scatter, 'D_test_scatter': D_test_scatter, 'T_test_scatter': T_test_scatter,'z_test_scatter': z_test_scatter})
        # df_train = pandas.DataFrame({'x_train_scatter': x_train_scatter, 'y_train_scatter': y_train_scatter, 'z_train_scatter': z_train_scatter})

        # define marker colors
        marker_color = {
            2.0: 'lightcoral',
            1.0: 'orange',
            0.0: 'lightgreen'
        }

        marker_colors = [marker_color[k] for k in df_test['z_test_scatter'].values]

        fig4_callback = go.Figure()

        # scatter plot of test data
        fig4_callback.add_trace(go.Scatter3d(
            x=df_test['L_test_scatter'],
            y=df_test['D_test_scatter'],
            z=df_test['T_test_scatter'],
            mode='markers',
            name='Testdaten',
            showlegend=False,
            marker_color=marker_colors,
            marker=dict(
                opacity = 0.5,
                # symbol = 'x',
                # color='rgb(255, 178, 102)',
                size=4,
                line=dict(
                    color='DarkSlateGrey',
                    width=0.1
                )
            )
        ))

        # scatter plot of single new test point
        fig4_callback.add_trace(go.Scatter3d(
            x=np.asarray(global_L_list_lackieren[-1][0]),
            y=np.asarray(global_D_list_lackieren[-1][0]),
            z=np.asarray(global_T_list_lackieren[-1][0]),
            mode='markers',
            name='Prozessparameter',
            marker_color='magenta',
            marker=dict(
                symbol='cross',
                # color='rgb(255, 178, 102)',
                size=30,
                line=dict(
                    color='DarkSlateGrey',
                    width=2
                )
            )
        ))

        fig4_callback.update_layout(
            margin = {'l': 0, 'r': 0, 't': 0, 'b': 0},
            scene=dict(
                xaxis=dict(
                    title='Leistung in kW'),
                yaxis=dict(
                    title='Druck in bar'),
                zaxis=dict(
                    title='Temperatur in °C'), ),
                legend=dict(
                    yanchor="top",
                    y=0.95,
                    xanchor="left",
                    x=0.70
                )
        )

        if z[0] == 0:
            cat_string = "Gutteil"
            cat_color = "green"
            empfehlung_alert = dbc.Alert(
                                [
                                    "Handlungsempfehlung: Weiter zur Station ",
                                    html.A("Montage", href="/montage", className="alert-link"),
                                ],
                                color="success",
                                style={'font-size': '130%'},
                                ),
        elif z[0] == 1:
            cat_string = "Nachbearbeiten"
            cat_color = "orange"
            empfehlung_alert = dbc.Alert(
                                [
                                    "Handlungsempfehlung: Nachbearbeitung an Station ",
                                    html.A("Lackieren", href="lackieren", className="alert-link"),
                                ],
                                color="warning",
                                style={'font-size': '130%'},
                                ),

        elif z[0] == 2:
            cat_string = "Ausschuss"
            cat_color = "darkred"
            empfehlung_alert = dbc.Alert(
                                [
                                    "Handlungsempfehlung: Klassifiziere Bauteil als ",
                                    html.A("Ausschuss", href="/", className="alert-link"),
                                ],
                                color="danger",
                                style={'font-size': '130%'},
                                ),

        category = html.Div(
                        [
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H5(cat_string, style={"text-align": "center", "color": "white",
                                                   'font-weight': 'bold', 'vertical-align': 'middle'}, id="3cat"),
                         ],
                    )

        style = {"background-color": cat_color, 'display': 'inline-block', "text-align": "center", 'vertical-align': 'middle'}

        return [fig1_callback, fig2_callback, fig3_callback, fig4_callback, category, style, empfehlung_alert, accuracy_callback,
                f1_score_callback, precision_callback, sensitivity_callback,
                lackieren_confusion_absolute_callback, lackieren_confusion_normalised_callback,
                lackieren_train_power, lackieren_train_pressure, lackieren_train_temp]

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
        accuracy_callback = None
        f1_score_callback = None
        precision_callback = None
        sensitivity_callback = None
        lackieren_confusion_absolute_callback = None
        lackieren_confusion_normalised_callback = None
        lackieren_train_power = None
        lackieren_train_pressure = None
        lackieren_train_temp = None

        return [fig1_callback, fig2_callback, fig3_callback, fig4_callback, category, style, empfehlung_alert, accuracy_callback,
                f1_score_callback, precision_callback, sensitivity_callback,
                lackieren_confusion_absolute_callback, lackieren_confusion_normalised_callback,
                lackieren_train_power, lackieren_train_pressure, lackieren_train_temp]