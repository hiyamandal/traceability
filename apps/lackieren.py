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

import os
from os import path
import re, ast

# saves last status of prediction result (ok / rework / scrap)
global_index_lackieren = [0]

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

# page content of page lackieren
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
                # bar plot of process parameters, updated by callbacks
                html.Div([
                    dbc.Row([
                        html.Div([
                            dcc.Graph(id='fig1_leistung_lackieren'),
                        ], style={'textAlign': 'center', 'width': '100%'}),
                    ]),

                    dbc.Row([
                        html.Div(
                            [
                                dcc.Graph(id='fig2_druck_lackieren'),
                             ], style={'textAlign': 'center', 'width': '100%'}),

                    ]),
                    dbc.Row([
                        html.Div([dcc.Graph(id='fig3_temperatur_lackieren'), ],
                                 style={'textAlign': 'center', 'width': '100%'}),
                    ]),
                ],
                className = "pretty_container",),
                # containers with evaluation (ok / rework / scrap) and accuracy measures (accuracy, f1score, precision, sensitivity)
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
            # plot with test data and newly generated data points
            # slider for increasing / decreasing training data
            dbc.Col([
                html.Div([
                    html.H5("Vorhergesagte Klasse für Testdaten", style={"text-align": "center",'font-weight': 'bold'}),
                    html.Br(),
                    html.Div([
                        dcc.Graph(id='fig4_classification_lackieren'),
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
        ),
    ], className="flex-hd-row, flex-column align-items-center p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),
    html.Hr(style={'height': '30px', 'font-weight': 'bold'}),
    # recommendation: either OK and go on to station 'montage' OR rework at station 'lackieren' OR scrap and go to 'anmeldung'
    # updated in callbacks
    html.H5('Handlungsempfehlung', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div(
        className="flex-hd-row, flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm",
        id="recommendation_lackieren"),
    html.Br(),
    # Buttons for folding/unfolding options: go on to station 'montage', rework at station 'lackieren', scrap and go to 'anmeldung'
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
    # button for unfolding detailed information
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
                            ]),
                        dcc.Tab(label='Konfusionsmatrix', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                        ],
                                            style={'width': '100 %', "text-align": "center"}, # 'display': "flex"
                                            id='lackieren_confusion_absolute',
                                        ),
                                    ),
                                    dbc.Col(
                                        html.Div([
                                        ],
                                            style={'width': '100 %', "text-align": "center"}, # 'display': "flex"
                                            id='lackieren_confusion_normalised',
                                        ),
                                    ),
                                ], align="center", )
                            ], className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),
                        ]),
                        dcc.Tab(label='Wirtschaftliche Bewertung', style=tab_style, selected_style=tab_selected_style,
                                children=[
                                    html.Div([
                                        html.H6("Kostensenkung durch Eliminierung der Qualitätskontrolle: "),
                                        html.Ul(
                                            html.Li("Ø 5 Arbeitsstunden mit Personalkostensatz von 50 €/h = 250€"),
                                        ),
                                        html.H6(id='increase_in_costs_lackieren'),
                                        html.Ul(
                                            [
                                            ],
                                            id='cost_misclassification_lackieren',
                                        ),
                                        html.H6(id='savings_lackieren')
                                    ],
                                        className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),
                                ]),
                    ], style=tabs_styles),
                ], className="flex-column, flex-hd-row p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                id="collapse-details_lackieren",

            ),
        ],
    ),
    html.Hr(),
    html.Div(id='hidden-div-lackieren', style={'display': 'none'})
])

# reset slider status in temp .csv file on page reload
@app.callback([
          Output('hidden-div-lackieren','value')
      ],[
          Input('url','pathname'),
       ])
def reset_slider_status(pathname):
    if pathname == '/lackieren':
        file = open("temp/temp_lackieren_slider.csv", "w")
        file.write(str(2000) + "\n")
        file.close()
    return [None]

# button for collapsing options
@app.callback(
    Output("collapse-options_lackieren", "is_open"),
    [Input("collapse-button-options_lackieren", "n_clicks")],
    [State("collapse-options_lackieren", "is_open")],
)
def toggle_collapse_options(n, is_open):
    if n:
        return not is_open
    return is_open

# callback for collapsing detailed information (confusion matrix, economic evaluation, sectional view of classification results)
@app.callback(
    Output("collapse-details_lackieren", "is_open"),
    [Input("collapse-button-details_lackieren", "n_clicks")],
    [State("collapse-details_lackieren", "is_open")],
)
def toggle_collapse_options(n, is_open):
    if n:
        return not is_open
    return is_open

# update bar graphs, update classification plot, update accuracy metrics, update recommendation
@app.callback([
          # bar graphs
          Output('fig1_leistung_lackieren', 'figure'),
          Output('fig2_druck_lackieren', 'figure'),
          Output('fig3_temperatur_lackieren', 'figure'),
          # classification plot
          Output('fig4_classification_lackieren', 'figure'),
          # update category container (colored in red / green / orange)
          Output('category_lackieren', 'children'),
          Output('category_lackieren', 'style'),
          # update recommendation for user
          Output('recommendation_lackieren', 'children'),
          # update accuracy metrics (based on number on training points)
          Output('accuracy_lackieren', 'children'),
          Output('f1score_lackieren', 'children'),
          Output('precision_lackieren', 'children'),
          Output('sensitivity_lackieren', 'children'),
          # update confusion matrix (based on number of training points)
          Output('lackieren_confusion_absolute', 'children'),
          Output('lackieren_confusion_normalised', 'children'),
          # update classification results (sectional view), based on number of training points
          Output('lackieren_train_power', 'children'),
          Output('lackieren_train_pressure', 'children'),
          Output('lackieren_train_temp', 'children'),
      ],[
          # input url
          Input('url','pathname'),
          # input data slider status
          Input('dataset-slider_lackieren','value'),
       ])
def update_inputs(pathname, slider_status):

    # save number of training data from status of slider
    n_train = slider_status

    # load model from pickle for doing prediction on new data points lateron
    clf = pickle.load(open("assets/lackieren/lackieren_knn_model_" + str(n_train) + ".sav", 'rb'))

    # load last slider status from temp file
    if path.exists("temp/temp_lackieren_slider.csv"):
        f = open("temp/temp_lackieren_slider.csv", "r")
        old_slider_status = int(f.read())
        f.close()
    else:
        old_slider_status= None

    # write momentary slider status to file
    file = open("temp/temp_lackieren_slider.csv", "w")
    file.write(str(slider_status) + "\n")
    file.close()

    if pathname == '/lackieren':

        # load training, test data and accuracy metrics
        with open("assets/lackieren/lackieren_knn_data_"+str(n_train)+".csv") as mycsv:
            count = 0
            for line in mycsv:
                if count == 0:
                    prediction_test_load = line
                if count == 1:
                    testdata_load = line
                if count == 2:
                    prediction_train_load = line
                if count == 3:
                    traindata_load = line
                if count == 4:
                    classification_report = line
                if count == 5:
                    break
                count += 1

        # transform strings to numpy lists, while conserving np.array dimensions
        prediction_test_load = re.sub('\s+', '', prediction_test_load)
        prediction_test_scatter = ast.literal_eval(prediction_test_load)

        testdata_load = re.sub('\s+', '', testdata_load)
        testdata_load = np.asarray(ast.literal_eval(testdata_load))
        Leistung_test_scatter = np.round(testdata_load[:, 0],2).tolist()
        Druck_test_scatter = np.round(testdata_load[:, 1],2).tolist()
        Temperatur_test_scatter = np.round(testdata_load[:, 2],2).tolist()

        classification_report = re.sub('\s+', '', classification_report)
        classification_report = ast.literal_eval(classification_report)

        # get accuracy metrics from classification_report
        accuracy = np.round(classification_report['accuracy'], 2)
        f1_score = np.round(classification_report['macroavg']['f1-score'], 2)
        precision_gutteil = np.round(classification_report['0.0']['precision'], 2)
        sensitivity_gutteil = np.round(classification_report['0.0']['recall'], 2)
        precision_nachbearbeiten = np.round(classification_report['1.0']['precision'], 2)
        sensitivity_nachbearbeiten = np.round(classification_report['1.0']['recall'], 2)
        precision_ausschuss = np.round(classification_report['2.0']['precision'], 2)
        sensitivity_ausschuss = np.round(classification_report['2.0']['recall'], 2)
        
        # load last process parameters from file
        if path.exists("temp/temp_process_params_lackieren.csv"):

            f = open("temp/temp_process_params_lackieren.csv", "r")
            process_params_load1 = f.read()
            f.close()

            process_params_load2 = re.sub('\s+', '', process_params_load1)
            process_params = ast.literal_eval(process_params_load2)
            
        # only simulate a new data point on page refresh
        if slider_status == old_slider_status or path.exists("temp/temp_process_params_lackieren.csv") == False:

            # make sure that class is not the same as before on page refrehs
            index = global_index_lackieren[-1]
            while global_index_lackieren[-1] == index:
                
                # create randomized process parameter from multivariate gaussian
                mean = [1.5, 6, 32]
                cov = [[1.5, 0, 0], [0, 5, 0], [0, 0, 120]] # 1,3,100

                L, D, T = np.random.multivariate_normal(mean, cov, 1).T
                L = np.abs(L)
                D = np.abs(D)
                T = np.abs(T)

                # use saved k-nearest-neighbor classification algorithm for predicting class (ok/rework/scrap) for newly generated data point
                z = clf.predict(np.c_[L, D, T])
                index = int(z)

                process_params = [L.tolist(), D.tolist(), T.tolist()]

                # save process_params to temp file
                file = open("temp/temp_process_params_lackieren.csv", "w")
                file.write(str(process_params) + "\n")
                file.close()
                
            global_index_lackieren.append(index)

        # use saved k-nearest-neighbor classification algorithm for predicting class (ok/rework/scrap)
        # for newly generated data point or for last data point loaded from file
        z = clf.predict(
            np.c_[process_params[0][0], process_params[1][0], process_params[2][0]])

        # load confusion matrix depending on training data
        lackieren_confusion_absolute_callback = html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_confusion_absolute_' + str(n_train) + '.png'))
        ],)
        lackieren_confusion_normalised_callback = html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_confusion_normalised_' + str(n_train) + '.png'))
        ],)

        # load predictions along planes of constant parameters, based on training data
        lackieren_train_power= html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_train_power_' + str(n_train) + '.png'))
        ],)
        lackieren_train_pressure = html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_train_pressure_' + str(n_train) + '.png'))
        ],)
        lackieren_train_temp= html.Div([
            html.Img(src=app.get_asset_url('lackieren/lackieren_train_temp_' + str(n_train) + '.png'))
        ],)

        # plot bar graph of pressure
        fig1_callback = go.Figure()

        fig1_callback.add_trace(go.Indicator(
            mode="number+gauge", value=process_params[1][0], number={'font': {'size': 30}},
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
        
        # plot bar graph of temperature
        fig2_callback = go.Figure()
        fig2_callback.add_trace(go.Indicator(
            mode="number+gauge", value=process_params[2][0], number={'font': {'size': 30}},
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
        
        # plot bar graph of power
        fig3_callback = go.Figure()
        fig3_callback.add_trace(go.Indicator(
            mode="number+gauge", value=process_params[0][0], number={'font': {'size': 30}},
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

        # update info boxes with accuracy metrics
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

        # dataframe for scatter of test data
        df_test = pandas.DataFrame({'Leistung_test_scatter': Leistung_test_scatter, 'Druck_test_scatter': Druck_test_scatter, 'Temperatur_test_scatter': Temperatur_test_scatter,'prediction_test_scatter': prediction_test_scatter})

        # define marker colors
        marker_color = {
            2.0: 'lightcoral',
            1.0: 'orange',
            0.0: 'lightgreen'
        }

        marker_colors = [marker_color[k] for k in df_test['prediction_test_scatter'].values]

        fig4_callback = go.Figure()

        # scatter plot of test data
        fig4_callback.add_trace(go.Scatter3d(
            x=df_test['Leistung_test_scatter'],
            y=df_test['Druck_test_scatter'],
            z=df_test['Temperatur_test_scatter'],
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
            x=np.asarray(process_params[0][0]),
            y=np.asarray(process_params[1][0]),
            z=np.asarray(process_params[2][0]),
            mode='markers',
            name='Prozessparameter',
            marker_color='magenta',
            marker=dict(
                symbol='cross',
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

        # update recommendation for user: go on to station 'montage', rework at station 'lackieren', scrap and go to 'anmeldung'
        if z[0] == 0:
            cat_string = "Gutteil"
            cat_color = "green"
            recommendation_alert = dbc.Alert(
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
            recommendation_alert = dbc.Alert(
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
            recommendation_alert = dbc.Alert(
                                [
                                    "Handlungsempfehlung: Klassifiziere Bauteil als ",
                                    html.A("Ausschuss", href="/", className="alert-link"),
                                ],
                                color="danger",
                                style={'font-size': '130%'},
                                ),

        # update colored box based on predicted category
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

        return [fig1_callback, fig2_callback, fig3_callback, fig4_callback, category, style, recommendation_alert, accuracy_callback,
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

# callback for updating economic evaluation
@app.callback([
    Output('cost_misclassification_lackieren', 'children'),
    Output('savings_lackieren', 'children'),
    Output('savings_lackieren', 'style'),
    Output('increase_in_costs_lackieren', 'children'),
], [
    Input('url', 'pathname'),
    Input('dataset-slider_lackieren', 'value'),
    Input('category_lackieren', 'children')
])
def update_economic_evaluation(pathname, slider_status, category):
    if pathname == '/lackieren':

        n_train = int(slider_status)

        # load accuracy metrics
        with open("assets/lackieren/lackieren_knn_data_" + str(n_train) + ".csv") as mycsv:
            count = 0
            for line in mycsv:
                if count == 4:
                    classification_report = line
                if count == 5:
                    break
                count += 1

        # transform strings to numpy lists, while conserving np.array dimensions
        classification_report = re.sub('\s+', '', classification_report)
        classification_report = ast.literal_eval(classification_report)

        # get accuracy metrics from classification_report
        precision_gutteil = np.round(classification_report['0.0']['precision'], 2)
        precision_nachbearbeiten = np.round(classification_report['1.0']['precision'], 2)
        precision_ausschuss = np.round(classification_report['2.0']['precision'], 2)

        # this gives prediction for newly simulated data point: gutteil, nachbearbeiten, aussschuss
        cat = category['props']['children'][3]['props']['children']

        # select correct precision and cost for misclassification depending on class
        if cat == 'Gutteil':
            precision = precision_gutteil
            # reasoning: if a part is wrongly classified as OK, there will be unnecessary cost of  processing the part
            # at station assembly (500). we assume additional cost due to customer complaint that
            # part is not OK (3000). cost needs to be multiplied by precision (probability of misclassification)
            cost = 3500
            list_cost_misclass = [
                html.Li(
                    'Kosten für weitere Bearbeitung an Stationen Lackieren und Montage: 500€ * (1 - ' + 'Präzision der Klasse (' + str(
                        precision) + '))'),
                html.Li('Kosten durch Kundenreklamation: 3000 € * (1 - ' + 'Präzision der Klasse (' + str(precision) + '))')
            ]
            increase_in_costs_lackieren = 'Kostenerhöhung durch Fehlklassifizierung als Klasse Gutteil:'

        elif cat == 'Nachbearbeiten':
            precision = precision_nachbearbeiten
            # reasoning: if a part is wrongly classified as rework, there will be unnecessary cost of  processing the part
            # at station lackieren again (500).
            # cost needs to be multiplied by precision (probability of misclassification)
            cost = 500
            list_cost_misclass = html.Li(
                'Kosten für Nachbearbeitung an Station Lackieren: 500 € * ( 1 - ' + 'Präzision der Klasse (' + str(
                    precision) + '))')
            increase_in_costs_lackieren = 'Kostenerhöhung durch Fehlklassifizierung als Klasse Nachbearbeiten:'

        elif cat == 'Ausschuss':
            precision = precision_ausschuss
            # reasoning: if a part is wrongly classified as scrap, we have unnecessary cost of wasting raw material (2000)
            # cost needs to be multiplied by precision (probability of misclassification)
            cost = 2000
            list_cost_misclass = html.Li(
                'Kosten für Verlust an Rohmaterial: 2000 € * (1 - ' + 'Präzision der Klasse (' + str(
                    precision) + '))')
            increase_in_costs_lackieren = 'Kostenerhöhung durch Fehlklassifizierung als Klasse Ausschuss:'

        cost_per_part = (1 - precision) * cost
        savings_temp = int(250 - cost_per_part)

        if savings_temp > 0:
            savings_lackieren = 'Kosteneinsparung aktuelles Bauteil: ' + str(savings_temp) + ' €'
            style_savings_lackieren = {"color": 'green'}
        else:
            savings_lackieren = 'Kostenerhöhung aktuelles Bauteil: ' + str(savings_temp) + ' €'
            style_savings_lackieren = {"color": 'darkred'}

        cost_misclassification_lackieren = list_cost_misclass

    return [cost_misclassification_lackieren, savings_lackieren, style_savings_lackieren, increase_in_costs_lackieren]