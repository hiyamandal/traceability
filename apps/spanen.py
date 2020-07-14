import dash as dash

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

global_index_spanen = [0]
global_slider_spanen = ['1000']
global_P_list_spanen = []
global_F_list_spanen = []

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
    html.H4('Station Spanen', style={'font-weight': 'bold', 'text-decoration': 'underline'}),
    html.Hr(),
    html.H4('Klassifizierungsergebnisse', style={'font-weight': 'bold'}),
    html.Br(),
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        html.Div([
                            dcc.Graph(id='fig1_callback_spanen'),
                        ], style={'textAlign': 'center', 'width': '100%'}),
                    ]),

                    dbc.Row([
                        html.Div(
                            [
                                dcc.Graph(id='fig2_callback_spanen'),
                             ], style={'textAlign': 'center', 'width': '100%'}),

                    ]),
                ],
                className = "pretty_container",),
                dbc.Row(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        id="category",
                                        className="pretty_container",
                                    ),
                                    html.Div(
                                        id="accuracy",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        id="f1score",
                                        className="pretty_container"
                                    ),
                                    html.Div(
                                        id="precision",
                                        className="pretty_container"
                                    ),
                                    html.Div(
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
            width=6),
            dbc.Col([
                html.Div([
                    html.H5("Gelernte Entscheidungsgrenzen", style={"text-align": "center",'font-weight': 'bold'}),
                    html.Br(),
                    html.Div([
                        dcc.Graph(id='fig3_callback_spanen'),
                        html.H6("Anzahl an Datenpunkten (Training + Test):", style={"text-align": "center",'font-weight': 'bold'}),
                        dcc.Slider(
                            id='dataset-slider_spanen',
                            min=1000,
                            max=3000,
                            value=1000,
                            marks={
                                1000: '# data: 1000 ',
                                2000: '2000',
                                3000: '3000',
                            },
                            step=1000,
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
        id="handlungsempfehlung_spanen"),
    html.Br(),
    html.Div(
    [
        dbc.Button(
            "Weitere Handlungsoptionen",
            id="collapse-button-options_spanen",
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
                                html.A("Lackieren", href="/lackieren", className="alert-link"),
                            ],
                            color="success",
                            style={'font-size': '130%'},
                        ),
                        dbc.Alert(
                            [
                                "Nachbearbeitung an Station  ",
                                html.A("Spanen", href="/spanen", className="alert-link"),
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
            ], className="flex-hd-row, flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  #  d-flex
            id="collapse-options_spanen",
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
                id="collapse-button-details",
                className="mb-3",
                color="primary",
                style={'font-size': '100%'},
            ),
            dbc.Collapse(
                html.Div([
                    dcc.Tabs(id='tabs-spanen', value='tab-1', children=[
                        dcc.Tab(label='Konfusionsmatrix', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                        ],
                                            style = {'width': '100 %', "text-align": "center", 'display': "flex"}, id='spanen_confusion_absolute',
                                        ),
                                    ),
                                    dbc.Col(
                                        html.Div([
                                        ],
                                            style={'width': '100 %', "text-align": "center", 'display': "flex"}, id ='spanen_confusion_normalised',
                                        ),
                                    ),
                                ], align="center",)
                            ], className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                        ]),
                        dcc.Tab(label='Wirtschaftliche Bewertung', style=tab_style, selected_style=tab_selected_style, children=[
                            html.Div([
                                html.H6("Wirtschaftliche Bewertung einfügen.")
                            ], className="flex-hd-row flex-column p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                        ]),
                    ], style=tabs_styles),
                ], className="flex-column, flex-hd-row p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm"),  # d-flex
                id="collapse-details",
            ),
        ],
    ),
    html.Hr(),
])

# callbacks
@app.callback(
    Output("collapse-options_spanen", "is_open"),
    [Input("collapse-button-options_spanen", "n_clicks")],
    [State("collapse-options_spanen", "is_open")],
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

@app.callback([
          Output('fig1_callback_spanen', 'figure'),
          Output('fig2_callback_spanen', 'figure'),
          Output('fig3_callback_spanen', 'figure'),
          Output('category', 'children'),
          Output('category', 'style'),
          Output('handlungsempfehlung_spanen', 'children'),
          Output('accuracy', 'children'),
          Output('f1score', 'children'),
          Output('precision', 'children'),
          Output('sensitivity', 'children'),
          Output('spanen_confusion_absolute', 'children'),
          Output('spanen_confusion_normalised', 'children'),
      ],[
          Input('url','pathname'),
          Input('dataset-slider_spanen','value')
       ])
def update_inputs(pathname, value):

    n_train = value
    ctx = dash.callback_context
    print(ctx.triggered)
    print(ctx.inputs)
    print(ctx.states)
    if pathname == '/spanen':

        # with open("assets/spanen/spanen_knn_data_" + str(n_train) + ".csv") as mycsv:
        #     for line in mycsv:
        #         old_slider_status = line
        # new_slider_status = value
        # if new_slider_status != old_slider_status:
        #     reload_datapoint = False
        #
        # #   append global url list
        # file = open("temp_spanen_slider.csv", "w+")
        # file.write(str(value) + "\n")
        # file.close()

        old_slider_status = global_slider_spanen[-1]
        global_slider_spanen.append(str(value))
        new_slider_status = global_slider_spanen[-1]
        reload_datapoint = True
        if new_slider_status != old_slider_status:
            reload_datapoint = False

        if reload_datapoint == True:
            # append global index list
            index = global_index_spanen[-1]
            while global_index_spanen[-1] == index:
                # erzeuge randomisierte prozessgrößen
                draw = multinomial.rvs(1, [0.4, 0.3, 0.3])  # 0.5, 0.3, 0.2
                index = int(np.where(draw == 1)[0])
            global_index_spanen.append(index)

            # gutteil, leisung P und kraft F sind OK
            if index == 0:

                mean = [2.5, 1.5]
                cov = [[1.4, 1], [1.2, 0.5]]

                bool = True
                while bool:
                    F, P = np.random.multivariate_normal(mean, cov, 1).T
                    if F > 0.5 and F < 5 and P > 0 and P < 3:
                        bool = False

            # nacharbeit
            elif index == 1:
                draw2 = multinomial.rvs(1, [0.5, 0.5])
                index2 = np.where(draw2 == 1)

                # Leistung zu hoch
                if index2[0] == 0:
                    P = expon.rvs(3.5, 0.3, size=1)
                    F = uniform.rvs(3.5, 1.5, size=1)

                # Kraft zu niedrig oder zu hoch
                elif index2[0] == 1:

                    draw3 = multinomial.rvs(1, [0.5, 0.5])
                    index3 = np.where(draw3 == 1)

                    # Kraft zu niedrig
                    if index3[0] == 0:
                        P = uniform.rvs(0.5, 1, size=1)
                        F = uniform.rvs(0, 0.25, size=1)

                    # Kraft zu hoch
                    elif index3[0] == 1:
                        P = uniform.rvs(2, 0.5, size=1)
                        F = expon.rvs(5.5, 0.2, size=1)

            # ausschuss: leistung und kraft zu hoch
            elif index == 2:
                P = expon.rvs(3.5, 0.3, size=1)  # loc, scale, size
                F = expon.rvs(5.5, 0.2, size=1)

            global_P_list_spanen.append(P)
            global_F_list_spanen.append(F)

        # load confusion matrix
        spanen_confusion_absolute_callback = html.Div([
            html.Img(src=app.get_asset_url('spanen/spanen_confusion_absolute_' + str(n_train) + '.png'))
        ],)
        spanen_confusion_normalised_callback = html.Div([
            html.Img(src=app.get_asset_url('spanen/spanen_confusion_normalised_' + str(n_train) + '.png'))
        ],)

        # plot für prozessgrößen
        fig1_callback = go.Figure()

        fig1_callback.add_trace(go.Indicator(
            mode="number+gauge", value=global_F_list_spanen[-1][0], number={'font': {'size': 30}},
            # delta = {'reference': 200},
            domain={'x': [0.25, 1], 'y': [0.3, 0.7]},
            title={'text': "Kraft in kN", 'font': {'size': 20}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 8]},
                'threshold': {
                    'line': {'color': 'black', 'width': 5},
                    'thickness': 0.75,
                    'value': 5},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 5], 'color': "green"},
                    {'range': [5, 8], 'color': "lightgray"}],
                'bar': {
                    'color': 'black'}
                        },
        ),
        )

        fig1_callback.update_layout(autosize=True, height=150, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                           paper_bgcolor="#f9f9f9", )

        fig2_callback = go.Figure()
        fig2_callback.add_trace(go.Indicator(
            mode="number+gauge", value=global_P_list_spanen[-1][0], number={'font': {'size': 30}},
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
        fig2_callback.update_layout(autosize=True, height=150, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                                    paper_bgcolor="#f9f9f9", )

        with open("assets/spanen/spanen_knn_data_"+str(n_train)+".csv") as mycsv:
            count = 0
            for line in mycsv:
                if count == 0:
                    z_test_load = line
                if count == 1:
                    data_test_load = line
                if count == 2:
                    xx_load = line
                if count == 3:
                    yy_load = line
                if count == 4:
                    Z_load = line
                if count == 5:
                    z_train_load = line
                if count == 6:
                    data_train_load = line
                if count == 7:
                    report = line
                if count == 8:
                    break
                count += 1

        # transform strings to numpy lists, while conserving np.array dimensions
        import re, ast

        z_test_load = re.sub('\s+', '', z_test_load)
        z_test_scatter = ast.literal_eval(z_test_load)

        z_train_load = re.sub('\s+', '', z_train_load)
        z_train_scatter = ast.literal_eval(z_train_load)

        data_test_load = re.sub('\s+', '', data_test_load)
        data_test_load = np.asarray(ast.literal_eval(data_test_load))
        x_test_scatter = np.round(data_test_load[:, 0],2).tolist()
        y_test_scatter = np.round(data_test_load[:, 1],2).tolist()

        data_train_load = re.sub('\s+', '', data_train_load)
        data_train_load = np.asarray(ast.literal_eval(data_train_load))
        x_train_scatter = np.round(data_train_load[:, 0],2).tolist()
        y_train_scatter = np.round(data_train_load[:, 1],2).tolist()

        xx_load = re.sub('\s+', '', xx_load)
        x_cont = ast.literal_eval(xx_load)

        yy_load = re.sub('\s+', '', yy_load)
        y_cont = ast.literal_eval(yy_load)

        Z_load = re.sub('\s+', '', Z_load)
        z_cont = np.asarray(ast.literal_eval(Z_load))

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
        df_test = pandas.DataFrame({'x_test_scatter': x_test_scatter, 'y_test_scatter': y_test_scatter, 'z_test_scatter': z_test_scatter})
        df_train = pandas.DataFrame({'x_train_scatter': x_train_scatter, 'y_train_scatter': y_train_scatter, 'z_train_scatter': z_train_scatter})

        # contour plot
        colorscale = [[0, 'green'], [0.5, 'darkorange'], [1, 'darkred']]
        fig3_callback = go.Figure(data=
        go.Contour(
            name='Klassifizierung',
            z=z_cont,
            x=x_cont,
            y=y_cont,
            colorscale=colorscale,
            showscale=False,
        ))

        # define marker colors
        marker_color = {
            2.0: 'lightcoral',
            1.0: 'orange',
            0.0: 'lightgreen'
        }

        marker_colors = [marker_color[k] for k in df_train['z_train_scatter'].values]

        # scatter plot of training data
        fig3_callback.add_trace(go.Scatter(
            x=df_train['x_train_scatter'],
            y=df_train['y_train_scatter'],
            mode='markers',
            name='Trainingsdaten',
            marker_color=marker_colors,
            marker=dict(
                opacity = 0.4,
                symbol = 'x',
                # color='rgb(255, 178, 102)',
                size=10,
                line=dict(
                    color='DarkSlateGrey',
                    width=1
                )
            )
        ))

        marker_colors = [marker_color[k] for k in df_test['z_test_scatter'].values]

        # scatter plot of training data
        fig3_callback.add_trace(go.Scatter(
            x=df_test['x_test_scatter'],
            y=df_test['y_test_scatter'],
            mode='markers',
            name='Testdaten',
            marker_color=marker_colors,
            marker=dict(
                size=10,
                line=dict(
                    color='DarkSlateGrey',
                    width=2
                )
            )
        ))

        # scatter plot of single new test point
        fig3_callback.add_trace(go.Scatter(
            x=np.asarray(global_F_list_spanen[-1][0]),
            y=np.asarray(global_P_list_spanen[-1][0]),
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

        fig3_callback.update_layout(
            margin = {'l': 15, 'r': 15, 't': 15, 'b': 15},
            xaxis_title='Kraft in kN',
            yaxis_title='Leistung in kW',
            font=dict(
                size=18,
            ),
            xaxis=dict(
                range=[0,8]
            ),
            yaxis = dict(
            range=[0, 6]
        )
        )

        # load model from pickle
        import pickle

        clf = pickle.load(open("assets/spanen/spanen_knn_model_"+str(n_train)+".sav", 'rb'))
        z = clf.predict(np.c_[global_F_list_spanen[-1][0], global_P_list_spanen[-1][0]])

        if z[0] == 0:
            cat_string = "Gutteil"
            cat_color = "green"
            empfehlung_alert = dbc.Alert(
                                [
                                    "Handlungsempfehlung: Weiter zur Station ",
                                    html.A("Lackieren", href="/lackieren", className="alert-link"),
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
                                    html.A("Spanen", href="spanen", className="alert-link"),
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

        return [fig1_callback, fig2_callback, fig3_callback, category, style, empfehlung_alert, accuracy_callback,
                f1_score_callback, precision_callback, sensitivity_callback,
                spanen_confusion_absolute_callback, spanen_confusion_normalised_callback]

    else:
        fig1_callback = go.Figure()
        fig2_callback = go.Figure()
        fig3_callback = go.Figure()
        z = 0
        z = np.asarray(z)
        category= None
        style = None
        empfehlung_alert = None
        accuracy_callback = None
        f1_score_callback = None
        precision_callback = None
        sensitivity_callback = None
        spanen_confusion_absolute_callback = None
        spanen_confusion_normalised_callback = None

        return [fig1_callback, fig2_callback, fig3_callback, category, style, empfehlung_alert, accuracy_callback,
                f1_score_callback, precision_callback, sensitivity_callback,
                spanen_confusion_absolute_callback, spanen_confusion_normalised_callback]
