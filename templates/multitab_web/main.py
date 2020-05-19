#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from templates.multitab_web.tabs import tab_2, tab_1

app = dash.Dash()

app.config.suppress_callback_exceptions = True

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Dash Tabs component demo'),
    #     dcc.Tabs(
    #         tabs=[
    #             {'label': 'Tab One', 'value': 'tab-1-example'},
    #             {'label': 'Tab Two', 'value': 'tab-2-example'}
    #         ],
    #         value='tab-1-example',
    #         id='tabs-example'
    #     ),
    # html.Div(id='tabs-content-example')
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Tab one', value='tab-1'),
        dcc.Tab(label='Tab two', value='tab-2'),
    ]),
    html.Div(id='tabs-content-example')
])

@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab_1.tab_1_layout
    elif tab == 'tab-2':
        return tab_2.tab_2_layout

# Tab 1 callback
@app.callback(dash.dependencies.Output('page-1-content', 'children'),
              [dash.dependencies.Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return 'You have selected "{}"'.format(value)

# Tab 2 callback
@app.callback(Output('page-2-content', 'children'),
              [Input('page-2-radios', 'value')])
def page_2_radios(value):
    return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)