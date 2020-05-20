import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import startseite, anmeldung, spanen, lackieren, montage


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
         return startseite.layout
    elif pathname == '/anmeldung':
         return anmeldung.layout
    elif pathname == '/spanen':
         return spanen.layout
    elif pathname == '/lackieren':
         return lackieren.layout
    elif pathname == '/montage':
         return montage.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)