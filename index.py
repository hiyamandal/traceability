import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app, server
from apps import anmeldung, spanen, lackieren, montage, commonmodules

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "26rem", # 18rem bei 5% body margin
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url", refresh=False), commonmodules.get_sidebar(), content])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if   pathname == '/':
         return anmeldung.layout
    elif pathname == '/spanen':
         return spanen.layout
    elif pathname == '/lackieren':
         return lackieren.layout
    elif pathname == '/montage':
         return montage.layout
    else:
        return dbc.Jumbotron(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ])


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
pathnames = ["empty", "/", "/spanen", "/lackieren", "/montage"]
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 5)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        return [True, False, False, False]

    return [pathname == pathnames[i] for i in range(1, 5)]

if __name__ == '__main__':
    app.run_server(debug=True)