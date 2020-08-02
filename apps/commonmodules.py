import dash_html_components as html
import dash_bootstrap_components as dbc
from app import app

# definition of page header
def get_header():
    header = html.Div([
        html.Div([
            html.H2(
                'Machine Learning Applikation zur automatisierten Qualitätssicherung')
        ], className="twelve columns"),
    ], className="row gs-header gs-text-header")
    return header

# definition of page sidebar
def get_sidebar():
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "3rem 2rem",
        "background-color": "#f8f9fa",
    }
    sidebar = html.Div(
        [
            html.Div(html.Img(src=app.get_asset_url('ptw_web4.png'), style={'height': '7vh'})),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink('Station Anmeldung   ', href='/', id="page-1-link", style={'font-size': 15}),
                    dbc.NavLink('Station Spanen   ', href='/spanen', id="page-2-link", style={'font-size': 15}),
                    dbc.NavLink('Station Lackieren    ', href='/lackieren', id="page-3-link", style={'font-size': 15}),
                    dbc.NavLink('Station Montage   ', href='/montage', id="page-4-link", style={'font-size': 15}),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )
    return sidebar


