import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app

def get_header():
    header = html.Div([
        html.Div([
            html.H1(
                'Machine Learning Applikation zur automatisierten Qualitätssicherung')
        ], className="twelve columns"),
    ], className="row gs-header gs-text-header")
    return header

def get_sidebar():
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "24rem",
        "padding": "3rem 2rem",
        "background-color": "#f8f9fa",
        'font': {'size': 30}
    }
    sidebar = html.Div(
        [
            html.Div(html.Img(src=app.get_asset_url('ptw_web4.png'), style={'height': '5vh'})),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink('Station Anmeldung   ', href='/', id="page-1-link", style={'font-size': 20}),
                    dbc.NavLink('Station Spanen   ', href='/spanen', id="page-2-link", style={'font-size': 20}),
                    dbc.NavLink('Station Lackieren    ', href='/lackieren', id="page-3-link", style={'font-size': 20}),
                    dbc.NavLink('Station Montage   ', href='/montage', id="page-4-link", style={'font-size': 20}),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    return sidebar


