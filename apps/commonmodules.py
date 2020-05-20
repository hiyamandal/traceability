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

tab_style = {
    'color': '#0074D9',
    'text-decoration': 'underline',
    'margin': 30,
    'cursor': 'pointer',
     'display':'inline-block',
    'font-weight': 'bold',
    'font_size': '156px',
}

def get_menu():
    menu = html.Div([

        dcc.Link('Startseite   ', href='/', className="p-2 text-dark", style=tab_style),
        dcc.Link('Station Anmeldung   ', href='/anmeldung', className="p-2 text-dark", style=tab_style),
        dcc.Link('Station Spanen   ', href='/spanen', className="p-2 text-dark", style=tab_style),
        dcc.Link('Station Lackieren    ', href='/lackieren', className="p-2 text-dark", style=tab_style),
        dcc.Link('Station Montage   ', href='/montage', className="p-2 text-dark", style=tab_style),

    ], className="d-flex flex-column flex-md-row align-items-center p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm")

    return menu

def get_sidebar():
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }
    sidebar = html.Div(
        [
            html.Div(html.Img(src=app.get_asset_url('ptw_web4.png'), style={'height': '7vh'})),
            html.H4("Navigation", className="display-5"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink("Startseite", href='/', id="page-1-link"),
                    dbc.NavLink('Station Anmeldung   ', href='/anmeldung', id="page-2-link"),
                    dbc.NavLink('Station Spanen   ', href='/spanen', id="page-3-link"),
                    dbc.NavLink('Station Lackieren    ', href='/lackieren', id="page-4-link"),
                    dbc.NavLink('Station Montage   ', href='/montage', id="page-5-link"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    return sidebar


