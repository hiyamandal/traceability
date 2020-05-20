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
        ], className="twelve columns padded"),

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

    # nav = dbc.Nav(
    #     [
    #         dbc.NavItem(dbc.NavLink('Home  ', href='/')),
    #         dbc.NavItem(dbc.NavLink('Station Anmeldung  ', href='/anmeldung')),
    #         dbc.NavItem(dbc.NavLink('Station Spanen   ', href='/spanen')),
    #         dbc.NavItem(dbc.NavLink('Station Lackieren   ', href='/lackieren',)),
    #         dbc.NavItem(dbc.NavLink('Station Montage   ', href='/montage', )),
    #     ],
    #     pills=True,
    # )
    return menu

def get_sidebar():
    # we use the Row and Col components to construct the sidebar header
    # it consists of a title, and a toggle, the latter is hidden on large screens
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
            html.H2("Sidebar", className="display-4"),
            html.Hr(),
            html.P(
                "A simple sidebar layout with navigation links", className="lead"
            ),
            dbc.Nav(
                [
                    dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
                    dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
                    dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    return sidebar