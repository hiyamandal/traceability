import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

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