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
    sidebar_header = dbc.Row(
        [
            dbc.Col(html.H2("Sidebar", className="display-4")),
            dbc.Col(
                [
                    html.Button(
                        # use the Bootstrap navbar-toggler classes to style
                        html.Span(className="navbar-toggler-icon"),
                        className="navbar-toggler",
                        # the navbar-toggler classes don't set color
                        style={
                            "color": "rgba(0,0,0,.5)",
                            "border-color": "rgba(0,0,0,.1)",
                        },
                        id="navbar-toggle",
                    ),
                    html.Button(
                        # use the Bootstrap navbar-toggler classes to style
                        html.Span(className="navbar-toggler-icon"),
                        className="navbar-toggler",
                        # the navbar-toggler classes don't set color
                        style={
                            "color": "rgba(0,0,0,.5)",
                            "border-color": "rgba(0,0,0,.1)",
                        },
                        id="sidebar-toggle",
                    ),
                ],
                # the column containing the toggle will be only as wide as the
                # toggle, resulting in the toggle being right aligned
                width="auto",
                # vertically align the toggle in the center
                align="center",
            ),
        ]
    )

    sidebar = html.Div(
        [
            sidebar_header,
            # we wrap the horizontal rule and short blurb in a div that can be
            # hidden on a small screen
            html.Div(
                [
                    html.Hr(),
                    html.P(
                        "A responsive sidebar layout with collapsible navigation "
                        "links.",
                        className="lead",
                    ),
                ],
                id="blurb",
            ),
            # use the Collapse component to animate hiding / revealing links
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
                        dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
                        dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
                    ],
                    vertical=True,
                    pills=True,
                ),
                id="collapse",
            ),
        ],
        id="sidebar",
    )

    return sidebar