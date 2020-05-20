import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import flask
import os


print(dcc.__version__) # 0.6.0 or above is required

# external_css = ["hhttps://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css",
#                 "//fonts.googleapis.com/css?family=Raleway:400,300,600",
#                 "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]
# external_css = ["https://github.com/plotly/dash-analytics-report/blob/master/dash-analytics-report.css"]
# external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_css = ['https://raw.githubusercontent.com/plotly/dash-sample-apps/master/apps/dash-mapd-demo/assets/dash-mapd.css']

# app = dash.Dash(__name__, external_stylesheets=external_css)
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
# app = dash.Dash(__name__)
# app = dash.Dash()
#
# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
# "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
# "//fonts.googleapis.com/css?family=Raleway:400,300,600",
# "https://codepen.io/bcd/pen/KQrXdb.css",
# "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]
#
# for css in external_css:
#    app.css.append_css({"external_url": css})

# css_directory = os.getcwd()
# stylesheets = ['stylesheet.css']
# static_css_route = '/static/'


# @app.server.route('{}<stylesheet>'.format(static_css_route))
# def serve_stylesheet(stylesheet):
#     if stylesheet not in stylesheets:
#         raise Exception(
#             '"{}" is excluded from the allowed static files'.format(
#                 stylesheet
#             )
#         )
#     return flask.send_from_directory(css_directory, stylesheet)
#
#
# for stylesheet in stylesheets:
#     app.css.append_css({"external_url": "/static/{}".format(stylesheet)})


server = app.server
app.config.suppress_callback_exceptions = True
