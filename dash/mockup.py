import pandas as pd
import numpy as np

from clark_dash.utils import build_options

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State



def build_mockup(df):
    app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP],
                    meta_tags=[{'content': 'width=device-width'}])
    app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1('prova 1')),
                dbc.Col(html.H1('prova 2')),
            ]),

            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(id = 'dropdown', options=build_options(['a', 'b', 'c'])),
                    width=3
                ),

                dbc.Col(width=3),

                dbc.Col(
                    dcc.Dropdown(id='dropdown2', options=build_options(['a', 'b', 'c'])),
                    width=6
                )
            ])
    ], fluid=True)
    return app