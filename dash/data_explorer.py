import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.datasets import load_iris
from abc import abstractmethod
import random

from clark_py.dash.utils import build_options, return_width_height

import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot

import dash
import dash_daq as daq
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


class DashDataExplorer(object):


    def __init__(self):

        # List of module type can be generated
        self._module_type = ['scatter', 'scatter_matrix', 'timeseries', '2timeseries']


    def run(self, df: pd.DataFrame, debug = False):

        self.df = df
        self._debug = debug


        app = self._build_app()
        app.run_server(debug=debug)

        return



    def _build_app(self) -> dash.Dash:
        '''
        Build dash app.
        Initialize app and write layout structure
        '''

        # Create a copy inside the method to be used in callback functions
        data = self.df.copy()

        app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
                        meta_tags=[{'content': 'width=device-width'}],
                        suppress_callback_exceptions=True)
        app.layout = dbc.Container([

            # Title
            dbc.Row([
                dbc.Col(html.H1('Data Explorer', style={'textAlign': 'center', 'backgroundColor': 'lightblue', 'color': 'white'})),
            ]),
            html.Div(style={'padding': 10}),

            # -------------- Module 0 --------------
            dbc.Row([
                dbc.Col(html.H4('Module 0', style={'backgroundColor': 'lightblue', 'color': 'white'})),
            ]),

            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(id = 'module_0_type',
                                 options=build_options(self._module_type),
                                 multi=False),
                    width=2,
                ),

                dbc.Col(
                    dcc.Dropdown(id='width_0',
                                 placeholder='width',
                                 options=build_options(np.arange(10, 110, step = 10)),
                                 multi=False),
                    width=1),

                dbc.Col(
                    dcc.Dropdown(id='height_0',
                                 placeholder='height',
                                 options=build_options(np.arange(10, 110, step=10)),
                                 multi=False),
                    width=1),

            ]),

            html.Div(style={'padding': 10}),
            html.Div(id = 'module_0'),


            # -------------- Module 1 --------------
            html.Div(style={'padding': 50}),
            dbc.Row([
                dbc.Col(html.H4('Module 1', style={'backgroundColor': 'lightblue', 'color': 'white'})),
            ]),

            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(id='module_1_type',
                                 options=build_options(self._module_type),
                                 multi=False),
                    width=2,
                ),

                dbc.Col(
                    dcc.Dropdown(id='width_1',
                                 placeholder='width',
                                 options=build_options(np.arange(10, 110, step=10)),
                                 multi=False),
                    width=1),

                dbc.Col(
                    dcc.Dropdown(id='height_1',
                                 placeholder='height',
                                 options=build_options(np.arange(10, 110, step=10)),
                                 multi=False),
                    width=1),


            ]),

            html.Div(style={'padding': 10}),
            html.Div(id='module_1'),


            # bottom space
            html.Div(style={'padding': 50}),
        ], fluid=True) # end layout






        # ------------------------------- call backs module 0 -------------------------------
        @app.callback(Output('plot_scatter_0', 'children'),
                      [Input('xaxis_scatter_0', 'value'), Input('yaxis_scatter_0', 'value'), Input('color_scatter_0', 'value'), Input('size_scatter_0', 'value'),
                       Input('reg_switch_0', 'value'),
                       Input('width_0', 'value'), Input('height_0', 'value')])
        def render_scatter_plot(xaxis, yaxis, color, size, reg_switch, width_input, height_input):

            if (xaxis is not None) and (yaxis is not None):

                # deduce size max
                size_max = None
                if size is not None:
                    size_max = data[size].max()

                # lin reg param
                trendline = None
                if reg_switch:
                    trendline = 'ols'

                # set height and width
                width, height = return_width_height(width_input, height_input, 70, 70)

                fig = px.scatter(data_frame = data, x=xaxis, y=yaxis, color=color, size=size, size_max=size_max, trendline=trendline)
                return dcc.Graph(figure=fig, style={'width': f'{width}vh', 'height': f'{height}vh'})
            else:
                return html.Div()


        @app.callback(Output('plot_scatter_matrix_0', 'children'),
                      [Input('cols_scatter_matrix_0', 'value'),  Input('color_scatter_matrix_0', 'value'),
                       Input('width_0', 'value'), Input('height_0', 'value')])
        def render_scatter_matrix_plot(cols, color, width_input, height_input):

            if (cols is not None):

                # set height and width
                width, height = return_width_height(width_input, height_input, 100, 100)

                fig = px.scatter_matrix(data_frame = data, dimensions=cols, color=color)
                fig.update_traces(diagonal_visible=False)
                return dcc.Graph(figure=fig, style={'width': f'{width}vh', 'height': f'{height}vh'})
            else:
                return html.Div()


        @app.callback(Output('plot_ts_0', 'children'), [Input('xaxis_ts_0', 'value'), Input('yaxis_ts_0', 'value')])
        def render_ts_plot(xaxis, yaxis):

            if (xaxis is not None) and (yaxis is not None):

                # sort data frame on x axis
                data.sort_values(by=xaxis, inplace=True)
                fig = px.line(data_frame=data, x = xaxis, y = yaxis)
                fig.update_xaxes(rangeslider_visible=True)
                return dcc.Graph(figure=fig, style={'height': '50vh'})

            else:
                return html.Div()


        @app.callback(Output('plot_2ts_0', 'children'), [Input('xaxis_2ts_0', 'value'), Input('yaxis1_2ts_0', 'value'), Input('yaxis2_2ts_0', 'value')])
        def render_ts_plot(xaxis, yaxis1, yaxis2):

            if (xaxis is not None) and (yaxis1 is not None) and (yaxis2):

                # sort data frame on x axis
                data.sort_values(by=xaxis, inplace=True)

                fig1 = px.line(data_frame=data, x=xaxis, y=yaxis1)
                fig2 = px.line(data_frame=data, x=xaxis, y=yaxis2)

                # create ssubplots grid and add traces
                fig = make_subplots(2, 1, shared_xaxes=True)
                for trace in fig1.select_traces():
                    fig.add_trace(trace, row=1, col=1)

                for trace in fig2.select_traces():
                    fig.add_trace(trace, row=2, col=1)

                return dcc.Graph(figure=fig, style={'height': '50vh'})

            else:
                return html.Div()




        @app.callback(Output('module_0', 'children'), [Input('module_0_type', 'value')])
        def render_module(type: str):
            if type is not None:
                return self._module_hub(type, 0)
            else:
                return html.Div()


        # ------------------------------- call backs module 1 -------------------------------
        @app.callback(Output('plot_scatter_1', 'children'),
                      [Input('xaxis_scatter_1', 'value'), Input('yaxis_scatter_1', 'value'), Input('color_scatter_1', 'value'), Input('size_scatter_1', 'value'),
                       Input('reg_switch_1', 'value'),
                       Input('width_1', 'value'), Input('height_1', 'value')])
        def render_scatter_plot(xaxis, yaxis, color, size, reg_switch, width_input, height_input):

            if (xaxis is not None) and (yaxis is not None):

                # deduce size max
                size_max = None
                if size is not None:
                    size_max = data[size].max()

                # lin reg param
                trendline = None
                if reg_switch:
                    trendline = 'ols'

                # set height and width
                width, height = return_width_height(width_input, height_input, 70, 70)

                fig = px.scatter(data_frame=data, x=xaxis, y=yaxis, color=color, size=size, size_max=size_max, trendline=trendline)
                return dcc.Graph(figure=fig, style={'width': f'{width}vh', 'height': f'{height}vh'})
            else:
                return html.Div()

        @app.callback(Output('plot_scatter_matrix_1', 'children'),
                      [Input('cols_scatter_matrix_1', 'value'), Input('color_scatter_matrix_1', 'value'),
                       Input('width_0', 'value'), Input('height_0', 'value')])
        def render_scatter_matrix_plot(cols, color, width_input, height_input):

            if (cols is not None):

                # set height and width
                width, height = return_width_height(width_input, height_input, 100, 100)

                fig = px.scatter_matrix(data_frame=data, dimensions=cols, color=color)
                fig.update_traces(diagonal_visible=False)
                return dcc.Graph(figure=fig, style={'width': f'{width}vh', 'height': f'{height}vh'})
            else:
                return html.Div()

        @app.callback(Output('plot_ts_1', 'children'), [Input('xaxis_ts_1', 'value'), Input('yaxis_ts_1', 'value')])
        def render_ts_plot(xaxis, yaxis):

            if (xaxis is not None) and (yaxis is not None):

                # sort data frame on x axis
                data.sort_values(by=xaxis, inplace=True)
                fig = px.line(data_frame=data, x=xaxis, y=yaxis)
                fig.update_xaxes(rangeslider_visible=True)
                return dcc.Graph(figure=fig, style={'height': '100vh'})

            else:
                return html.Div()

        @app.callback(Output('plot_2ts_1', 'children'),
                      [Input('xaxis_2ts_1', 'value'), Input('yaxis1_2ts_1', 'value'), Input('yaxis2_2ts_1', 'value')])
        def render_ts_plot(xaxis, yaxis1, yaxis2):

            if (xaxis is not None) and (yaxis1 is not None) and (yaxis2):

                # sort data frame on x axis
                data.sort_values(by=xaxis, inplace=True)

                fig1 = px.line(data_frame=data, x=xaxis, y=yaxis1)
                fig2 = px.line(data_frame=data, x=xaxis, y=yaxis2)

                # create ssubplots grid and add traces
                fig = make_subplots(2, 1, shared_xaxes=True)
                for trace in fig1.select_traces():
                    fig.add_trace(trace, row=1, col=1)

                for trace in fig2.select_traces():
                    fig.add_trace(trace, row=2, col=1)

                return dcc.Graph(figure=fig, style={'height': '100vh'})

            else:
                return html.Div()

        @app.callback(Output('module_1', 'children'), [Input('module_1_type', 'value')])
        def render_module(type: str):
            if type is not None:
                return self._module_hub(type, 1)
            else:
                return html.Div()


        return app

    # --------------------------------------------------------------------
    # ---------------------------- MODULE HUB ----------------------------
    # --------------------------------------------------------------------
    def _module_hub(self, type: str, id: int):
        '''Module selection'''

        if type == 'scatter':
            return self._scatter_module_builder(id)
        elif type == 'scatter_matrix':
            return self._scatter_matrix_module_builder(id)
        elif type == 'timeseries':
            return self._timeseries_module_builder(id)
        elif type == '2timeseries':
            return self._2timeseries_module_builder(id)
        else:
            raise NotImplementedError(type)



    # ------------------------------------------------------------------------
    # ---------------------------- MODULE BUILDER ----------------------------
    # ------------------------------------------------------------------------



    # ---------------------------- SCATTER MODULE ----------------------------
    def _scatter_module_builder(self, id: int):
        '''Build module for scatter plot'''

        # prepare ids
        xaxis_id = f'xaxis_scatter_{id}'
        yaxis_id = f'yaxis_scatter_{id}'
        color_id = f'color_scatter_{id}'
        size_id = f'size_scatter_{id}'
        reg_switch_id = f'reg_switch_{id}'
        plot_scatter_id = f'plot_scatter_{id}'


        module_list = [

            # vars selection
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id = xaxis_id,
                        options=build_options(self.df.columns),
                        multi=False,
                        placeholder='x variable',
                    ), width = 2),

                dbc.Col(
                    dcc.Dropdown(
                        id = yaxis_id,
                        options=build_options(self.df.columns),
                        multi=False,
                        placeholder='y variable',
                    ), width = 2),

                dbc.Col(
                    dcc.Dropdown(
                        id = color_id,
                        options=build_options(self.df.columns),
                        multi=False,
                        placeholder='hue variable',
                    ), width = 2),

                dbc.Col(
                    dcc.Dropdown(
                        id=size_id,
                        options=build_options(self.df.columns),
                        multi=False,
                        placeholder='size variable',
                    ), width=2),

                dbc.Col(
                   daq.ToggleSwitch(id = reg_switch_id, value = False, label = 'Regression line'), width=2)
            ]),

            # graph
            dbc.Row([
                dbc.Col(html.Div(id=plot_scatter_id))
            ])
        ] # end list

        return module_list


    # ---------------------------- SCATTER MATRIX MODULE ----------------------------
    def _scatter_matrix_module_builder(self, id: int) -> html.Div:
        '''Build module for scatter plot'''

        # prepare ids
        cols_scatter_matrix_id = f'cols_scatter_matrix_{id}'
        color_scatter_matrix_id = f'color_scatter_matrix_{id}'
        plot_scatter_matrix_id = f'plot_scatter_matrix_{id}'

        module_list = [

            # vars selection
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id = cols_scatter_matrix_id,
                        options=build_options(self.df.columns),
                        multi=True,
                        placeholder='variables list',
                    ), width = 3,  style={'display': 'inline-block'}),

                dbc.Col(
                    dcc.Dropdown(
                        id = color_scatter_matrix_id,
                        options=build_options(self.df.columns),
                        multi=False,
                        placeholder='hue variable',
                    ), width = 3,  style={'display': 'inline-block'}),

            ]),

            # graph
            dbc.Row([
                dbc.Col(html.Div(id = plot_scatter_matrix_id))
            ])
        ]

        return module_list

    # ---------------------------- TIMESERIES MODULE ----------------------------
    def _timeseries_module_builder(self, id: int) -> html.Div:
        '''Build single timeseries plot module'''

        # prepare ids
        xaxis_ts_id = f'xaxis_ts_{id}'
        yaxis_ts_id = f'yaxis_ts_{id}'
        plot_ts_id = f'plot_ts_{id}'

        module_list = [

            # vars selection
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id=xaxis_ts_id,
                        options=build_options(self.df.columns),
                        multi=False,
                        placeholder='x variable',
                    ), width=3, style={'display': 'inline-block'}),

                dbc.Col(
                    dcc.Dropdown(
                        id=yaxis_ts_id,
                        options=build_options(self.df.columns),
                        multi=True,
                        placeholder='y variables',
                    ), width=3, style={'display': 'inline-block'}),

            ]),

            # graph
            dbc.Row([
                dbc.Col(html.Div(id=plot_ts_id))
            ])
        ]


        return module_list





    # ---------------------------- 2 TIMESERIES MODULE ----------------------------
    def _2timeseries_module_builder(self, id: int) -> html.Div:
        '''Build single timeseries plot module'''

        # prepare ids
        xaxis_ts_id = f'xaxis_2ts_{id}'
        yaxis1_ts_id = f'yaxis1_2ts_{id}'
        yaxis2_ts_id = f'yaxis2_2ts_{id}'
        plot_ts_id = f'plot_2ts_{id}'

        module_list = [

            # vars selection
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id=xaxis_ts_id,
                        options=build_options(self.df.columns),
                        multi=False,
                        placeholder='x variable',
                    ), width=3, style={'display': 'inline-block'}),

                dbc.Col(
                    dcc.Dropdown(
                        id=yaxis1_ts_id,
                        options=build_options(self.df.columns),
                        multi=True,
                        placeholder='y variables',
                    ), width=3, style={'display': 'inline-block'}),

                dbc.Col(
                    dcc.Dropdown(
                        id=yaxis2_ts_id,
                        options=build_options(self.df.columns),
                        multi=True,
                        placeholder='y variables',
                    ), width=3, style={'display': 'inline-block'}),

            ]),

            # graph
            dbc.Row([
                dbc.Col(html.Div(id=plot_ts_id))
            ])
        ]

        return module_list

    # ---------------------------- 3 TIMESERIES MODULE ----------------------------








def _row_decorator(module_generator):
    def wrapper():
        module = dbc.Row(
            module_generator()
        )
        return module
    return wrapper





if __name__ == '__main__':

    # ----------------------- Load data --------------------------
    df = px.data.iris()
    df['valuedate'] = pd.date_range(datetime(2021, 1, 1), datetime(2021, 1, 1) + timedelta(days=len(df)-1), freq='D')

    dde = DashDataExplorer()
    dde.run(df, debug=True)
