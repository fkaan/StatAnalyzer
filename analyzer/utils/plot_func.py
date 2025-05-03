import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from io import StringIO
import base64

def plot_histogram(data, column, nbins=20):
    """Create interactive histogram plot"""
    fig = px.histogram(data, x=column, nbins=nbins,
                      title=f'Distribution of {column}',
                      labels={column: column},
                      marginal='box')
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12))
    
    return fig.to_html(full_html=False)

def plot_boxplot(data, column, group_column=None):
    """Create interactive boxplot with optional grouping"""
    if group_column:
        fig = px.box(data, x=group_column, y=column,
                    title=f'Boxplot of {column} by {group_column}',
                    color=group_column)
    else:
        fig = px.box(data, y=column,
                    title=f'Boxplot of {column}')
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12))
    
    return fig.to_html(full_html=False)

def plot_scatter(data, x_col, y_col, trendline=False):
    """Create interactive scatter plot with optional trendline"""
    fig = px.scatter(data, x=x_col, y=y_col,
                    title=f'{y_col} vs {x_col}',
                    trendline='ols' if trendline else None)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12))
    
    return fig.to_html(full_html=False)

def plot_correlation_matrix(data):
    """Create interactive correlation matrix heatmap"""
    numeric_df = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10))
    
    return fig.to_html(full_html=False)

def plot_bar_chart(data, x_col, y_col):
    """Create interactive bar chart"""
    fig = px.bar(data, x=x_col, y=y_col,
                title=f'{y_col} by {x_col}')
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12))
    
    return fig.to_html(full_html=False)