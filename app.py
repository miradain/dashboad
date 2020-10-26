import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go


import pandas_datareader as pdr
import datetime 
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt 
#import seaborn as sns
#import statsmodels.api as sm
from yahoo_fin import stock_info as si 

# Extracting data function

tickers=['GLE.PA', 'ML.PA', 'ENGI.PA', 'TEP.PA', 'EN.PA', 'BNP.PA', 'VIE.PA', 'CA.PA', 'KER.PA', 'SU.PA']
startdate=datetime.datetime(2005, 1, 1)
enddate=datetime.datetime(2020, 10, 20)


def retrieve_data(tickers, startdate=startdate, enddate=enddate):
  def data(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map (data, tickers)
  data=pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])
  # Isolate the `Adj Close` values and transform the DataFrame
  portfolio = data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

  # Compute log return
  returns=np.log(1+portfolio.pct_change()).dropna()

  return(returns)

df=retrieve_data(tickers, startdate=startdate, enddate=enddate)
df_csub=df.copy()
df = df.reset_index()
# Compute cummulative return
for ticker in tickers:
    df[ticker]= np.cumprod(1 + df[ticker].values) - 1
#df[tickers]=((1+df[tickers]).cumprod()-1)

Assets=["SG", "Michelin", "Engie", "Teleperformance", "Bouygues", "BNP", "Veolia", "Carrefour","Kering", "Schneider"]


## Monte Carlo vs TDA weight data frame
w_mc= np.array([0.002690, 0.053972,  0.047570, 0.199497, 0.014557, 0.015709, 0.241335, 0.049635, 0.249791, 0.125243])
w_tda= np.array([ 0.0000404, 0.000364, 0.002657, 0.0008564, 0.001257, 0.000106, 0.0029905, 0.0021241, 0.95415, 0.035585])
df_weights=pd.DataFrame({"assets":np.array(Assets), "weights_mc": w_mc, "weights_tda": w_tda})


## Monte Carlo vs TDA cummulative return data frame

return_tda= np.array(df_csub[tickers]).dot(np.array(w_tda).T)
cum_return_tda = np.cumprod(1+ return_tda) - 1

return_mc= np.array(df_csub[tickers]).dot(np.array(w_mc).T)
cum_return_mc = np.cumprod(1 + return_mc) - 1

portfolio_methods=pd.DataFrame({'Date':df['Date'], 'TDA':cum_return_tda, 'MC':cum_return_mc})





# Method VaR data
#df_var=pd.read_csv( '/Users/miradain/Documents/Financial mathematics/VaR.portfolio.csv', sep=';')
df_var=pd.read_csv( 'VaR.portfolio.csv', sep=';')
df_var = df_var.stack().str.replace(',','.').unstack()
df_var["VaR"]= pd.to_numeric(df_var["VaR"],errors='coerce').abs()
df_var.loc[:,"VaR"] *=100
df_var["CVaR"]=pd.to_numeric(df_var["CVaR"],errors='coerce').abs()
df_var.loc[:,"CVaR"] *=100

trace_var=go.Bar(name='MC loss risk(€)', x=df_var['Methods'], y=df_var["VaR"])
trace_cvar=go.Bar(name='TDA loss risk(€)', x=df_var['Methods'], y=df_var["CVaR"])


# Assets VaR data
#df_var_assets=pd.read_csv( '/Users/miradain/Documents/Financial mathematics/VaR.assets.csv', sep=';')
df_var_assets=pd.read_csv( 'VaR.assets.csv', sep=';')
df_var_assets = df_var_assets.stack().str.replace(',','.').unstack()

df_var_assets["VaR"]= pd.to_numeric(df_var_assets["VaR"],errors='coerce').abs()
df_var_assets.loc[:,'VaR'] *= 100
#df_var_assets["VaR"] = df_var_assets["VaR"].apply(format)

df_var_assets["Expected_Shortfall"]=pd.to_numeric(df_var_assets["Expected_Shortfall"],errors='coerce').abs()
df_var_assets.loc[:,"Expected_Shortfall"] *= 100
#df_var_assets["Expected_Shortfall"] = df_var_assets["Expected_Shortfall"].apply(format)

trace_var_assets=go.Bar(name='Expected loss (€) ', x=df_var_assets['Assets'], y=df_var_assets["VaR"])
trace_cvar_assets=go.Bar(name='Extreme mean loss (€)', x=df_var_assets['Assets'], y=df_var_assets["Expected_Shortfall"])










# Creates a list of dictionaries, which have the keys 'label' and 'value'.
def get_options(Assets):
    N=len(Assets)
    dict_list = []
    for i in range(N):
        dict_list.append({'label': Assets[i], 'value': tickers[i]})
    return dict_list



# Plot the stack barplot

trace1=go.Bar(name='MC weight', x=df_weights['assets'], y=df_weights["weights_mc"])
trace2=go.Bar(name='TDA weight', x=df_weights['assets'], y=df_weights["weights_tda"])

 










# Initialise the app
app = dash.Dash(__name__)
server = app.server

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Define the app
app.layout = html.Div(
    children=[
         html.Div( className='row',  # Define the row element
                children = [
                    html.Div( className='four columns div-user-controls',
                          children=[
                             html.H1('Dashboard-Comparing Monte Carlo simulation and Topological Data Analysis (TDA) in portfolio optimization', style={
            'textAlign': 'center',
            'color': '#5E0DAC'
        }),
                             dcc.Markdown(''' We consider two portfolios: One built using the modern portfolio theory and the other using TDA. The two portfolios contain stocks from the CaC40 index. These are
                            :  **Société Générale, Michelin, Engie, Teleperformance, Bouygues, BNP Paribas, Veolia Environ, Carrefour, Kering** and finally **Schneider electic**.
                             We present for each portfolio:  
                              '''),
                              dcc.Markdown('''* The percentage of gain over more than 7 years;
                                               '''),
                            dcc.Markdown('''* The risk or possible loss (in euros) per day for a €100 investment. 
                                               '''),
                             
                             html.H2('Percentage of gain over years', style={'color': '#375CB1'}),
                             html.P('''Pick one or more stocks from the dropdown below.'''),
                             html.Div(
                                 className='div-for-dropdown',
                                 children=[
                                     dcc.Dropdown(id='stockselector',
                                             options=get_options(Assets),
                                             multi=True,
                                             value=tickers,
                                             #style={'backgroundColor': '#1E1E1E'},
                                             className='stockselector'
                                             ),
                                   ],
                                   style={'color': '#1E1E1E'},
                                   #titlefont= {"size": 12, "color": "#000000"}
                                   )
                               ]
                    ),
                    html.Div( className='eight columns fiv-for-charts bg-grey',
                          children=[
                             dcc.Graph(id='timeseries',config={'displayModeBar': False}),
                             html.H2('Stock risk (in euros)', style={'color': '#375CB1'}),
                             html.P('''If an investor put € 100 on any of these stocks, how much money could they lose each day in a worst-case scenario? '''),
                             #html.P(''' To find out, play with the bar chart below where there is an expected extreme loss and the average extreme loss with a false tolerance over 5 times over 100 cases.. '''),
                             dcc.Graph(id='bar_plot_assets_var1',
                                       #figure=go.Figure()
                                       figure=go.Figure(data=[trace_var_assets, trace_cvar_assets], 
                                                        layout=go.Layout(barmode='group', plot_bgcolor='rgb(254, 247, 234)', yaxis=dict(title='€'))),
                                       #figure.update_layout(yaxis_title="€")
                                       # Prefix y-axis tick labels with dollar sign
                                       #figure.update_yaxes(tickprefix="$")
                                      ),

                             #dcc.Graph(id='change', config={'displayModeBar': False}),
                             html.H2('Weights repartition by portfolio', style={'color': '#375CB1'}),
                             dcc.Markdown(''' * The MC weight is given by the Maximum Sharpe ratio portfolio (MSRP) in modern portfolio theory;    
                                              '''),
                            dcc.Markdown('''  * The TDA weight was discussed in our first paper of this series: [Investing on CaC40](https://github.com/miradain/Topological-Data-Analysis/blob/master/How%20to%20invest%20on%20CaC40-Revisited.pdf).
                                              '''),
                             dcc.Graph(id='bar_plot',
                                       figure = go.Figure(data=[trace1, trace2], 
                                                        layout=go.Layout(barmode='stack', plot_bgcolor='rgb(254, 247, 234)'))
                                       ),
                            html.H2('Percentage of gain over years by portfolio', style={'color': '#375CB1'}),
                            dcc.Graph(id='cummulative_returns',
                                       config={'displayModeBar': False},
                                       animate=True,
                                       figure=px.line(portfolio_methods,
                                                      x='Date',
                                                      y= ['TDA', 'MC'],
                                                      template= 'plotly_dark').update_layout(
                                                                { 'plot_bgcolor': 'rgb(254,247,234)',
                                                                 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
                                        ),
                            html.H2(' Portfolio risk (in euros) ', style={'color': '#375CB1'}),
                            html.P('''If an investor put €100 in a portfolio provided by one of these two approaches, how much money could they lose each day in the worst case scenario? '''),
                            #html.P(''' To find out, play with the bar chart below where there is an expected extreme loss and the average extreme loss with a false tolerance over 5 times over 100 cases. '''),
                            
                            dcc.Graph(id='bar_plot_method_var',
                                       figure = go.Figure(data=[trace_var, trace_cvar], 
                                                        layout=go.Layout(barmode='group', plot_bgcolor='rgb(254, 247, 234)', yaxis=dict(title='€')))
                                      ),
                            html.H2('Reference(s)', style={'color': '#375CB1'}),
                            #html.P(''' Your are interested by the risk methodology, then'''),
                            #html.Link(href='http://localhost/https://miradain.github.io/', refresh=True, children=['www.google.com']),
                            #html.Link('https://miradain.github.io/')
                            #dcc.Location('/Users/miradain/Documents/Financial mathematics/Managing_Risk_CaC40_final.html'),
                            #html.Div(src='/Users/miradain/Documents/Financial mathematics/Managing_Risk_CaC40_final.html')
                            #html.A('Your are interested by the used risk methodology, then click here.', href='https://miradain.github.io/Automated_portfolio_risk2.html'),
                            dcc.Markdown('''Your are interested in our risk methodology, then click [here](https://miradain.github.io/Automated_portfolio_risk2.html)''') 
                              ])
                ])
    ]
)
                                

#methods=['cReturn_tda', 'cReturn_mc']




@app.callback(Output('timeseries', 'figure'),
              [Input('stockselector', 'value')])
def update_timeseries(tickers):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    # STEP 1
    trace = []  
    df_sub = df
    # STEP 2
    # Draw and append traces for each stock
    for ticker in tickers:   
        trace.append(go.Scatter(x=df_sub['Date'],
                                 y=df_sub[ticker],
                                 mode='lines',
                                 opacity=0.7,
                                 name=ticker,
                                 textposition='bottom center'))  
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
     
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  #colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056', "#5E0DAC", '#fdca26', '#fb9f3a', '#d8576b'],
                  colorway=['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  #plot_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgb(254, 247, 234)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
                  #xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
              ),

              }

    return figure


"""
@app.callback(Output('change', 'figure'),
              [Input('stockselector', 'value')])
def update_change(tickers):
    ''' Draw traces of the feature 'change' based one the currently selected stocks '''
    df_sub = df_weights
    # Draw and append traces for each stock
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sub['Date'],
        y=df_sub["weights_mc"],
        name='MC weight',
        marker_color='indianred'
        ))
   
    fig.add_trace(go.Bar(
        x=df_sub['Date'],
        y=df_sub["weights_tda"],
        name='TDA weight',
        marker_color='lightsalmon'
        ))
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45) 

    return fig
"""
"""
@app.callback(Output('var', 'children'),
              [Input('stockselector1', 'value')])

def update_timeseries1(ticker):
    #trace = []  
    # Draw and append traces for each stock
    #for ticker in tickers:  
     #   ticker_var=df_var_assets[df_var_assets[Assets]==ticker]
     #   var=ticker_var['VaR'] 
      #  trace.append(var*1000)  
    # STEP 3
    ticker_var=df_var_assets[df_var_assets[Assets]==ticker]
    #traces = [trace]
    var=ticker_var['VaR']
    loss=var*1000
    #data = [val for sublist in traces for val in sublist]
    return html.Div(loss)
"""
















# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=True)
