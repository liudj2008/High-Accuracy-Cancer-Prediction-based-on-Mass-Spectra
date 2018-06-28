# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 12:35:29 2018

@author: Liu
"""
import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import glob
import re
import pickle
import random
import plotly.graph_objs as go
from plotly import tools 

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

'''
First part on data preparation
'''


file_list = glob.glob('pickle/*.pickle')

# All the files in file_list and store in file_dic
file_dic = {}
for i in file_list:
    
    # Extract the name of file
    name = re.split('[\\\.]',i)[1]
    
    # Load file to dictionary
    with open(i, 'rb') as f:
        file_dic[name] = pickle.load(f)

# Cancer group prepared by robotic
index = random.choices(range(len(file_dic['ovarian_robotic'][0])), k = len(file_dic['ovarian_robotic'][0]))
cancer_robotic_mean = np.mean(file_dic['ovarian_robotic'][0].loc[index,:])

# Control group prepared by robotic
index = random.choices(range(len(file_dic['ovarian_robotic'][1])), k = len(file_dic['ovarian_robotic'][1]))
control_robotic_mean = np.mean(file_dic['ovarian_robotic'][1].loc[index,:])

# Cancer group prepared by hand
index = random.choices(range(len(file_dic['ovarian_hand'][0])), k = len(file_dic['ovarian_hand'][0]))
cancer_hand_mean = np.mean(file_dic['ovarian_hand'][0].loc[index,:])

## Control group prepared by hand
index = random.choices(range(len(file_dic['ovarian_hand'][1])), k = len(file_dic['ovarian_hand'][1]))
control_hand_mean = np.mean(file_dic['ovarian_hand'][1].loc[index,:])

# Prostate Cancer group
index = random.choices(range(len(file_dic['prostate'][0])), k = len(file_dic['prostate'][0]))
cancer_prostate_mean = np.mean(file_dic['prostate'][0].loc[index,:])

# Prostate Control group
index = random.choices(range(len(file_dic['prostate'][1])), k = len(file_dic['prostate'][1]))
control_prostate_mean = np.mean(file_dic['prostate'][1].loc[index,:])
##########################################################################################
"""
plot_spectrum function

Input any spectrum as dataframe or directory, output 1D heatmap and/or the spectrum
Parameters:
file or file path
low_limit
high_limit
heatmap
plot
"""

def plot_spectrum(file, low_limit = 200, high_limit = 10000, heatmap=False, plot=True):
    file = file.loc[(file.iloc[:,0]> low_limit) & (file.iloc[:,0]<high_limit)]
    if heatmap and (not plot):
        data = go.Heatmap(z = file.iloc[:,1], x = file.iloc[:,0], 
                          y = np.repeat(0.1, len(file)), colorscale = 'Viridis',
                          colorbar=dict(title = 'Peak Intensity'))
        layout = dict(title = 'Spectrum Shown by Heatmap',
                      yaxis = dict(ticks = '', showticklabels=False),
                      xaxis = dict(title = 'M/Z'))
        return dict(data = [data], layout = layout)
    if plot and (not heatmap):
        data = go.Scatter(x = file.iloc[:,0], y=file.iloc[:,1])
        layout = dict(title = 'Spectrum Shown by Plot',
                      yaxis = dict(title = 'Peak Intensity'),
                      xaxis = dict(title = 'M/Z'))
        return dict(data=[data], layout=layout)
###################################################################################
'''

Plot PCA projection using first two principal components
Parameters:
file (in pd series) or directory path
group (optional, selected from 'robotic', 'hand' and 'prostate', default 'robotic')

2.2 Predict the probability of the sample being cancer or no cancer
Parameters:
file (in pd series) or directory path
model (optional, ['svc', 'knn', 'rf', 'ensemble'], default 'svc')
group (optional, ['robotic','hand','prostate'], default 'robotic')
criteria (optional, ['cancer', 'no cancer'], default 'cancer')
'''

# Plot projection of data using first two principal components
def data_transform(df):
    df.columns = ['M/Z', 'Intensity']
    # Select mass window between 200 and 10000
    df = df[(df['M/Z']>200) & (df['M/Z']<10000)]
    
    # Baseline substration. Substract the spectra with the median of lowest 20% intensity
    median = np.median(np.sort(df.Intensity)[:(len(df.Intensity)//5)])
    df.Intensity = df.Intensity.apply(lambda x: x-median if (x-median)>=0 else 0)
    
    # Rescale by dividing the intensity with the median of top 5% intensity and take the rootsqure
    top_median = np.median(np.sort(df.Intensity)[::-1][:(int(len(df.Intensity)*0.05))])
    df.Intensity = df.Intensity.apply(lambda x: x/top_median)
    df.Intensity = np.sqrt(df.Intensity)
    
    # Pivot table and return the resulting data frame
    df_transform= df.pivot_table(columns='M/Z')
    df_transform = df_transform.reset_index().iloc[:,1:]
    
    return df_transform

def pca_projection(file, group='robotic ovarian'):
    
    file = data_transform(file).loc[0]
        
    if group =='robotic ovarian':
        
        new_file = file_dic['ovarian_robotic_transformed_x_y'][0]
        label = file_dic['ovarian_robotic_transformed_x_y'][1]
        # Tranform file using fitted pca model by robotic ovarian training set
        file_transform = file_dic['pca_ovarian_robotic'].transform(file[np.newaxis,:])
        
        data1 = go.Scatter(x=new_file[:,0], y= new_file[:,1], mode = 'markers', marker = dict(color = label, size = 10))
        data2 = go.Scatter(x=[-7.5,10], y=[-4,3], mode = 'lines', line = dict(color = 'red', width = 3, dash='dash'))
        data3 = go.Scatter(x=file_transform[:,0], y=file_transform[:,1], mode = 'markers', marker = dict(size=30, color = 'green'))
        layout = go.Layout(title = 'Sample Projection in Robotic Prepared Ovarian Group',
                           showlegend = False, 
                           xaxis = dict(title = 'PC1'),
                           yaxis = dict(title = 'PC2'),
                           annotations = [dict(x=-2, y=6, text = 'Cancer', showarrow=False, 
                                                                   font=dict(size=30)),
                                                              dict(x=6, y=-4, text = 'Non Cancer', showarrow=False,
                                                                  font = dict(size=30))])
        return dict(data = [data1, data2, data3], layout=layout)

        
    if group == 'hand ovarian':
        
        new_file = file_dic['ovarian_hand_transformed_x_y'][0]
        label = file_dic['ovarian_hand_transformed_x_y'][1]
        # Tranform file using fitted pca model by hand prepared ovarian training set
        file_transform = file_dic['pca_ovarian_hand'].transform(file[np.newaxis,:])
    
        data1 = go.Scatter(x=new_file[:,0], y= new_file[:,1], mode = 'markers', marker = dict(color = label, size = 10))
        data2 = go.Scatter(x=file_transform[:,0], y= file_transform[:,1], mode = 'markers', marker = dict(color = 'green', size = 30))
        layout = go.Layout(title = 'Sample Projection in Hand Prepared Ovarian Group', 
                           showlegend = False,
                           xaxis = dict(title = 'PC1'),
                           yaxis = dict(title = 'PC2'))
        return dict(data = [data1, data2], layout=layout)
    
    if group == 'prostate':
        
        new_file = file_dic['prostate_transformed_x_y'][0]
        label = file_dic['prostate_transformed_x_y'][1]
        # Tranform file using fitted pca model by robotic ovarian training set
        file_transform = file_dic['pca_prostate'].transform(file[np.newaxis,:])
        
        new_file = file_dic['ovarian_robotic_transformed_x_y'][0]
        label = file_dic['ovarian_robotic_transformed_x_y'][1]
        # Tranform file using fitted pca model by robotic ovarian training set
        file_transform = file_dic['pca_ovarian_robotic'].transform(file[np.newaxis,:])
        
        data1 = go.Scatter(x=new_file[:,0], y= new_file[:,1], mode = 'markers', marker = dict(color = label, size = 10))
        data2 = go.Scatter(x=[-10,20], y=[-8,15], mode = 'lines', line = dict(color = 'red', width = 3, dash='dash'))
        data3 = go.Scatter(x=file_transform[:,0], y= file_transform[:,1], mode = 'markers', marker = dict(color = 'green', size = 30))
        layout = go.Layout(title = 'Sample Projection in Hand Prepared Prostate Group', 
                           showlegend = False, 
                           xaxis = dict(title = 'PC1'),
                           yaxis = dict(title = 'PC2'),
                           annotations = [dict(x=-6, y=6, text = 'Cancer', showarrow=False, 
                                                                   font=dict(size=30)),
                                                              dict(x=3, y=-6, text = 'Non Cancer', showarrow=False,
                                                                  font = dict(size=30))])
        return dict(data = [data1, data2,data3], layout=layout)

# Predict the probability of the sample being cancer
        
# Predict the probability of the sample being cancer
        
def predict_cancer(file, group):
    
    file = data_transform(file).loc[0]
        
    if group == 'robotic ovarian':
        
        model_total = {'svc':file_dic['svc_ovarian_robotic'], 
                         'knn':file_dic['knn_ovarian_robotic'], 
                         'rf':file_dic['rf_ovarian_robotic'],
                         'ensemble': file_dic['ensemble_ovarian_robotic']}
        
        important_features = file_dic['feature_index_ovarian']
        
    if group == 'hand ovarian':
        
        model_total = {'svc':file_dic['svc_ovarian_hand'], 
                      'knn':file_dic['knn_ovarian_hand'], 
                      'rf':file_dic['rf_ovarian_hand'], 
                      'ensemble': file_dic['ensemble_ovarian_hand']}
        
        important_features = file_dic['feature_index_ovarian_hand']
    
        
    if group == 'prostate':
    
        model_total = {'svc':file_dic['svc_prostate'], 
                          'knn':file_dic['knn_prostate'], 
                          'rf':file_dic['rf_prostate'], 
                          'ensemble': file_dic['ensemble_prostate']}
        
        important_features = file_dic['feature_index_prostate']
    
    return file, model_total, important_features


def plot_proba_cancer(file, group='robotic ovarian'):
    
    file, model_total, important_features = predict_cancer(file, group)
    
    data = []
    for model in model_total.keys():
        proba = model_total[model].predict_proba(file.iloc[important_features][np.newaxis,:])

        trace = go.Bar(x=['Cancer','No Cancer'], y= [proba[:,1][0]*100, proba[:,0][0]*100], name = model)
        data.append(trace)

    layout = go.Layout(title = 'Predicted Probability by Four Models', barmode='group', xaxis = dict(title = 'Groups'), yaxis = dict(title = 'Probability(%)'))
    return go.Figure(data = data, layout=layout)
        
###################################################################################
'''For unknown samples
Input any spectrum as dataframe or directory, output sample projection in the first two principal components space, and the probability of samples belongs to each class according to different criteria

1. View whether the unknown sample belongs to any group in the first two principal components space
Parameters:
file (in pd series) or directory path

2. Multiclassfication
Parameters:
file (in pd series) or directory path
model (optional, ['svc', 'knn', 'rf', 'ensemble'], default 'svc')
criteria (optional, ['all', 'preparation', 'sex'], default 'all') 
'''

def view_relation(file):
    file = data_transform(file).loc[0]
        
    xdata_transform = file_dic['whole_data_multiclass'][0]
    file_transform = file_dic['pca_whole_data'].transform(file[np.newaxis,:])
    
    data1 = go.Scatter(x=xdata_transform[:,0],y=xdata_transform[:,1], mode = 'markers', name = 'Traning Samples')
    data2 = go.Scatter(x=file_transform[:,0], y=file_transform[:,1], mode='markers', marker = dict(color = 'red', size = 10), name = 'Unknown Sample')
    layout = go.Layout(
                       title = 'Sample Projection using First Two Principal Components',
                       xaxis = dict(title = 'PC1'),
                       yaxis = dict(title = 'PC2'))
    return dict(data = [data1, data2], layout=layout)

# Multiclass prediction
def predict_proba_multiclass(file, criteria = 'all'):
    file = data_transform(file).loc[0]
        
    data = []
    if criteria == 'sex':
        proba = pd.DataFrame(columns = ['Female', 'Male'])
        important_index = file_dic['important_features_sex']
        
        for model in ['svc','knn','rf','ensemble']:
            model_fit = file_dic[model+'_predict_sex']
            proba.loc[0] = model_fit.predict_proba(file.iloc[important_index][np.newaxis,:])[0]
            trace = go.Bar(x = proba.columns, y=proba.values[0]*100, name = model)
            data.append(trace)
    
        layout = go.Layout(title = 'Predicted Probability by Four Models',
                           xaxis = dict(title = 'Groups'),
                           yaxis = dict(title = 'Probability(%)'),
                           barmode = 'group')
        return dict(data = data, layout= layout)

    
    if criteria == 'preparation':
        proba = pd.DataFrame(columns = ['Robotic', 'Hand'])
        important_index = file_dic['important_features_robotic_hand']
        
        for model in ['svc','knn','rf','ensemble']:
            model_fit = file_dic[model+'_predict_robotic_hand']
            proba.loc[0] = model_fit.predict_proba(file.iloc[important_index][np.newaxis,:])[0]
            trace = go.Bar(x = proba.columns, y=proba.values[0]*100, name = model)
            data.append(trace)
    
        layout = go.Layout(title = 'Predicted Probability by Four Models',
                           xaxis = dict(title = 'Groups'),
                           yaxis = dict(title = 'Probability(%)'),
                           barmode = 'group')
        return dict(data = data, layout= layout)
          
    if criteria =='all':
        proba = pd.DataFrame(columns = ['robotic ovarian cancer', 'robotic ovarian control',
                                        'hand ovarian cancer','hand ovarian control',
                                        'prostate cancer','prostate control'])
        important_index = file_dic['important_features_multiclass']
        
        for model in ['svc','knn','rf','ensemble']:
            model_fit = file_dic[model+'_predict_multiclass']
            proba.loc[0] = model_fit.predict_proba(file.iloc[important_index][np.newaxis,:])[0]
            trace = go.Bar(x = proba.columns, y=proba.values[0]*100, name = model)
            data.append(trace)
    
        layout = go.Layout(title = 'Predicted Probability by Four Models',
                           xaxis = dict(title = 'Groups'),
                           yaxis = dict(title = 'Probability(%)'),
                           barmode = 'group')
        return dict(data = data, layout= layout)
    
################################################################################   
'''
4. Selected important molecules determining cancer
INPUT group information, low_limit and high_limit
OUTPUT the important mass values,
       the heatmap of selected molecular mass with cancer group,
       the heatmap of selected molecular mass with non cancer group

Parameters:
group (optional, ['robotic prepared ovarian', 'hand prepared ovarian', 'prostate'], default 'robotic prepared ovarian')
low_limit (optional, should be higher than 200, default 200)
high_limit (optional, should be lower than 10000, default 10000)

'''

def plot_select_molecules(group, mass, file1, file2):
    
    data1 = go.Heatmap(x = mass.columns, y=np.repeat(0.1, len(mass.columns)), z = mass.values[0], colorscale = 'Viridis',
                      showscale = False)
       
    data2 = go.Heatmap(x = file1.index, y=np.repeat(0.1, len(file1.index)), z = file1.values, colorscale = 'Viridis',
                      showscale = False)
    
    data3 = go.Heatmap(x = file2.index, y=np.repeat(0.1, len(file2.index)), z = file2.values, colorscale = 'Viridis',
                       showscale = False)
    
    fig = tools.make_subplots(rows=3, cols=1, subplot_titles = ('Fingerprint masses for ' + group + ' samples',
                                                                'Full load spectrum by averaging ' + group + ' cancer samples',
                                                                'Full load spectrum by averaging ' + group + ' control samples'))
    fig.append_trace(data1, 1, 1)
    fig.append_trace(data2, 2, 1)
    fig.append_trace(data3, 3, 1)
    
    fig['layout']['xaxis1'].update(ticks='', showticklabels=False)
    fig['layout']['xaxis2'].update(ticks='', showticklabels=False)
    fig['layout']['xaxis3'].update(title = 'M/Z')

    for i in range(1,4): fig['layout']['yaxis'+str(i)].update(ticks='', showticklabels=False) 

    return fig

def select_molecules(group, low_limit, high_limit):
        
    mass = file_dic['df_mass_important'].copy()
    
    if group == 'robotic ovarian':
        
        mass.iloc[0,file_dic['feature_index_ovarian']]=1
        mass = mass.loc[:, mass.columns[(mass.columns>low_limit) & (mass.columns<high_limit)]]
        
        cancer_robotic_mean_new = cancer_robotic_mean[(cancer_robotic_mean.index>low_limit) & 
                                                  (cancer_robotic_mean.index<high_limit)]
        control_robotic_mean_new = control_robotic_mean[(control_robotic_mean.index>low_limit)&
                                                    (control_robotic_mean.index<high_limit)]
        
        mass_important = mass.columns[mass.loc[0]==1]
        
        num_selected_mol =('The number of fingerprint masses between {} and {} are: {}'.format(low_limit, high_limit, len(mass_important)))
        selected_mol = ('The fingerprint masses are: \n{}'.format(list(mass_important.astype(int))))
        
        fig = plot_select_molecules(group, mass, cancer_robotic_mean_new, control_robotic_mean_new)
        return num_selected_mol, selected_mol, fig
    
    if group == 'hand ovarian':
        mass.iloc[0, file_dic['feature_index_ovarian_hand']]=1
        mass = mass.loc[:, mass.columns[(mass.columns>low_limit) & (mass.columns<high_limit)]]
        
        cancer_hand_mean_new = cancer_hand_mean[(cancer_hand_mean.index>low_limit) & 
                                                  (cancer_hand_mean.index<high_limit)]
        control_hand_mean_new = control_hand_mean[(control_hand_mean.index>low_limit)&
                                                    (control_hand_mean.index<high_limit)]
        
        mass_important = mass.columns[mass.loc[0]==1]
        num_selected_mol = ('The number of fingerprint masses between {} and {} are: {}'.format(low_limit, high_limit, len(mass_important)))
        selected_mol = ('The fingerprint masses are: \n{}'.format(list(mass_important.astype(int))))
        
        fig = plot_select_molecules(group, mass, cancer_hand_mean_new, control_hand_mean_new)
        
        return num_selected_mol, selected_mol, fig
    
    if group == 'prostate':
        mass.iloc[0, file_dic['feature_index_prostate']] = 1
        mass = mass.loc[:, mass.columns[(mass.columns>low_limit) & (mass.columns<high_limit)]]
        
        cancer_prostate_mean_new = cancer_prostate_mean[(cancer_prostate_mean.index>low_limit) & 
                                                  (cancer_prostate_mean.index<high_limit)]
        control_prostate_mean_new = control_prostate_mean[(control_prostate_mean.index>low_limit)&
                                                    (control_prostate_mean.index<high_limit)]
        
        mass_important = mass.columns[mass.loc[0]==1]
        num_selected_mol = ('The number of fingerprint masses between {} and {} are: {}'.format(low_limit, high_limit, len(mass_important)))
        selected_mol = ('The fingerprint masses are: \n{}'.format(list(mass_important.astype(int))))
        
        fig = plot_select_molecules(group, mass, cancer_prostate_mean_new, control_prostate_mean_new)
        return num_selected_mol, selected_mol, fig


################################################################################
app = dash.Dash()
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

# Function 1. Show spectrum
app.layout = html.Div([
        html.Div('Welcome to Cancer Diagnosis 1.0', 
                 style={'fontSize': 25, 'marginLeft': 10, 'marginBottom': 25, 'marginTop': 25}),
        html.Hr(),
        html.Details([
        html.Summary('About', style = dict(fontSize = 20)),
        html.Div(dcc.Markdown('''
        
THIS APP IS FOR DEMONSTRATION PURPOSE ONLY!
* This web app is developed for diagnosing ovarian cancer and prostate cancer based on mass spectrum data. 
* 585 Training samples were mass spectra retrieved from National Cancer Institute (NCI) database, 
including robotic prepared ovarian samples (253 samples), hand prepared ovarian samples (200 samples) and prostate samples(132 samples).
Each group has cancer and non-cancer (control) samples

        
        '''))
                     
        
        ]), 
        html.Details([
        html.Summary('Instructions', style = dict(fontSize = 20)),
        html.Div(dcc.Markdown('''
        
This is a brief demo of how to use this app for cancer diagnosis

>**Not sure which group your sample belongs to?**

Just **upload** the mass spectrum of your sample. This App will tell you
* Mass spectrum (heatmap and plot)
* Visualization of your sample projection within all the training samples
* Prediction of probabilities of your sample belonging to different classes according to selected criteria

>**Sure on which group your sample belongs to?**

Just **upload** the mass spectrum of your sample, **select** sample group with options of robotic ovarian samples, hand ovarian samples and prostate samples.
This App will tell you
* Mass spectrum (heatmap and plot)
* Visualization of your sample projection within training samples in your selected group
* Prediction of probabilities of your sample being cancer or no cancer
* Selected fingerprint masses for cancer diagnosis

##### *Simple and Fun! Be Sure to Play with it!*       
        
        '''))
                     
        
        ]),
        html.Hr(),
        html.Div(id = 'file_upload', children = 
                 [html.P('Please upload mass spectrum file:', style = dict(fontSize = 20, fontWeight = 'bold')),
                  dcc.Upload(id='upload_data', children = html.Button('Upload File')),
                  html.Br(),
                  html.P('Please select sample group. If unknown, select \'Unknown\'', style = dict(fontSize = 20, fontWeight = 'bold')),
                  dcc.Dropdown(id = 'group_selection',
                               options = [dict(label='Robotic Ovarian Samples', value = 'robotic ovarian'),
                                          dict(label='Hand Ovarian Samples', value = 'hand ovarian'),
                                          dict(label='Prostate Samples', value = 'prostate'),
                                          dict(label='Unknown Samples', value = 'Unknown')],
                                          value = 'Unknown')]),
        html.Hr(),
       
        html.H5('Sample Mass Spectrum', style = dict(color='blue', fontWeight = 'bold')),
               
        html.Div(
                [html.P('Please select mass range to show mass spectrum:'),
                 dcc.RangeSlider(
                       id='mass_range_slider0',
                       value=[200, 10000],
                       allowCross = False,
                       marks = {i:str(i) for i in range(0,20001,1000)}
                    ),
                 html.Br(),
                 html.P(id = 'mass_range_selected')]),
        html.Div(
                [
                        
                html.Div(
                dcc.Graph(
                id = 'example0'
                ), className = 'six columns'),
                        
                html.Div(dcc.Graph(
                id = 'example1'
                ), className = 'six columns')]),
               
        html.H5('Cancer Prediction Results', style = dict(color='blue', fontWeight = 'bold')),
        html.P('Please select classification criteria:'),
        dcc.Dropdown(id = 'classification_criteria', value = 'all'),
        html.Div(
                [html.Div(dcc.Graph(
                id = 'example2'
                ),className = 'six columns'),
                html.Div(dcc.Graph(
                id = 'example3'
                ),className = 'six columns')]),
        
                
        html.Div(id = 'mass_selection',
                children = [
                html.H5('Fingerprint Masses for Cancer Diagnosis', style = dict(color='blue', fontWeight = 'bold')),
                html.P('Please select mass range:'),
                html.Div(
                dcc.RangeSlider(
                       id='mass_range_slider',
                       min=200,
                       max=10000,
                       step=100,
                       marks={i:str(i) for i in range(1000,10001,1000)},
                       value=[200, 1000],
                       allowCross = False,
                    ), style= {'left':'10%','width': '50%'}),
                html.Br(),
                html.P(id = 'mass_selected'),
                html.Br(), 
                html.P(id = 'select_molecules_num'),
                html.P(id = 'select_molecules'),
                   
                dcc.Graph(
                id = 'example4'
                )
        ]),
        
        html.Hr(),
        html.Div(
                [dcc.Markdown('''*Authored by Peter Liu | COPYRIGHT 2018*'''),
                dcc.Markdown('''For more info, please visit my [LinkedIn](https://www.linkedin.com/in/dajiangliu/) and [Github](https://github.com/liudj2008/High-Accuracy-Cancer-Prediction-based-on-Mass-Spectra)''' )])
    ])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename.lower():
        # Assume that the user uploaded a CSV file
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        df = pd.read_excel(io.BytesIO(decoded))
    
    return df



@app.callback(
        Output('mass_range_slider0','min'),
       [Input('upload_data','contents'),
         Input('upload_data','filename')])
def min_mass(contents, filename):
    file = parse_contents(contents, filename)
    return np.min(file.iloc[:,0])

@app.callback(
        Output('mass_range_slider0','max'),
       [Input('upload_data','contents'),
         Input('upload_data','filename')])
def max_mass(contents, filename):
    file = parse_contents(contents, filename)
    return np.max(file.iloc[:,0])

@app.callback(
        Output('mass_range_selected','children'),
        [Input('mass_range_slider0','value')])
def mass_range_selected(value):
    return 'You have selected mass range from {} to {}'.format(value[0], value[1])

@app.callback(
        Output('example0','figure'),
        [Input('upload_data','contents'),
         Input('upload_data','filename'),
         Input('mass_range_slider0','value')])

def output_update(contents, filename, value):
    file = parse_contents(contents, filename)
    return plot_spectrum(file, low_limit = value[0], high_limit=value[1], heatmap=True, plot=False)

@app.callback(
        Output('example1','figure'),
        [Input('upload_data','contents'),
         Input('upload_data','filename'),
         Input('mass_range_slider0','value')])


def output_update1(contents, filename, value):
    file = parse_contents(contents, filename)
    return plot_spectrum(file, low_limit = value[0], high_limit=value[1], heatmap=False, plot=True)

@app.callback(
        Output('classification_criteria', 'options'),
        [Input('group_selection','value')]
                  
)
                  
def criteria(value):
    if value == 'Unknown':
        return [dict(label='All', value = 'all'),
                dict(label='Sex', value = 'sex'),
                dict(label='Preparation', value = 'preparation')]
    else:
        return [dict(label='Cancer/No Cancer', value = 'Cancer/No Cancer')]
        
@app.callback(
        Output('example2','figure'),
        [Input('group_selection','value'),
         Input('upload_data','contents'),
         Input('upload_data','filename')])
def classification(value, contents, filename):
    
    file = parse_contents(contents, filename)
    if value == 'Unknown':
        return view_relation(file)
    else: 
        return pca_projection(file, group=value)

@app.callback(
        Output('example3','figure'),
        [Input('group_selection','value'),
         Input('upload_data','contents'),
         Input('upload_data','filename'),
         Input('classification_criteria','value')])
def classification1(value, contents, filename, criteria):
    file = parse_contents(contents, filename)
    if value == 'Unknown':
        return predict_proba_multiclass(file, criteria = criteria)
    else: 
        return plot_proba_cancer(file, group = value)
    
@app.callback(
        Output('mass_selection','style'),
        [Input('group_selection','value'),
         Input('upload_data','contents'),
         Input('upload_data','filename'),
         Input('mass_range_slider','value')])
def show_mass_selection(value, contents, filename, mass_value):
    file = parse_contents(contents, filename)
    if value =='Unknown':
        return dict(display='none')
    
@app.callback(
        Output('select_molecules_num','children'),
        [Input('group_selection','value'),
         Input('upload_data','contents'),
         Input('upload_data','filename'),
         Input('mass_range_slider','value')])
def select_molecules1(value, contents, filename, mass_value):
    file = parse_contents(contents, filename)
    if value!='Unknown':
        return select_molecules(group = value, low_limit = mass_value[0], high_limit = mass_value[1])[0]

@app.callback(
        Output('select_molecules','children'),
        [Input('group_selection','value'),
         Input('upload_data','contents'),
         Input('upload_data','filename'),
         Input('mass_range_slider','value')])
def select_molecules2(value, contents, filename, mass_value):
    file = parse_contents(contents, filename)
    if value!='Unknown':
        return select_molecules(group = value, low_limit = mass_value[0], high_limit = mass_value[1])[1]
    
    
@app.callback(
        Output('example4','figure'),
        [Input('group_selection','value'),
         Input('upload_data','contents'),
         Input('upload_data','filename'),
         Input('mass_range_slider','value')])
def select_molecules3(value, contents, filename, mass_value):
    file = parse_contents(contents, filename)
    if value!='Unknown':
        return select_molecules(group = value, low_limit = mass_value[0], high_limit = mass_value[1])[2]


@app.callback(
        Output('mass_selected','children'),
        [Input('mass_range_slider','value')])
def mass_range(value):
    return 'You have selected mass range from {} to {}'.format(value[0], value[1])

if __name__=='__main__':
    app.run_server(debug=True)
