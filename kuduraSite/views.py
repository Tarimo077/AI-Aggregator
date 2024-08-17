from django.shortcuts import render, redirect
import requests
from django.contrib import messages
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from requests.auth import HTTPBasicAuth
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.io as pio
from django.core.paginator import Paginator
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.conf import settings
import os

# Define paths to the model and scaler files
iso_forest_path = os.path.join(settings.BASE_DIR, 'models', 'iso_forest.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'models', 'scaler.pkl')
multi_output_rf_path = os.path.join(settings.BASE_DIR, 'models', 'multi_output_rf.pkl')
file_path_test = os.path.join(settings.BASE_DIR, 'models', 'new_data.csv')

# Load model and scaler
multi_output_rf = joblib.load(multi_output_rf_path)
scaler = joblib.load(scaler_path)
iso_forest = joblib.load(iso_forest_path)


# Constants
BASE_URL = "http://5.22.218.175:1880/"
AUTH = HTTPBasicAuth('kudura', 'pw4kudura')
appliance_labels = ['refrigerator', 'microwave', 'coffee_maker', 'cake_mixer']

def logout_page(request):
    logout(request)
    return redirect('login')

def login_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')

@login_required
def homepage(request):
    range_value = request.GET.get('range', 9999999)
    data = fetch_data_index('kuduraData', range_value)

    if not data:
        messages.warning(request, 'No data for the selected range. Showing default data.')
        data = fetch_data_index('kuduraData', 9999999)
        range_value = 9999999

    df = pd.DataFrame(data)
    total_kwh_used = df['totalKwh'].sum()
    total_kwh_bought = df['kwhPurchased'].sum()
    df['legend'] = df['reference'] + "("+df['meterNumber']+")"
    connections = len(df)
    consumption_pie = create_pie_chart(df['legend'], df['totalKwh'])
    consumption_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 legend_font_color="#fff", title="ENERGY CONSUMPTION",
                                 title_font_color="#fff", title_x=0.45, autosize=True,
                                 annotations=[dict(text=str(round(total_kwh_used,1))+" KWH", x=0.5, y=0.5, showarrow=False)],
                                 legend_title_text='Connections')
    consumption_pie.update_traces(hole=.6, hovertemplate='<b>Customer Ref: %{label}<br>Energy: %{value} kWh</b>')
    consumption_pie.update_annotations(font=dict(color="#fff"))
    consumption_pie = pio.to_html(consumption_pie, full_html=False)


    purchases_pie = create_pie_chart(df['legend'], df['kwhPurchased'])
    purchases_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 legend_font_color="#fff", title="ENERGY PURCHASES",
                                 title_font_color="#fff", title_x=0.45, autosize=True,
                                 annotations=[dict(text=str(round(total_kwh_bought,1))+" KWH", x=0.5, y=0.5, showarrow=False)],
                                 legend_title_text='Connections')
    purchases_pie.update_traces(hole=.6, hovertemplate='<b>Customer Ref: %{label}<br>Energy Purchases: %{value} kWh</b>')
    purchases_pie.update_annotations(font=dict(color="#fff"))
    purchases_pie = pio.to_html(purchases_pie, full_html=False)
    ml_predictions = predict()


    context = {
        "kwhUsed": total_kwh_used,
        "kwhBought": total_kwh_bought,
        "connections": connections,
        "selected_range": str(range_value),
        "consumption_pie": consumption_pie,
        "purchases_pie": purchases_pie,
        "predictions": ml_predictions
    }

    return render(request, 'index.html', context)


def fetch_data_index(endpoint, range_value):
    try:
        response = requests.get(f"{BASE_URL}{endpoint}?range={range_value}", auth=AUTH)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def fetch_data(endpoint):
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", auth=AUTH)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def fetch_data_connection(endpoint, argmnt):
    try:
        response = requests.get(f"{BASE_URL}{endpoint}?{argmnt}", auth=AUTH)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def create_pie_chart(names, values):
    pie_chart = px.pie(names=names, values=values)
    return pie_chart

@login_required
def connections_page(request):
    data = fetch_data("kuduraCustomers")
    data = pd.DataFrame(data)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    # Handle search query
    query = request.GET.get('q')
    if query:
        data = data[data.apply(lambda row: query.lower() in row['name'].lower() or
                                            query.lower() in row['reference'].lower() or
                                            query.lower() in row['meterNumber'].lower(), axis=1)]

    # Convert the DataFrame to a list of dictionaries
    connections_list = data.to_dict(orient='records')

    # Implement pagination with 10 items per page
    paginator = Paginator(connections_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {
        "connections_table": page_obj
    }
    return render(request, 'connections.html', context)

@login_required
def connection_data_page(request, meter_number):
    range_value = request.GET.get('range', 9999999)
    data = fetch_data_connection('kuduraConnections', 'meter='+str(meter_number)+'&range='+str(range_value))
    #data = pd.DataFrame(data)
    reference = data["reference"]
    kwhUsed = data["kwhUsed"]
    kwhBought = data["kwhBought"]
    name = data["name"]
    purchases = data["purchases"][::-1]
    meterData = data["meterData"]
    #purchases_list = purchases.to_dict(orient='records')
    paginator = Paginator(purchases, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    if len(meterData) != 0:
        fig_line = px.line(meterData, x='time', y='kwh', title='Energy Consumption', labels={'time': 'Time', 'kwh': 'kwh'},
                            line_shape='spline')
        fig_line.update_traces(line=dict(color="#fff"), hovertemplate='Time: %{x}<br>kwh: %{y}',  mode='lines+markers')
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    legend_font_color="#fff", title="ENERGY CONSUMPTION",
                                    title_font_color="#fff", title_x=0.45, height=400)
        fig_line.update_layout(xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False))
        fig_line.update_yaxes(title_font_color="#fff") 
        fig_line.update_xaxes(title_font_color="#fff")   
        fig_line.update_xaxes(tickfont_color='#fff')     
        fig_line.update_yaxes(tickfont_color='#fff')                      
        line_chart = pio.to_html(fig_line, full_html=False)
        context = {
            "kwhUsed": kwhUsed,
            "kwhBought": kwhBought,
            "name": name,
            "purchases": page_obj,
            "selected_range": str(range_value),
            "reference": reference,
            "meter_number": meter_number,
            "line_chart": line_chart
        }
    else:
        context = {
            "kwhUsed": kwhUsed,
            "kwhBought": kwhBought,
            "name": name,
            "purchases": page_obj,
            "selected_range": str(range_value),
            "reference": reference,
            "meter_number": meter_number
        }

    return render(request, 'connection_data.html', context)

def predict():
    readable_predictions = []
    predictions = main_pred(file_path_test)
    for pred in predictions:
        readable_pred = {}
        is_known = False
        for i in range(len(pred)):
            readable_pred[appliance_labels[i]] = int(pred[i])
            if pred[i] == 1:
                is_known = True
        if not is_known:
            readable_pred['Unknown'] = 1
        readable_predictions.append(readable_pred)
    return readable_predictions


def predict_new_data_with_anomaly_detection(multi_output_rf, iso_forest, scaler, new_data, appliance_labels):
    new_data_scaled = scaler.transform(new_data)
    anomalies = iso_forest.predict(new_data_scaled)
    results = []
    for idx, (pred, anomaly) in enumerate(zip(multi_output_rf.predict(new_data_scaled), anomalies)):
        if anomaly == -1:
            results.append("Unknown or anomalous appliance detected")
        else:
            on_appliances = [appliance_labels[i] for i, state in enumerate(pred) if state == 1]
            appliances_on = ', '.join(on_appliances) if on_appliances else "No appliances detected"
            results.append(f"ON appliances: {appliances_on}")
    return results


def preprocess_data(df):
    df['energy(kWh)'] = df['energy(kWh)'] * 2
    df = df.drop(columns=['timestamp(DATETIME)', 'time'], errors='ignore')
    df.fillna(df.median(), inplace=True) 
    return df


def main_pred(file_path):    
    new_data_df = pd.read_csv(file_path)
    new_data_df = preprocess_data(new_data_df)
    new_data = new_data_df.values
    predictions = predict_new_data_with_anomaly_detection(multi_output_rf, iso_forest, scaler, new_data, appliance_labels)
    return predictions