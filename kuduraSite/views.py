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
from django.http import JsonResponse
import json
from .preprocessing import preprocess_live_data
from .data_ingestion import chunk_array
from keras.models import load_model
import numpy as np
from collections import defaultdict
from db.models import userProfile

# Define paths to the model and scaler files
#iso_forest_path = os.path.join(settings.BASE_DIR, 'models', 'iso_forest.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'models', 'final_scalerT.pkl')
#multi_output_rf_path = os.path.join(settings.BASE_DIR, 'models', 'multi_output_rf.pkl')
file_path_test = os.path.join(settings.BASE_DIR, 'models', 'synthetic_test.csv')
model_path = os.path.join(settings.BASE_DIR, 'models', 'final_modelT.h5')
# Load model and scaler
#multi_output_rf = joblib.load(multi_output_rf_path)
scaler = joblib.load(scaler_path)
#iso_forest = joblib.load(iso_forest_path)
model = load_model(model_path)

# Constants
BASE_URL = "http://5.22.218.175:1880/"
AUTH = HTTPBasicAuth('kudura', 'pw4kudura')
appliance_columns = [
    'Electric_Mill',
    'Freezer',
    'Electric_Pressure_Cooker',
    'Other_Appliances'
]

def logout_page(request):
    logout(request)
    return redirect('login')


def login_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            user_profile, created = userProfile.objects.get_or_create(user=user)
            # If T&C not accepted, show popup (but don't log in yet)
            if not user_profile.tnc_flag:
                return render(request, 'partials/tnc_popup.html', {'username': username, 'p': password})

            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')

def accept_tnc(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)  # Parse JSON body
            username = data.get('username')
            password = data.get('p')

            user = authenticate(username=username, password=password)
            if user:
                user_profile, created = userProfile.objects.get_or_create(user=user)
                user_profile.tnc_flag = True
                user_profile.save()

                login(request, user)
                return JsonResponse({'success': True, 'redirect_url': '/'})  # Redirect to home_page

            return JsonResponse({'error': 'User not found'}, status=404)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)


def terms_of_service(request):
    return render(request, 'terms_of_service.html')

@login_required
def profile_view(request):
    return render(request, 'profile.html')

@login_required
def edit_profile(request):
    profile, created = userProfile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        profile.org_name = request.POST.get("org_name", "")
        profile.org_address = request.POST.get("org_address", "")
        profile.org_phone_number = request.POST.get("org_phone_number", "")
        profile.org_email = request.POST.get("org_email", "")
        profile.save()
        return redirect("profile")  # Redirect back to profile page

    return render(request, "my_profile.html", {"profile": profile})


@login_required
def homepage(request):
    #range_value = request.GET.get('range', 9999999)
    range_value = request.GET.get('range', 5)
    data = fetch_data_index('kuduraLiveData', range_value)

    if not data:
        messages.warning(request, 'No data for the selected range. Showing default data.')
        data = fetch_data_index('kuduraLiveData', 9999999)
        range_value = 9999999

    #df = pd.DataFrame(data)
    #df['meterData'] = pd.DataFrame(data["meterData"])
    meterData = pd.DataFrame(data["meterData"])
    customerData = pd.DataFrame(data["customerData"])
    total_kwh_used = meterData["energy(kWh)"].sum()
    connections = len(customerData)
    #total_kwh_bought = df['kwhPurchased'].sum()
    # Merge customerData and meterData on 'meterNumber' to align the data
    merged_data = pd.merge(meterData, customerData[['meterNumber', 'accountID']], on='meterNumber', how='left')

# Create the legend field in the merged dataframe
    merged_data['legend'] = merged_data["accountID"] + " (" + merged_data['meterNumber'] + ")"

# Prepare inputs for pie chart
    labels = merged_data['legend']
    values = merged_data["energy(kWh)"]

# Create the pie chart
    consumption_pie = create_pie_chart(labels, values)
    consumption_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 legend_font_color="#fff", title="ENERGY CONSUMPTION",
                                 title_font_color="#fff", title_x=0.45, autosize=True,
                                 annotations=[dict(text=str(round(total_kwh_used,4))+" KWH", x=0.5, y=0.5, showarrow=False)],
                                 legend_title_text='Connections')
    consumption_pie.update_traces(hole=.6, hovertemplate='<b>Customer Ref: %{label}<br>Energy: %{value} kWh</b>')
    consumption_pie.update_annotations(font=dict(color="#fff"))
    consumption_pie = pio.to_html(consumption_pie, full_html=False)

    appliance_percentages, appliance_energy, ml_pie = prepare_meterData(meterData)
    context = {
        "kwhUsed": total_kwh_used,
        "connections": connections,
        "selected_range": str(range_value),
        "consumption_pie": consumption_pie,
        "ml_pie": ml_pie,
        "appliance_energy": appliance_energy,
        "appliance_percentages": appliance_percentages
    }

    return render(request, "index.html", context)


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
    query = request.GET.get('q')
    if query:
        data = data[data.apply(lambda row: query.lower() in row['name'].lower() or
                                            query.lower() in row['accountID'].lower() or
                                            query.lower() in row['meterNumber'].lower(), axis=1)]

    connections_list = data.to_dict(orient='records')

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
    customerData = data["customerData"]
    meterData = pd.DataFrame(data["meterData"])
    accountID = customerData["accountID"]
    name = customerData["name"]
    kwhUsed = 0.0
    if len(meterData) != 0:
        kwhUsed = meterData["energy(kWh)"].sum()
        meterData["timestamp(DATETIME)"] = pd.to_datetime(meterData["timestamp(DATETIME)"], format="%m/%d/%Y %H:%M:%S", errors='coerce')  # <-- This skips invalid timestamps by turning them into NaT
        # Drop rows with invalid timestamps
        meterData = meterData.dropna(subset=["timestamp(DATETIME)"])
        meterData = meterData[meterData["energy(kWh)"] >= 0]
        meterData['day'] = meterData["timestamp(DATETIME)"].dt.date
        meterData_daily = meterData.groupby("day")["energy(kWh)"].sum().reset_index()
        meterData_daily['day'] = pd.to_datetime(meterData_daily['day'])
        
        fig_line = px.line(meterData_daily, x='day', y='energy(kWh)', title='Energy Consumption', labels={'time': 'Time', 'energy(kWh)': 'kwh'},
                            line_shape='spline')
        fig_line.update_traces(line=dict(color="#fff"), hovertemplate='Date: %{x|%Y-%m-%d}<br>Energy: %{y:.2f} kWh',  mode='lines+markers', fill='tozeroy', fillcolor='#495057')
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    legend_font_color="#fff", title="ENERGY CONSUMPTION",
                                    title_font_color="#fff", title_x=0.45, height=400)
        fig_line.update_layout(xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
        fig_line.update_yaxes(title_font_color="#fff") 
        fig_line.update_xaxes(title_font_color="#fff")   
        fig_line.update_xaxes(tickfont_color='#fff')     
        fig_line.update_yaxes(tickfont_color='#fff')                      
        line_chart = pio.to_html(fig_line, full_html=False)
        appliance_percentages, appliance_energy, ml_pie = prepare_meterData(meterData)
        context = {
            "kwhUsed": kwhUsed,
            "name": name,
            "selected_range": str(range_value),
            "accountID": accountID,
            "meter_number": meter_number,
            "line_chart": line_chart,
            "ml_pie": ml_pie,
            "appliance_energy": appliance_energy,
            "appliance_percentages": appliance_percentages
        }
    else:
        context = {
            "kwhUsed": kwhUsed,
            "name": name,
            "selected_range": str(range_value),
            "accountID": accountID,
            "meter_number": meter_number
        }

    return render(request, 'connection_data.html', context)


def predict_appliance_state(new_data, new_timestamp):
        # Preprocess new data
    X_new_scaled = preprocess_live_data(new_data, new_timestamp)
    data_chunks = chunk_array(X_new_scaled)
    predictions = []
    for chunk in data_chunks:
        if len(chunk) < 600:
        # Pad the chunk to 600 timesteps with zeros (or other desired values)
            padding = np.zeros((600 - len(chunk), chunk.shape[1]))
            chunk = np.vstack((chunk, padding))
        input_sequence = np.array(chunk).reshape(1, 600, -1)
        y_pred_proba = model.predict(input_sequence)
        y_pred = (y_pred_proba >= 0.5).astype(int)  # Thresholded binary predictions
        # Map predictions to appliances
        prediction_result = dict(zip(appliance_columns, y_pred[0]))
        predictions.append(prediction_result)
    return predictions

def prepare_meterData(meterData):
    total_kwh_used = meterData["energy(kWh)"].sum()
    df_renamed = meterData.rename(columns={
    "timestamp(DATETIME)": "timestamp",
    "time": "time",
    "voltage(V)": "voltage(V)",
    "power(kW)": "Power(kW)",
    "powerFactor": "Power_Factor(PF)",
    "current(A)": "Current(A)",
    "energy(kWh)": "Energy(kWh)"
     })
        # Calculate Active Power (kW)
    df_renamed["Power(kW)"] = df_renamed["voltage(V)"] * df_renamed["Current(A)"] * df_renamed["Power_Factor(PF)"] / 1000

    # Calculate Apparent Power (kVA)
    df_renamed["Apparent_Power(kVA)"] = df_renamed["voltage(V)"] * df_renamed["Current(A)"] / 1000

    # Calculate Reactive Power (kVAr)
    df_renamed["Reactive_Power(kVAr)"] = np.sqrt(df_renamed["Apparent_Power(kVA)"]**2 - df_renamed["Power(kW)"]**2)

    #new_dt = pd.read_csv(file_path_test)
    new_dt = df_renamed
    new_ts = datetime.now()
    appliance_counts = defaultdict(int, {appliance: 0 for appliance in appliance_columns})
    appliance_energy = defaultdict(float, {appliance: 0.0 for appliance in appliance_columns})
    #print("Total rows in df_renamed:", len(new_dt))
    for _,row in new_dt.iterrows():
        new_data = row.to_dict()
        energy_value = new_data.get("Energy(kWh)", 0)
        predictions = predict_appliance_state(new_data, new_ts)
        #print("Predicted Appliance States:")
        for idx, pred in enumerate(predictions, start=1):
            #print(f"Chunk {idx}: {pred}")
            for appliance, state in pred.items():
                if state == 1:  # Appliance is ON
                    appliance_counts[appliance] += 1
                    appliance_energy[appliance] += energy_value

    pred_dict = dict(appliance_counts)
    count_dict = {}

    # Iterate through each item in the array
    for key, value in pred_dict.items():
        count_dict[key] = (value * 2) / 3600  # Convert from 2-second intervals to hours

    ml_pie = create_pie_chart(list(count_dict.keys()), list(count_dict.values()))
    ml_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 legend_font_color="#fff", title="METER DISAGREGATION",
                                 annotations=[dict(text=str(round(sum(list(count_dict.values())), 4))+" HRS", x=0.5, y=0.5, showarrow=False)],
                                 title_font_color="#fff", title_x=0.45, autosize=True,
                                 legend_title_text='Appliances')
    ml_pie.update_traces(hole=.6, hovertemplate='<b>Appliance: %{label}<br>Time: %{value} hours</b>')
    ml_pie.update_annotations(font=dict(color="#fff"))
    ml_pie = pio.to_html(ml_pie, full_html=False)
    appliance_percentages = {}
    #total_kwh_used = 100.0
    #appliance_energy = {'Electric_Mill': 10.0, 'Freezer': 20.0, 'Electric_Pressure_Cooker': 30.0, 'Other_Appliances': 40.0}
    for appliance, energy in appliance_energy.items():
        if total_kwh_used > 0:
            appliance_percentages[appliance] = (energy / total_kwh_used) * 100
        else:
            appliance_percentages[appliance] = 0  # Handle case where kWhUsed is 0
    return appliance_percentages, dict(appliance_energy), ml_pie



