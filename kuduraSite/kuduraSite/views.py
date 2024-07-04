from django.shortcuts import render
import requests
from django.contrib import messages
import pandas as pd
from requests.auth import HTTPBasicAuth
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Constants
BASE_URL = "http://5.22.218.175:1880/"
AUTH = HTTPBasicAuth('kudura', 'pw4kudura')


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
                                 annotations=[dict(text=str(round(total_kwh_used,1))+" KWH", x=0.5, y=0.5, font_size=20, showarrow=False)])
    consumption_pie.update_traces(hole=.6, hovertemplate='<b>Customer Ref: %{label}<br>Energy: %{value} kWh</b>')
    consumption_pie.update_annotations(font=dict(size=18, color="#fff"))
    consumption_pie = pio.to_html(consumption_pie, full_html=False)


    purchases_pie = create_pie_chart(df['legend'], df['kwhPurchased'])
    purchases_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 legend_font_color="#fff", title="ENERGY PURCHASES",
                                 title_font_color="#fff", title_x=0.45, autosize=True,
                                 annotations=[dict(text=str(round(total_kwh_bought,1))+" KWH", x=0.5, y=0.5, font_size=20, showarrow=False)])
    purchases_pie.update_traces(hole=.6, hovertemplate='<b>Customer Ref: %{label}<br>Energy Purchases: %{value} kWh</b>')
    purchases_pie.update_annotations(font=dict(size=18, color="#fff"))
    purchases_pie = pio.to_html(purchases_pie, full_html=False)


    context = {
        "kwhUsed": total_kwh_used,
        "kwhBought": total_kwh_bought,
        "connections": connections,
        "selected_range": str(range_value),
        "consumption_pie": consumption_pie,
        "purchases_pie": purchases_pie
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

def create_pie_chart(names, values):
    pie_chart = px.pie(names=names, values=values)
    return pie_chart
