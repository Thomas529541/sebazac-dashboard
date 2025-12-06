import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Sébazac 360°", page_icon="📊", layout="wide")

# --- MOTEUR DE DONNÉES ---
@st.cache_data
def load_data():
    # Simulation des données basées sur la structure de vos fichiers
    start_date = pd.Timestamp('2024-11-01')
    end_date = pd.Timestamp('2025-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    hourly_rows = []
    daily_rows = []
    
    for date in dates:
        is_weekend = date.dayofweek >= 5
        base_flux = 8000 + np.random.randint(-1500, 2000)
        if is_weekend: base_flux += 1500
        
        # Génération Horaire
        day_total_ca = 0
        for h in range(7, 21):
            w = 1.0
            if h == 8: w = 2.2 
            elif h == 12: w = 1.6 
            elif h == 18: w = 2.4 
            
            clients = int((base_flux / 2800) * w * np.random.uniform(0.8, 1.2))
            avg_basket = 12 if h <= 10 else (21 if h >= 17 else 15)
            ca = clients * avg_basket * np.random.uniform(0.9, 1.1)
            
            hourly_rows.append({
                'Date': date,
                'Mois': date.strftime('%Y-%m'),
                'Heure': h,
                'HeureLabel': f"{h}h",
                'Clients': clients,
                'CA': ca
            })
            day_total_ca += ca
            
        # Génération Familles (Journalier)
        mix = {'Tabac': 0.55, 'Jeux': 0.30, 'Bar': 0.10, 'Presse': 0.05}
        if is_weekend: mix = {'Tabac': 0.45, 'Jeux': 0.30, 'Bar': 0.20, 'Presse': 0.05}
        
        for fam, share in mix.items():
            daily_rows.append({
                'Date': date,
                'Mois': date.strftime('%Y-%m'),
                'Famille': fam,
                'CA': day_total_ca * share
            })
            
    return pd.DataFrame(hourly_rows), pd.DataFrame(daily_rows)

df_hourly, df_daily = load_data()

# --- INTERFACE ---
with st.sidebar:
    st.title("🚀 Sébazac 360°")
    page = st.radio("Navigation", ["Vision Globale", "Stratégie Panier", "Staffing"])
    st.markdown("---")
    view_mode = st.selectbox("Période", ["Annuelle", "Mensuelle"])
    selected_month = None
    if view_mode == "Mensuelle":
        months = sorted(df_hourly['Mois'].unique(), reverse=True)
        selected_month = st.selectbox("Mois", months)

# Filtrage
if view_mode == "Mensuelle":
    data_h = df_hourly[df_hourly['Mois'] == selected_month]
    data_d = df_daily[df_daily['Mois'] == selected_month]
    x_label = 'Date'
else:
    data_h = df_hourly
    data_d = df_daily
    x_label = 'Mois'

# --- PAGE 1 : VISION GLOBALE ---
if page == "Vision Globale":
    st.title("📊 Vision Globale")
    
    # KPIs
    total_ca = data_h['CA'].sum()
    total_clients = data_h['Clients'].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Chiffre d'Affaires", f"{total_ca:,.0f} €")
    c2.metric("Clients", f"{total_clients:,.0f}")
    c3.metric("Panier Moyen", f"{(total_ca/total_clients):.2f} €")
    
    st.markdown("---")
    
    # Graphique Principal
    if view_mode == "Mensuelle":
        agg = data_h.groupby('Date').agg({'CA':'sum', 'Clients':'sum'}).reset_index()
        agg['Label'] = agg['Date'].dt.strftime('%d')
    else:
        agg = data_h.groupby('Mois').agg({'CA':'sum', 'Clients':'sum'}).reset_index()
        agg['Label'] = agg['Mois']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg['Label'], y=agg['CA'], name='CA (€)', marker_color='#6366f1', yaxis='y1'))
    fig.add_trace(go.Scatter(x=agg['Label'], y=agg['Clients'], name='Clients', line=dict(color='#10b981', width=3), yaxis='y2'))
    
    fig.update_layout(
        yaxis=dict(title="CA (€)", side='left'),
        yaxis2=dict(title="Clients", side='right', overlaying='y', showgrid=False),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Camembert Familles
    st.subheader("Répartition Activités")
    fam_agg = data_d.groupby('Famille')['CA'].sum().reset_index()
    fig_pie = px.pie(fam_agg, values='CA', names='Famille', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- PAGE 2 : PANIER ---
elif page == "Stratégie Panier":
    st.title("🛒 Stratégie Panier & Horaire")
    
    hourly_stats = data_h.groupby('Heure').agg({'CA':'sum', 'Clients':'sum'}).reset_index()
    hourly_stats['Panier'] = hourly_stats['CA'] / hourly_stats['Clients']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly_stats['Heure'], y=hourly_stats['Clients'], name='Flux', marker_color='#cbd5e1'))
    fig.add_trace(go.Scatter(x=hourly_stats['Heure'], y=hourly_stats['Panier'], name='Panier (€)', line=dict(color='#4f46e5', width=4), yaxis='y2'))
    
    fig.update_layout(
        title="Horloge de Rentabilité (Flux vs Panier)",
        yaxis=dict(title="Clients", showgrid=False),
        yaxis2=dict(title="Panier (€)", overlaying='y', side='right'),
        xaxis=dict(tickmode='linear', dtick=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 3 : STAFFING ---
elif page == "Staffing":
    st.title("👥 Matrice de Staffing")
    
    pivot = data_h.groupby(['JourSemaine', 'Heure'])['Clients'].mean().reset_index()
    days = {0:'Lun', 1:'Mar', 2:'Mer', 3:'Jeu', 4:'Ven', 5:'Sam', 6:'Dim'}
    pivot['Jour'] = pivot['JourSemaine'].map(days)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot['Clients'], x=pivot['Heure'], y=pivot['Jour'],
        colorscale='Purples'
    ))
    fig.update_layout(yaxis=dict(categoryarray=['Dim','Sam','Ven','Jeu','Mer','Mar','Lun']))
    st.plotly_chart(fig, use_container_width=True)
