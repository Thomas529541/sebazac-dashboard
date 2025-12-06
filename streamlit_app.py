import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Sébazac 360°", page_icon="📊", layout="wide")

# =============================================================================
# 2. MOTEUR DE DONNÉES (CORRIGÉ)
# =============================================================================
@st.cache_data
def load_data():
    # Simulation des dates
    start_date = pd.Timestamp('2024-11-01')
    end_date = pd.Timestamp('2025-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    hourly_rows = []
    daily_rows = []
    
    for date in dates:
        is_weekend = date.dayofweek >= 5
        
        # --- 1. Simulation Fichier HORAIRES ---
        day_total_ca = 0
        base_flux = 8000 + np.random.randint(-1500, 2000)
        if is_weekend: base_flux += 1500
        
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
                'JourSemaine': date.dayofweek, # AJOUT CRUCIAL ICI (0=Lundi)
                'Heure': h,
                'HeureLabel': f"{h}h",
                'Clients': clients,
                'CA': ca
            })
            day_total_ca += ca
            
        # --- 2. Simulation Fichier FAMILLES ---
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

# =============================================================================
# 3. INTERFACE & NAVIGATION
# =============================================================================

with st.sidebar:
    st.title("🚀 Sébazac 360°")
    page = st.radio("Navigation", ["Vision Globale", "Stratégie Panier", "Staffing"])
    st.markdown("---")
    
    view_mode = st.selectbox("Vue", ["Annuelle", "Mensuelle"])
    selected_month = None
    
    if view_mode == "Mensuelle":
        months = sorted(df_hourly['Mois'].unique(), reverse=True)
        selected_month = st.selectbox("Mois", months)

if view_mode == "Mensuelle":
    data_h = df_hourly[df_hourly['Mois'] == selected_month]
    data_d = df_daily[df_daily['Mois'] == selected_month]
    title_context = f"Mois de {selected_month}"
else:
    data_h = df_hourly
    data_d = df_daily
    title_context = "Année Complète"

# =============================================================================
# PAGE 1 : VISION GLOBALE
# =============================================================================
if page == "Vision Globale":
    st.title(f"📊 Vision Globale - {title_context}")
    
    total_ca = data_h['CA'].sum()
    total_clients = data_h['Clients'].sum()
    panier = total_ca / total_clients if total_clients else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Chiffre d'Affaires", f"{total_ca:,.0f} €")
    c2.metric("Passages Clients", f"{total_clients:,.0f}")
    c3.metric("Panier Moyen", f"{panier:.2f} €")
    
    st.markdown("---")
    
    if view_mode == "Mensuelle":
        agg = data_h.groupby('Date').agg({'CA':'sum', 'Clients':'sum'}).reset_index()
        agg['Label'] = agg['Date'].dt.strftime('%d')
    else:
        agg = data_h.groupby('Mois').agg({'CA':'sum', 'Clients':'sum'}).reset_index()
        agg['Label'] = agg['Mois']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg['Label'], y=agg['CA'], name='CA (€)', marker_color='#6366f1', yaxis='y1'))
    fig.add_trace(go.Scatter(x=agg['Label'], y=agg['Clients'], name='Clients (Nb)', mode='lines+markers', line=dict(color='#10b981', width=3), yaxis='y2'))
    
    fig.update_layout(
        title="Évolution CA vs Clients",
        yaxis=dict(title="CA (€)", side='left'),
        yaxis2=dict(title="Nombre de Clients", side='right', overlaying='y', showgrid=False),
        legend=dict(orientation="h", y=1.1),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Répartition Activités")
    fam_agg = data_d.groupby('Famille')['CA'].sum().reset_index()
    c_chart, c_data = st.columns([2, 1])
    with c_chart:
        fig_pie = px.pie(fam_agg, values='CA', names='Famille', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c_data:
        st.dataframe(fam_agg.sort_values('CA', ascending=False).style.format({'CA': '{:,.0f} €'}), hide_index=True, use_container_width=True)

# =============================================================================
# PAGE 2 : STRATÉGIE PANIER
# =============================================================================
elif page == "Stratégie Panier":
    st.title("🛒 Stratégie Panier & Horaire")
    
    hourly_stats = data_h.groupby('Heure').agg({'CA':'sum', 'Clients':'sum'}).reset_index()
    hourly_stats['Panier'] = hourly_stats['CA'] / hourly_stats['Clients']
    hourly_stats['HeureLabel'] = hourly_stats['Heure'].astype(str) + "h"
    
    st.info("💡 **Analyse Chrono-Rentabilité :** Comparez l'affluence (Gris) avec la dépense moyenne (Violet).")
    
    fig_clock = go.Figure()
    fig_clock.add_trace(go.Bar(x=hourly_stats['HeureLabel'], y=hourly_stats['Clients'], name='Flux Clients', marker_color='#cbd5e1', opacity=0.7))
    fig_clock.add_trace(go.Scatter(x=hourly_stats['HeureLabel'], y=hourly_stats['Panier'], name='Panier Moyen (€)', line=dict(color='#4f46e5', width=4), yaxis='y2'))
    
    fig_clock.update_layout(
        title="Horloge de Rentabilité (Flux vs Panier)",
        yaxis=dict(title="Nombre de Clients", showgrid=False),
        yaxis2=dict(title="Panier Moyen (€)", overlaying='y', side='right'),
        legend=dict(orientation="h", y=1.1),
        height=500
    )
    st.plotly_chart(fig_clock, use_container_width=True)
    
    c1, c2 = st.columns(2)
    c1.warning("⚠️ **Matin (8h-10h)** : Fort trafic mais panier faible. Objectif : Fluidité.")
    c2.success("✅ **Soir (17h-20h)** : Le panier monte avec le flux. C'est l'heure de vendre !")

# =============================================================================
# PAGE 3 : STAFFING (CORRIGÉ AVEC PIVOT)
# =============================================================================
elif page == "Staffing":
    st.title("👥 Matrice de Staffing")
    st.write("Moyenne des clients par créneau pour la période sélectionnée.")
    
    # 1. PIVOT (Transformation en grille 2D robuste)
    # On fait la moyenne des clients pour chaque couple (Jour, Heure)
    heatmap_data = data_h.pivot_table(
        index='JourSemaine', 
        columns='Heure', 
        values='Clients', 
        aggfunc='mean'
    ).fillna(0) # On remplit les trous par 0 pour éviter les bugs
    
    # 2. Mapping des jours pour l'axe Y
    days_map = {0:'Lundi', 1:'Mardi', 2:'Mercredi', 3:'Jeudi', 4:'Vendredi', 5:'Samedi', 6:'Dimanche'}
    y_labels = [days_map.get(i, f"J{i}") for i in heatmap_data.index]
    
    # 3. Création de la Heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h}h" for h in heatmap_data.columns],
        y=y_labels,
        colorscale='Purples',
        hoverongaps=False
    ))
    
    fig_heat.update_layout(
        title=f"Affluence Moyenne ({title_context})",
        xaxis_title="Heure de la journée",
        # On inverse l'ordre pour avoir Lundi en haut
        yaxis=dict(categoryarray=['Dimanche','Samedi','Vendredi','Jeudi','Mercredi','Mardi','Lundi']),
        height=600
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
