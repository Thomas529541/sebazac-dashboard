import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_gsheets import GSheetsConnection

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Sébazac 360°", page_icon="📊", layout="wide")

# =============================================================================
# 2. MOTEUR DE DONNÉES (GOOGLE SHEETS)
# =============================================================================
@st.cache_data(ttl=600) # Mise en cache 10 minutes pour éviter de trop appeler Google
def load_data():
    # Création de la connexion
    conn = st.connection("gsheets", type=GSheetsConnection)

    try:
        # 1. Lecture Onglet HORAIRES (Worksheet 0 ou par nom)
        # Note: Assurez-vous que vos onglets s'appellent exactement comme ça dans le Sheet
        df_h = conn.read(worksheet="ANALYSE HORAIRE")
        
        # 2. Lecture Onglet FAMILLES
        df_d = conn.read(worksheet="ANALYSE FAMILLES")
        
        # --- NETTOYAGE HORAIRES ---
        # Renommage (Adapter si vos colonnes Google Sheet diffèrent)
        df_h = df_h.rename(columns={
            'Période': 'Date', 
            'Nombre de clients': 'Clients', 
            'CA TTC': 'CA'
        })
        
        # Conversion Types
        df_h['Date'] = pd.to_datetime(df_h['Date'], dayfirst=True)
        df_h['Mois'] = df_h['Date'].dt.strftime('%Y-%m')
        df_h['JourSemaine'] = df_h['Date'].dt.dayofweek
        
        # Nettoyage Heure (ex: "07:00 - 08:00" -> 7)
        def clean_hour(h):
            try:
                return int(str(h)[:2])
            except:
                return 0
        
        df_h['Heure'] = df_h['Heure'].apply(clean_hour)
        df_h['HeureLabel'] = df_h['Heure'].astype(str) + "h"

        # --- NETTOYAGE FAMILLES ---
        df_d = df_d.rename(columns={
            'FAMILLE': 'Famille',
            'Période': 'Date',
            'CA TTC': 'CA'
        })
        df_d['Date'] = pd.to_datetime(df_d['Date'], dayfirst=True)
        df_d['Mois'] = df_d['Date'].dt.strftime('%Y-%m')

        return df_h, df_d

    except Exception as e:
        st.error(f"Erreur de connexion Google Sheets : {e}")
        st.stop()

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
    
    # Sélecteur dynamique
    if view_mode == "Mensuelle":
        if not df_hourly.empty:
            months = sorted(df_hourly['Mois'].unique(), reverse=True)
            selected_month = st.selectbox("Mois", months)

# Filtrage dynamique
if view_mode == "Mensuelle" and selected_month:
    data_h = df_hourly[df_hourly['Mois'] == selected_month]
    data_d = df_daily[df_daily['Mois'] == selected_month]
    title_context = f"Mois de {selected_month}"
else:
    data_h = df_hourly
    data_d = df_daily
    title_context = "Année Complète"

# =============================================================================
# PAGE 1 : VISION GLOBALE (CEO)
# =============================================================================
if page == "Vision Globale":
    st.title(f"📊 Vision Globale - {title_context}")
    
    # KPIs
    total_ca = data_h['CA'].sum()
    total_clients = data_h['Clients'].sum()
    panier = total_ca / total_clients if total_clients else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Chiffre d'Affaires", f"{total_ca:,.0f} €")
    c2.metric("Passages Clients", f"{total_clients:,.0f}")
    c3.metric("Panier Moyen", f"{panier:.2f} €")
    
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
    fig.add_trace(go.Scatter(x=agg['Label'], y=agg['Clients'], name='Clients (Nb)', mode='lines+markers', line=dict(color='#10b981', width=3), yaxis='y2'))
    
    fig.update_layout(
        title="Évolution CA vs Clients",
        yaxis=dict(title="CA (€)", side='left'),
        yaxis2=dict(title="Nombre de Clients", side='right', overlaying='y', showgrid=False),
        legend=dict(orientation="h", y=1.1),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Camembert Familles
    st.subheader("Répartition Activités")
    fam_agg = data_d.groupby('Famille')['CA'].sum().reset_index()
    
    c_chart, c_data = st.columns([2, 1])
    with c_chart:
        fig_pie = px.pie(fam_agg, values='CA', names='Famille', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c_data:
        st.dataframe(
            fam_agg.sort_values('CA', ascending=False).style.format({'CA': '{:,.0f} €'}),
            hide_index=True, 
            use_container_width=True
        )

# =============================================================================
# PAGE 2 : PANIER
# =============================================================================
elif page == "Stratégie Panier":
    st.title("🛒 Stratégie Panier & Horaire")
    
    # Agrégation Horaire
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

# =============================================================================
# PAGE 3 : STAFFING
# =============================================================================
elif page == "Staffing":
    st.title("👥 Matrice de Staffing")
    st.write("Moyenne des clients par créneau pour la période sélectionnée.")
    
    # Pivot pour Heatmap
    pivot = data_h.groupby(['JourSemaine', 'Heure'])['Clients'].mean().reset_index()
    
    days_map = {0:'Lundi', 1:'Mardi', 2:'Mercredi', 3:'Jeudi', 4:'Vendredi', 5:'Samedi', 6:'Dimanche'}
    pivot['JourLabel'] = pivot['JourSemaine'].map(days_map)
    
    # Création Matrice Carrée (Pivot Table)
    matrix = pivot.pivot(index='JourLabel', columns='Heure', values='Clients').fillna(0)
    # Réordonner les jours
    order = ['Dimanche','Samedi','Vendredi','Jeudi','Mercredi','Mardi','Lundi']
    matrix = matrix.reindex(order)
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=[f"{h}h" for h in matrix.columns],
        y=matrix.index,
        colorscale='Purples',
        hoverongaps=False
    ))
    
    fig_heat.update_layout(
        title=f"Affluence Moyenne ({title_context})",
        xaxis_title="Heure de la journée",
        height=600
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
