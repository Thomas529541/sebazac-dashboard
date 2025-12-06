import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Sébazac 360°", page_icon="📊", layout="wide")

# =============================================================================
# 2. MOTEUR DE DONNÉES (LECTURE FICHIERS LOCAUX)
# =============================================================================
@st.cache_data
def load_data():
    # Noms des fichiers sur GitHub
    FILE_HORAIRE = "horaires.csv"
    FILE_FAMILLE = "familles.csv"

    try:
        # Lecture directe
        df_h = pd.read_csv(FILE_HORAIRE)
        df_d = pd.read_csv(FILE_FAMILLE)

        # --- NETTOYAGE HORAIRES ---
        df_h.columns = df_h.columns.str.strip()
        
        # Mapping des colonnes (Vos noms -> Code)
        col_map_h = {
            'Période': 'Date', 
            'Nombre de clients': 'Clients', 
            'CA TTC': 'CA'
        }
        df_h = df_h.rename(columns={k: v for k, v in col_map_h.items() if k in df_h.columns})
        
        # Conversion Date
        df_h['Date'] = pd.to_datetime(df_h['Date'], errors='coerce') # Gère le format YYYY-MM-DD automatiquement
        df_h = df_h.dropna(subset=['Date'])
        
        df_h['Mois'] = df_h['Date'].dt.strftime('%Y-%m')
        df_h['JourSemaine'] = df_h['Date'].dt.dayofweek
        
        # Nettoyage Heure (Format "07:00 - 08:00")
        def clean_hour(val):
            try:
                s = str(val).strip()
                # Prend les 2 premiers caractères ("07" -> 7)
                return int(s[:2])
            except:
                return 0
        df_h['Heure'] = df_h['Heure'].apply(clean_hour)
        df_h['HeureLabel'] = df_h['Heure'].astype(str) + "h"

        # --- NETTOYAGE FAMILLES ---
        df_d.columns = df_d.columns.str.strip()
        col_map_d = {
            'FAMILLE': 'Famille', 
            'Période': 'Date', 
            'CA TTC': 'CA'
        }
        df_d = df_d.rename(columns={k: v for k, v in col_map_d.items() if k in df_d.columns})
        
        df_d['Date'] = pd.to_datetime(df_d['Date'], errors='coerce')
        df_d = df_d.dropna(subset=['Date'])
        df_d['Mois'] = df_d['Date'].dt.strftime('%Y-%m')

        return df_h, df_d

    except FileNotFoundError as e:
        st.error(f"❌ Fichier introuvable : {e.filename}")
        st.info("Vérifiez que vous avez bien uploadé 'horaires.csv' et 'familles.csv' sur GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur de lecture : {e}")
        st.stop()

df_hourly, df_daily = load_data()

# =============================================================================
# 3. INTERFACE
# =============================================================================

if df_hourly.empty:
    st.warning("Données vides.")
    st.stop()

with st.sidebar:
    st.title("🚀 Sébazac 360°")
    page = st.radio("Navigation", ["Vision Globale", "Stratégie Panier", "Staffing"])
    st.markdown("---")
    
    view_mode = st.selectbox("Vue", ["Annuelle", "Mensuelle"])
    selected_month = None
    
    if view_mode == "Mensuelle" and not df_hourly.empty:
        months = sorted(df_hourly['Mois'].unique(), reverse=True)
        if months:
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

# --- PAGE 1 : VISION GLOBALE ---
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
    if not data_d.empty:
        fam_agg = data_d.groupby('Famille')['CA'].sum().reset_index()
        c_chart, c_data = st.columns([2, 1])
        with c_chart:
            fig_pie = px.pie(fam_agg, values='CA', names='Famille', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c_data:
            st.dataframe(fam_agg.sort_values('CA', ascending=False).style.format({'CA': '{:,.0f} €'}), hide_index=True, use_container_width=True)
    else:
        st.warning("Pas de données Familles disponibles pour cette vue.")

# --- PAGE 2 : PANIER ---
elif page == "Stratégie Panier":
    st.title("🛒 Stratégie Panier & Horaire")
    
    hourly_stats = data_h.groupby('Heure').agg({'CA':'sum', 'Clients':'sum'}).reset_index()
    hourly_stats['Panier'] = hourly_stats['CA'] / hourly_stats['Clients']
    hourly_stats['HeureLabel'] = hourly_stats['Heure'].astype(str) + "h"
    
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

# --- PAGE 3 : STAFFING ---
elif page == "Staffing":
    st.title("👥 Matrice de Staffing")
    st.write("Moyenne des clients par créneau pour la période sélectionnée.")
    
    if not data_h.empty:
        pivot = data_h.groupby(['JourSemaine', 'Heure'])['Clients'].mean().reset_index()
        days_map = {0:'Lundi', 1:'Mardi', 2:'Mercredi', 3:'Jeudi', 4:'Vendredi', 5:'Samedi', 6:'Dimanche'}
        pivot['JourLabel'] = pivot['JourSemaine'].map(days_map)
        
        matrix = pivot.pivot(index='JourLabel', columns='Heure', values='Clients').fillna(0)
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
    else:
        st.warning("Pas assez de données pour générer la matrice.")
