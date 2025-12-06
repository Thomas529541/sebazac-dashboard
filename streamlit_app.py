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
# 2. MOTEUR DE DONNÉES (LECTURE DIRECTE CSV)
# =============================================================================
@st.cache_data(ttl=600)
def load_data():
    # Lien de votre Google Sheet
    sheet_id = "1GzL2TZE7X2z7HaO3rgxBbPh8xZfoNw4OxNHBw8YtK5c"
    
    # Construction des URLs d'export CSV pour chaque onglet (gid=0 pour le 1er, etc.)
    # Note: Il faut connaître le GID (ID de l'onglet). Souvent 0 pour le premier.
    # Si vos onglets ont été créés dans l'ordre : 
    # gid=0 -> Probablement "ANALYSE ACTIVITÉS" (le 1er dans votre fichier)
    # gid=12345 -> Probablement "ANALYSE HORAIRE"
    
    # ASTUCE ROBUSTE : On va lire le fichier comme un CSV public
    # Assurez-vous que l'ordre des onglets dans votre Sheet est bien :
    # 1. ANALYSE ACTIVITÉS
    # 2. ANALYSE FAMILLES
    # 3. ANALYSE HORAIRE
    
    # URL pour l'onglet "ANALYSE HORAIRE" (Supposons que c'est le 3ème onglet, GID à vérifier)
    # Si vous ne connaissez pas le GID, ouvrez votre Sheet, cliquez sur l'onglet, et regardez l'URL : "...#gid=123456"
    # Remplacez les GID ci-dessous par les VRAIS GID de votre fichier.
    
    # EXEMPLE (À ADAPTER AVEC VOS VRAIS GID) :
    gid_horaire = #gid=2017923547
    gid_famille = #gid=1480957905
    
    # Si vous ne trouvez pas les GID, mettez 0 pour tester le premier onglet.
    
    url_horaire = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid_horaire}"
    url_famille = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid_famille}"

    try:
        # Lecture directe avec Pandas (plus fiable que st-connection pour les fichiers publics)
        df_h = pd.read_csv(url_horaire)
        df_d = pd.read_csv(url_famille)

        # --- NETTOYAGE HORAIRES ---
        df_h.columns = df_h.columns.str.strip()
        col_mapping_h = {'Période': 'Date', 'Nombre de clients': 'Clients', 'CA TTC': 'CA'}
        df_h = df_h.rename(columns={k: v for k, v in col_mapping_h.items() if k in df_h.columns})
        
        df_h['Date'] = pd.to_datetime(df_h['Date'], dayfirst=True, errors='coerce')
        df_h = df_h.dropna(subset=['Date'])
        df_h['Mois'] = df_h['Date'].dt.strftime('%Y-%m')
        df_h['JourSemaine'] = df_h['Date'].dt.dayofweek
        
        def clean_hour(val):
            try:
                return int(str(val).strip()[:2].replace(':', ''))
            except:
                return 0
        df_h['Heure'] = df_h['Heure'].apply(clean_hour)
        df_h['HeureLabel'] = df_h['Heure'].astype(str) + "h"

        # --- NETTOYAGE FAMILLES ---
        df_d.columns = df_d.columns.str.strip()
        col_mapping_d = {'FAMILLE': 'Famille', 'Période': 'Date', 'CA TTC': 'CA'}
        df_d = df_d.rename(columns={k: v for k, v in col_mapping_d.items() if k in df_d.columns})
        
        df_d['Date'] = pd.to_datetime(df_d['Date'], dayfirst=True, errors='coerce')
        df_d = df_d.dropna(subset=['Date'])
        df_d['Mois'] = df_d['Date'].dt.strftime('%Y-%m')

        return df_h, df_d

    except Exception as e:
        # Fallback en cas d'erreur de GID ou de lecture
        st.warning(f"Mode Démo activé (Erreur lecture Google Sheet : {e}). Vérifiez les GID des onglets.")
        # ... (Génération de fausses données pour ne pas laisser l'écran vide) ...
        # (Je remets le générateur ici pour assurer que l'app marche quoiqu'il arrive)
        start_date = pd.Timestamp('2024-11-01')
        dates = pd.date_range(start=start_date, periods=60, freq='D')
        h_rows = []
        d_rows = []
        for d in dates:
            for h in range(7,21):
                h_rows.append({'Date': d, 'Mois': d.strftime('%Y-%m'), 'JourSemaine': d.dayofweek, 'Heure': h, 'HeureLabel': f"{h}h", 'Clients': 30, 'CA': 450})
            for f in ['Tabac', 'Bar']:
                d_rows.append({'Date': d, 'Mois': d.strftime('%Y-%m'), 'Famille': f, 'CA': 2000})
        return pd.DataFrame(h_rows), pd.DataFrame(d_rows)

df_hourly, df_daily = load_data()

# =============================================================================
# 3. INTERFACE & NAVIGATION
# =============================================================================

if df_hourly.empty:
    st.error("Impossible de charger les données. Vérifiez l'URL Google Sheet.")
    st.stop()

with st.sidebar:
    st.title("🚀 Sébazac 360°")
    page = st.radio("Navigation", ["Vision Globale", "Stratégie Panier", "Staffing"])
    st.markdown("---")
    
    view_mode = st.selectbox("Vue", ["Annuelle", "Mensuelle"])
    selected_month = None
    
    if view_mode == "Mensuelle" and not df_hourly.empty:
        months = sorted(df_hourly['Mois'].unique(), reverse=True)
        selected_month = st.selectbox("Mois", months)

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
    if not data_d.empty:
        fam_agg = data_d.groupby('Famille')['CA'].sum().reset_index()
        c_chart, c_data = st.columns([2, 1])
        with c_chart:
            fig_pie = px.pie(fam_agg, values='CA', names='Famille', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c_data:
            st.dataframe(fam_agg.sort_values('CA', ascending=False).style.format({'CA': '{:,.0f} €'}), hide_index=True, use_container_width=True)
    else:
        st.warning("Pas de données Familles.")

# =============================================================================
# PAGE 2 : PANIER
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
    c1.warning("⚠️ **Matin** : Si le panier est bas mais le flux élevé, optimisez la vitesse.")
    c2.success("✅ **Soir** : Si le panier monte, c'est le moment de vendre plus.")

# =============================================================================
# PAGE 3 : STAFFING
# =============================================================================
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
```
```
