import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURATION PAGE & CSS (Correction Hauteur) ---
st.set_page_config(page_title="Cockpit Commerce", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    /* 1. Remonter tout le contenu vers le haut */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0rem !important;
    }
    
    /* Style KPI */
    .kpi-box {
        background-color: #262730; border-radius: 8px; padding: 15px;
        text-align: center; border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 10px;
    }
    .kpi-label { font-size: 14px; color: #aaaaaa; margin: 0; }
    .kpi-val { font-size: 26px; font-weight: bold; color: white; margin: 5px 0; }
    .kpi-delta { font-size: 13px; }
    
    /* Couleurs variations */
    .pos { color: #00FF00; font-weight: bold; }
    .neg { color: #FF4444; font-weight: bold; }
    .neu { color: #888888; }
    
    /* Style Résumé Cascade */
    .bridge-summary {
        font-size: 16px; font-weight: bold; text-align: center; 
        margin-bottom: 5px; padding: 8px; background-color: #31333F; 
        border-radius: 8px; border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. CHARGEMENT ET NETTOYAGE ---
@st.cache_data
def load_data():
    try:
        fichier = 'donnees.xlsx'
        df_h = pd.read_excel(fichier, sheet_name='ANALYSE HORAIRE')
        df_a = pd.read_excel(fichier, sheet_name='Analyse Activités')

        def clean_curr(s):
            if s.dtype == 'object':
                return s.astype(str).str.replace('€', '').str.replace(' ', '').str.replace(',', '.').astype(float)
            return s

        # Nettoyage
        cols_h = {'CA TTC': clean_curr, 'Nombre de clients': pd.to_numeric}
        for col, func in cols_h.items():
            if col in df_h.columns: df_h[col] = func(df_h[col])
        
        # DATE : On s'assure d'avoir des dates propres
        if 'Période' in df_h.columns: 
            df_h['DateFull'] = pd.to_datetime(df_h['Période'], dayfirst=True)
            # CORRECTION CRITIQUE : On crée une colonne "Jour" pur (sans heure minute seconde)
            df_h['Date'] = df_h['DateFull'].dt.normalize() 

        cols_a = {'CA TTC': clean_curr, 'Quantité': pd.to_numeric}
        for col, func in cols_a.items():
            if col in df_a.columns: df_a[col] = func(df_a[col])
        if 'Période' in df_a.columns: 
            df_a['DateFull'] = pd.to_datetime(df_a['Période'], dayfirst=True)
            df_a['Date'] = df_a['DateFull'].dt.normalize()

        return df_h, df_a
    except Exception as e:
        st.error(f"Erreur critique : {e}")
        st.stop()

df_horaire, df_activite = load_data()

# --- FONCTION DE CONSOLIDATION (La clé du correctif) ---
def consolidate_registers(df):
    """
    Fusionne les lignes qui ont la même Date (Jour) et la même Tranche Horaire (Heure).
    C'est ça qui corrige le bug des -50%.
    """
    if df.empty: return df
    # On groupe par JOUR (sans l'heure exacte) et par CRÉNEAU (ex: '07:00 - 08:00')
    # On somme les CA et Clients des 2 caisses
    df_conso = df.groupby(['Date', 'Heure']).sum(numeric_only=True).reset_index()
    return df_conso

# --- 2. FILTRES ---
with st.sidebar:
    st.header("⚡ Pilotage")
    annees = sorted(df_horaire['Date'].dt.year.unique(), reverse=True)
    annee_sel = st.selectbox("Année", annees)
    
    mois_dispo = sorted(df_horaire[df_horaire['Date'].dt.year == annee_sel]['Date'].dt.month.unique(), reverse=True)
    noms_mois = {1:'Jan', 2:'Fév', 3:'Mars', 4:'Avr', 5:'Mai', 6:'Juin', 7:'Juil', 8:'Août', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Déc'}
    mois_sel = st.selectbox("Mois", mois_dispo, format_func=lambda x: noms_mois[x])
    
    st.markdown("---")
    waterfall_comp = st.radio("Cascade vs :", ["Mois Précédent (M-1)", "Année Précédente (N-1)"])

# --- PRÉPARATION DES DONNÉES ---
# Fonction pour extraire et consolider une période donnée
def get_period_data(df, year, month):
    mask = (df['Date'].dt.year == year) & (df['Date'].dt.month == month)
    raw_data = df[mask].copy()
    # On consolide TOUT DE SUITE pour ne plus traîner de doublons caisses
    return consolidate_registers(raw_data)

# Données Actuelles (Consolidées)
df_curr = get_period_data(df_horaire, annee_sel, mois_sel)
df_act_curr = df_activite[(df_activite['Date'].dt.year == annee_sel) & (df_activite['Date'].dt.month == mois_sel)]

# Calcul dates précédentes
if mois_sel == 1: prev_m, prev_y_m = 12, annee_sel - 1
else: prev_m, prev_y_m = mois_sel - 1, annee_sel

# Données M-1 et N-1 (Consolidées)
df_m1 = get_period_data(df_horaire, prev_y_m, prev_m)
df_n1 = get_period_data(df_horaire, annee_sel - 1, mois_sel)

# --- 3. KPIs ---
def calc_evo(curr, prev):
    if prev == 0 or pd.isna(prev): return 0, "neu", "="
    val = ((curr - prev) / prev) * 100
    if val > 0: return val, "pos", "▲"
    elif val < 0: return val, "neg", "▼"
    return val, "neu", "="

def kpi_compact(title, val, val_m1, val_n1, unit=""):
    evo_m, class_m, sym_m = calc_evo(val, val_m1)
    evo_n, class_n, sym_n = calc_evo(val, val_n1)
    val_fmt = f"{val:,.0f} {unit}".replace(",", " ")
    if unit == "€" and val < 100: val_fmt = f"{val:.2f} {unit}"
    
    st.markdown(f"""
    <div class="kpi-box">
        <p class="kpi-label">{title}</p>
        <p class="kpi-val">{val_fmt}</p>
        <div class="kpi-delta">
            <span style="color:#aaa;">vs M-1:</span> <span class="{class_m}">{sym_m} {abs(evo_m):.1f}%</span>
            &nbsp;|&nbsp;
            <span style="color:#aaa;">vs N-1:</span> <span class="{class_n}">{sym_n} {abs(evo_n):.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Calculs sur données consolidées
ca_c, cli_c = df_curr['CA TTC'].sum(), df_curr['Nombre de clients'].sum()
pm_c = ca_c/cli_c if cli_c else 0

ca_m1, cli_m1 = df_m1['CA TTC'].sum(), df_m1['Nombre de clients'].sum()
pm_m1 = ca_m1/cli_m1 if cli_m1 else 0

ca_n1, cli_n1 = df_n1['CA TTC'].sum(), df_n1['Nombre de clients'].sum()
pm_n1 = ca_n1/cli_n1 if cli_n1 else 0

# Affichage Ligne 1
k1, k2, k3 = st.columns(3)
with k1: kpi_compact("Chiffre d'Affaires", ca_c, ca_m1, ca_n1, "€")
with k2: kpi_compact("Fréquentation", cli_c, cli_m1, cli_n1, "")
with k3: kpi_compact("Panier Moyen", pm_c, pm_m1, pm_n1, "€")

st.markdown("---")

# --- 4. ANALYSES ---
c_left, c_right = st.columns([1, 2])

# A. DONUTS
with c_left:
    st.subheader("🌞 Matin vs Soir")
    if not df_curr.empty:
        # On recrée l'heure int à partir de la colonne consolidée 'Heure'
        df_curr['Heure_Int'] = df_curr['Heure'].astype(str).str.slice(0, 2).astype(int)
        df_curr['Moment'] = df_curr['Heure_Int'].apply(lambda x: 'Matin' if x < 13 else 'Après-Midi')
        stats = df_curr.groupby('Moment').agg({'CA TTC': 'sum', 'Nombre de clients': 'sum'}).reset_index()
        
        fig_ca = px.pie(stats, names='Moment', values='CA TTC', hole=0.6, 
                        color_discrete_sequence=['#29B6F6', '#01579B'], title="Part du CA")
        fig_ca.update_traces(textinfo='percent+label')
        fig_ca.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0), height=180)
        st.plotly_chart(fig_ca, use_container_width=True)
        
        fig_cli = px.pie(stats, names='Moment', values='Nombre de clients', hole=0.6, 
                         color_discrete_sequence=['#66BB6A', '#1B5E20'], title="Part Clients")
        fig_cli.update_traces(textinfo='percent+label')
        fig_cli.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0), height=180)
        st.plotly_chart(fig_cli, use_container_width=True)

# B. CASCADE
with c_right:
    # Setup Comparaison
    if "M-1" in waterfall_comp:
        label_prev = f"{noms_mois[prev_m]} {prev_y_m}"
        # Pour les activités, on filtre à la main car pas de fonction consolidate (pas de bug caisse ici normalement)
        df_act_prev = df_activite[(df_activite['Date'].dt.year == prev_y_m) & (df_activite['Date'].dt.month == prev_m)]
        start_total = ca_m1
    else:
        label_prev = f"{noms_mois[mois_sel]} {annee_sel-1}"
        df_act_prev = df_activite[(df_activite['Date'].dt.year == annee_sel - 1) & (df_activite['Date'].dt.month == mois_sel)]
        start_total = ca_n1

    label_curr = f"{noms_mois[mois_sel]} {annee_sel}"
    st.subheader(f"🌉 Pont CA : {label_prev} ➔ {label_curr}")

    if not df_act_curr.empty and not df_act_prev.empty:
        # Résumé haut
        delta_total = ca_c - start_total
        delta_pct = (delta_total / start_total * 100) if start_total > 0 else 0
        color_delta = "#00FF00" if delta_total >= 0 else "#FF4444"
        sym_delta = "+" if delta_total >= 0 else ""
        
        st.markdown(f"""
        <div class='bridge-summary'>
            Variation Globale : <span style='color:{color_delta}'>{sym_delta}{delta_total:,.0f} € ({sym_delta}{delta_pct:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

        # Calcul Deltas
        grp_curr = df_act_curr.groupby('ACTIVITE')['CA TTC'].sum()
        grp_prev = df_act_prev.groupby('ACTIVITE')['CA TTC'].sum()
        
        df_bridge = pd.DataFrame({'Actuel': grp_curr, 'Prev': grp_prev}).fillna(0)
        df_bridge['Delta'] = df_bridge['Actuel'] - df_bridge['Prev']
        df_bridge['AbsDelta'] = df_bridge['Delta'].abs()
        df_bridge = df_bridge.sort_values('AbsDelta', ascending=False)
        
        top_n = 6
        if len(df_bridge) > top_n:
            main_acts = df_bridge.head(top_n)
            other_delta = df_bridge.iloc[top_n:]['Delta'].sum()
            final_deltas = main_acts['Delta'].to_dict()
            final_deltas['Autres'] = other_delta
        else:
            final_deltas = df_bridge['Delta'].to_dict()

        measures = ["absolute"] + ["relative"] * len(final_deltas) + ["absolute"]
        x_vals = [label_prev] + list(final_deltas.keys()) + [label_curr]
        y_vals = [start_total] + list(final_deltas.values()) + [ca_c]
        text_vals = [f"{start_total/1000:.0f}k"] + [f"{v/1000:+.1f}k" for v in final_deltas.values()] + [f"{ca_c/1000:.0f}k"]

        running = [start_total]
        curr_run = start_total
        for v in final_deltas.values():
            curr_run += v
            running.append(curr_run)
        max_h = max(running + [ca_c]) * 1.15

        fig_water = go.Figure(go.Waterfall(
            name="Pont CA", orientation="v",
            measure=measures, x=x_vals, y=y_vals, text=text_vals, textposition="outside",
            connector={"mode": "between", "line": {"width": 1, "color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#FF4444"}},
            increasing={"marker": {"color": "#00C851"}},
            totals={"marker": {"color": "#33b5e5"}}
        ))
        fig_water.update_layout(height=350, showlegend=False, yaxis=dict(range=[0, max_h], title="CA TTC (€)"), margin=dict(t=10, r=10))
        st.plotly_chart(fig_water, use_container_width=True)
    else:
        st.warning(f"Données manquantes.")

st.markdown("---")

# --- 5. HEATMAP CORRECTIVE ---
st.subheader("🔥 Heatmap Hebdomadaire (Fiabilisée)")

c_h1, c_h2 = st.columns([2, 8]) 
indic = c_h1.selectbox("Indicateur Heatmap", ["CA TTC", "Clients", "Panier"])
vue = c_h2.selectbox("Type d'analyse", ["Valeur Moyenne", "Évolution vs M-1", "Évolution vs N-1"])

if not df_curr.empty:
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fr_days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    all_hours = sorted(df_horaire['Heure'].unique())

    def get_hm_data(df_conso, indicator):
        # NOTE : On reçoit déjà un DF consolidé (df_curr, df_n1, etc.)
        # On rajoute juste le jour pour le pivot
        df_conso['Day'] = df_conso['Date'].dt.day_name()
        
        if indicator == "Panier":
            # Moyenne pondérée
            g = df_conso.groupby(['Day', 'Heure']).agg({'CA TTC':'sum', 'Nombre de clients':'sum'}).reset_index()
            g['Val'] = g['CA TTC'] / g['Nombre de clients']
        else:
            col = 'CA TTC' if indicator == "CA TTC" else 'Nombre de clients'
            # Moyenne simple des jours consolidés
            g = df_conso.groupby(['Day', 'Heure'])[col].mean().reset_index()
            g.rename(columns={col:'Val'}, inplace=True)
        
        pivot = g.pivot(index='Day', columns='Heure', values='Val')
        pivot = pivot.reindex(index=days_order, columns=all_hours, fill_value=0)
        return pivot.fillna(0)

    # Calcul
    matrix_curr = get_hm_data(df_curr, indic)
    
    z_values = matrix_curr.values
    text_values = np.round(matrix_curr.values).astype(str)
    colors = "Blues"
    zmin, zmax = None, None
    
    if "Évolution" in vue:
        df_ref = df_m1 if "M-1" in vue else df_n1
        if not df_ref.empty:
            matrix_ref = get_hm_data(df_ref, indic)
            
            curr_vals = matrix_curr.values
            ref_vals = matrix_ref.values
            diff_pct = np.zeros(curr_vals.shape)
            display_text = np.empty(curr_vals.shape, dtype=object)
            rows, cols = curr_vals.shape
            
            for r in range(rows):
                for c in range(cols):
                    v_cur = curr_vals[r, c]
                    v_ref = ref_vals[r, c]
                    if v_ref == 0 and v_cur == 0:
                        diff_pct[r, c] = 0; display_text[r, c] = "-"
                    elif v_ref == 0 and v_cur > 0:
                        diff_pct[r, c] = 100; display_text[r, c] = "Ouv."
                    elif v_ref > 0 and v_cur == 0:
                        diff_pct[r, c] = -100; display_text[r, c] = "Ferm."
                    else:
                        pct = ((v_cur - v_ref) / v_ref) * 100
                        diff_pct[r, c] = pct; display_text[r, c] = f"{pct:+.0f}%"

            z_values = diff_pct
            text_values = display_text
            colors = "RdBu"
            zmin, zmax = -100, 100
        else:
            st.warning("Pas de données ref.")
            z_values[:] = 0; text_values[:] = "-"

    fig_hm = go.Figure(data=go.Heatmap(
        z=z_values, x=all_hours, y=fr_days,
        colorscale=colors, zmid=0 if "Évolution" in vue else None,
        zmin=zmin, zmax=zmax, xgap=2, ygap=2,
        text=text_values, texttemplate="%{text}",
        textfont={"size": 10}, hoverongaps=False, showscale=False
    ))
    fig_hm.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), yaxis_autorange='reversed', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hm, use_container_width=True)
