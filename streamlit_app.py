import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# --- CONFIGURATION PAGE & CSS ---
st.set_page_config(page_title="Pilotage Commerce V3", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    /* RESET ET MARGES */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        margin-top: 0rem !important;
    }
    
    /* KPI GLOBAL STYLES */
    .kpi-container {
        background-color: #262730; border-radius: 8px; padding: 10px 15px;
        border-left: 5px solid #FF4B4B; margin-bottom: 10px;
    }
    .kpi-title { font-size: 14px; color: #bbb; font-weight: 500; }
    .kpi-value { font-size: 24px; font-weight: bold; color: white; margin: 2px 0; }
    .kpi-sub { font-size: 13px; margin-top: 2px; }
    
    /* TABLEAUX STYLISÉS */
    .dataframe { font-size: 12px !important; }
    
    /* CARTES FAMILLES */
    .fam-card {
        background-color: #1E1E1E; border-radius: 8px; padding: 15px;
        border: 1px solid #333; margin-bottom: 10px;
    }
    .fam-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; padding-bottom: 8px; margin-bottom: 8px; }
    .fam-name { font-size: 16px; font-weight: bold; color: #FFD700; }
    .fam-stat-row { display: flex; justify-content: space-between; margin-bottom: 5px; }
    .fam-label { font-size: 12px; color: #aaa; }
    .fam-val { font-size: 14px; font-weight: bold; color: #fff; }
    
    /* COULEURS */
    .pos { color: #00FF00; font-weight: bold; }
    .neg { color: #FF4444; font-weight: bold; }
    .neu { color: #888888; }
</style>
""", unsafe_allow_html=True)

# --- 0. PALETTE DE COULEURS FIXES (Pour stabilité visuelle) ---
COLOR_MAP = {
    "Tabac": "#607D8B",       # Gris Bleu (Neutre mais imposant)
    "Jeux": "#29B6F6",        # Bleu clair (FDJ)
    "Bar Brasserie": "#AB47BC", # Violet
    "Presse": "#FF7043",      # Orange
    "Divers": "#8D6E63",      # Marron
    "Monétique": "#FFCA28",   # Jaune
    "Autres": "#78909C"       # Gris
}
def get_color(act_name):
    return COLOR_MAP.get(act_name, "#78909C")

# --- 1. CHARGEMENT ROBUSTE ---
@st.cache_data
def load_data():
    try:
        fichier = 'donnees.xlsx'
        df_h = pd.read_excel(fichier, sheet_name='ANALYSE HORAIRE')
        df_a = pd.read_excel(fichier, sheet_name='Analyse Activités')
        try:
            df_f = pd.read_excel(fichier, sheet_name='ANALYSE FAMILLES')
        except:
            df_f = pd.DataFrame()

        def clean_curr(s):
            if s.dtype == 'object':
                return s.astype(str).str.replace('€', '').str.replace(' ', '').str.replace(',', '.').astype(float)
            return s

        cols_h = {'CA TTC': clean_curr, 'Nombre de clients': pd.to_numeric}
        for col, func in cols_h.items():
            if col in df_h.columns: df_h[col] = func(df_h[col])
        if 'Période' in df_h.columns: 
            df_h['DateFull'] = pd.to_datetime(df_h['Période'], dayfirst=True)
            df_h['Date'] = df_h['DateFull'].dt.normalize()

        cols_a = {'CA TTC': clean_curr, 'Quantité': pd.to_numeric}
        for col, func in cols_a.items():
            if col in df_a.columns: df_a[col] = func(df_a[col])
        if 'Période' in df_a.columns: 
            df_a['DateFull'] = pd.to_datetime(df_a['Période'], dayfirst=True)
            df_a['Date'] = df_a['DateFull'].dt.normalize()
            
        if not df_f.empty:
            cols_f = {'CA TTC': clean_curr, 'Quantité': pd.to_numeric}
            for col, func in cols_f.items():
                if col in df_f.columns: df_f[col] = func(df_f[col])
            if 'Période' in df_f.columns:
                df_f['DateFull'] = pd.to_datetime(df_f['Période'], dayfirst=True)
                df_f['Date'] = df_f['DateFull'].dt.normalize()

        return df_h, df_a, df_f
    except Exception as e:
        st.error(f"Erreur critique : {e}")
        st.stop()

df_horaire, df_activite, df_famille = load_data()

# --- CONSOLIDATION CAISSES ---
def consolidate_registers(df):
    if df.empty: return df
    return df.groupby(['Date', 'Heure']).sum(numeric_only=True).reset_index()

# --- UTILS CALCULS ---
def calc_evo(curr, prev):
    if prev == 0 or pd.isna(prev): return 0, "neu", "="
    val = ((curr - prev) / prev) * 100
    if val > 0: return val, "pos", "▲"
    elif val < 0: return val, "neg", "▼"
    return val, "neu", "="

def fmt_val(val, unit=""):
    if unit == "€": return f"{val:,.0f} €".replace(",", " ")
    if unit == "PM": return f"{val:.2f} €"
    return f"{val:,.0f}".replace(",", " ")

# ==============================================================================
# MENU DE NAVIGATION
# ==============================================================================
st.sidebar.title("🎯 Pilotage")
page = st.sidebar.radio("Navigation :", ["🏠 Synthèse Mensuelle", "📅 Focus Jour & Semaine", "📈 Tendances & Familles"])
st.sidebar.markdown("---")

# ==============================================================================
# PAGE 1 : SYNTHÈSE MENSUELLE
# ==============================================================================
if page == "🏠 Synthèse Mensuelle":
    # FILTRES
    st.sidebar.header("Filtres Cockpit")
    annees = sorted(df_horaire['Date'].dt.year.unique(), reverse=True)
    annee_sel = st.sidebar.selectbox("Année", annees)
    mois_dispo = sorted(df_horaire[df_horaire['Date'].dt.year == annee_sel]['Date'].dt.month.unique(), reverse=True)
    noms_mois = {1:'Jan', 2:'Fév', 3:'Mars', 4:'Avr', 5:'Mai', 6:'Juin', 7:'Juil', 8:'Août', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Déc'}
    mois_sel = st.sidebar.selectbox("Mois", mois_dispo, format_func=lambda x: noms_mois[x])
    waterfall_comp = st.sidebar.radio("Cascade vs :", ["Mois Précédent (M-1)", "Année Précédente (N-1)"])

    # DATA PREP (Code existant conservé pour la synthèse)
    df_curr = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == annee_sel) & (df_horaire['Date'].dt.month == mois_sel)])
    df_act_curr = df_activite[(df_activite['Date'].dt.year == annee_sel) & (df_activite['Date'].dt.month == mois_sel)]
    
    if mois_sel == 1: prev_m, prev_y_m = 12, annee_sel - 1
    else: prev_m, prev_y_m = mois_sel - 1, annee_sel
    
    df_m1 = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == prev_y_m) & (df_horaire['Date'].dt.month == prev_m)])
    df_act_m1 = df_activite[(df_activite['Date'].dt.year == prev_y_m) & (df_activite['Date'].dt.month == prev_m)]
    
    df_n1 = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == annee_sel - 1) & (df_horaire['Date'].dt.month == mois_sel)])
    df_act_n1 = df_activite[(df_activite['Date'].dt.year == annee_sel - 1) & (df_activite['Date'].dt.month == mois_sel)]

    st.title(f"Cockpit : {noms_mois[mois_sel]} {annee_sel}")
    
    # 3 KPIs GLOBAUX (Code existant)
    ca_c, cli_c = df_curr['CA TTC'].sum(), df_curr['Nombre de clients'].sum()
    pm_c = ca_c/cli_c if cli_c else 0
    ca_m1, cli_m1 = df_m1['CA TTC'].sum(), df_m1['Nombre de clients'].sum()
    pm_m1 = ca_m1/cli_m1 if cli_m1 else 0
    ca_n1, cli_n1 = df_n1['CA TTC'].sum(), df_n1['Nombre de clients'].sum()
    pm_n1 = ca_n1/cli_n1 if cli_n1 else 0

    def kpi_display(title, val, val_m1, val_n1, unit=""):
        e_m, c_m, s_m = calc_evo(val, val_m1)
        e_n, c_n, s_n = calc_evo(val, val_n1)
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{fmt_val(val, unit)}</div>
            <div class="kpi-sub">
                <span style="color:#aaa;">vs M-1:</span> <span class="{c_m}">{s_m}{abs(e_m):.1f}%</span> |
                <span style="color:#aaa;">vs N-1:</span> <span class="{c_n}">{s_n}{abs(e_n):.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1: kpi_display("Chiffre d'Affaires", ca_c, ca_m1, ca_n1, "€")
    with k2: kpi_display("Fréquentation", cli_c, cli_m1, cli_n1, "")
    with k3: kpi_display("Panier Moyen", pm_c, pm_m1, pm_n1, "PM")

    st.markdown("---")

    # SECTION MATIN / SOIR & CASCADE
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.subheader("⚖️ Équilibre Journée")
        def get_moment_stats(df, moment):
            if df.empty: return 0, 0, 0
            df = df.copy()
            df['Heure_Int'] = df['Heure'].astype(str).str.slice(0, 2).astype(int)
            d = df[df['Heure_Int'] < 13] if moment == 'Matin' else df[df['Heure_Int'] >= 13]
            ca = d['CA TTC'].sum()
            cli = d['Nombre de clients'].sum()
            pm = ca / cli if cli > 0 else 0
            return ca, cli, pm

        for mom, icon in [("Matin", "🌞"), ("Après-Midi", "🌙")]:
            ca, cli, pm = get_moment_stats(df_curr, mom)
            ca_m, cli_m, pm_m = get_moment_stats(df_m1, mom)
            ca_n, cli_n, pm_n = get_moment_stats(df_n1, mom)
            
            share = (ca / ca_c * 100) if ca_c > 0 else 0
            ev_ca, cl_ca, _ = calc_evo(ca, ca_m)
            ev_ca_n, cl_ca_n, _ = calc_evo(ca, ca_n)

            st.markdown(f"""
            <div class="fam-card">
                <div class="fam-header">
                    <span class="fam-name">{icon} {mom}</span>
                    <span style="color:#aaa; font-size:12px;">{share:.0f}% CA</span>
                </div>
                <div class="fam-stat-row">
                    <span class="fam-label">CA</span>
                    <span class="fam-val">{ca/1000:.1f}k€</span>
                    <span class="fam-label {cl_ca}">M-1 {ev_ca:+.0f}%</span>
                </div>
                <div class="fam-stat-row">
                    <span class="fam-label">N-1</span>
                    <span class="fam-val"></span>
                    <span class="fam-label {cl_ca_n}">{ev_ca_n:+.0f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with c_right:
        if "M-1" in waterfall_comp:
            lbl_prev = f"{noms_mois[prev_m]} {prev_y_m}"; df_act_prev = df_act_m1; start_val = ca_m1
        else:
            lbl_prev = f"{noms_mois[mois_sel]} {annee_sel-1}"; df_act_prev = df_act_n1; start_val = ca_n1
        lbl_curr = f"{noms_mois[mois_sel]} {annee_sel}"
        
        st.subheader(f"🌉 Pont CA : {lbl_prev} ➔ {lbl_curr}")

        if not df_act_curr.empty and not df_act_prev.empty:
            grp_c = df_act_curr.groupby('ACTIVITE')['CA TTC'].sum()
            grp_p = df_act_prev.groupby('ACTIVITE')['CA TTC'].sum()
            df_b = pd.DataFrame({'Actuel': grp_c, 'Prev': grp_p}).fillna(0)
            df_b['Delta'] = df_b['Actuel'] - df_b['Prev']
            df_b['Abs'] = df_b['Delta'].abs()
            df_b = df_b.sort_values('Abs', ascending=False)
            
            if len(df_b) > 6:
                main = df_b.head(6); other = df_b.iloc[6:]['Delta'].sum()
                deltas = main['Delta'].to_dict(); deltas['Autres'] = other
            else:
                deltas = df_b['Delta'].to_dict()

            measures = ["absolute"] + ["relative"] * len(deltas) + ["absolute"]
            x = [lbl_prev] + list(deltas.keys()) + [lbl_curr]
            y = [start_val] + list(deltas.values()) + [ca_c]
            text = [f"{start_val/1000:.0f}k"] + [f"{v/1000:+.1f}k" for v in deltas.values()] + [f"{ca_c/1000:.0f}k"]
            
            run = [start_val]; cur = start_val
            for v in deltas.values(): cur+=v; run.append(cur)
            max_h = max(run + [ca_c]) * 1.15

            fig = go.Figure(go.Waterfall(
                orientation="v", measure=measures, x=x, y=y, text=text, textposition="outside",
                connector={"mode":"between", "line":{"width":1, "color":"#555"}},
                decreasing={"marker":{"color":"#FF4444"}}, increasing={"marker":{"color":"#00C851"}}, 
                totals={"marker":{"color":"#37474F"}}
            ))
            fig.update_layout(height=400, showlegend=False, yaxis=dict(range=[0, max_h], title="CA TTC"), margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Données manquantes.")

    st.markdown("---")
    st.subheader("🔥 Analyse Hebdomadaire")
    
    if not df_curr.empty:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fr_days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        all_h = sorted(df_horaire['Heure'].unique())

        df_curr['Day'] = df_curr['Date'].dt.day_name()
        hm_data = df_curr.groupby(['Day', 'Heure'])['CA TTC'].mean().reset_index()
        mat_curr = hm_data.pivot(index='Day', columns='Heure', values='CA TTC').reindex(index=days, columns=all_h, fill_value=0).fillna(0)
        
        # Phrases
        day_sums = mat_curr.sum(axis=1)
        best = day_sums.idxmax(); worst = day_sums[day_sums>0].idxmin() if len(day_sums[day_sums>0]) > 0 else "N/A"
        fr_map = dict(zip(days, fr_days))
        
        st.info(f"💡 Votre **{fr_map.get(best, best)}** est le jour le plus fort. Le **{fr_map.get(worst, worst)}** est le plus calme.")
        
        fig = go.Figure(go.Heatmap(z=mat_curr.values, x=all_h, y=fr_days, colorscale="Blues", xgap=2, ygap=2))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), yaxis_autorange='reversed')
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 2 : FOCUS JOUR & SEMAINE (NOUVELLE LOGIQUE)
# ==============================================================================
elif page == "📅 Focus Jour & Semaine":
    st.title("📅 Analyse Opérationnelle")
    
    # 1. Barre de contrôle
    c_ctrl1, c_ctrl2 = st.columns([1, 2])
    view_mode = c_ctrl1.radio("Vue :", ["Journée", "Semaine Complète"], horizontal=True)
    
    min_date = df_horaire['Date'].min()
    max_date = df_horaire['Date'].max()
    date_focus = c_ctrl2.date_input("Sélectionner une date pivot", value=max_date, min_value=min_date, max_value=max_date)
    date_focus = pd.to_datetime(date_focus)
    
    # 2. Détermination de la période
    if view_mode == "Journée":
        start_date = date_focus
        end_date = date_focus
        period_label = f"Journée du {date_focus.strftime('%d/%m/%Y')}"
        group_col = 'Heure'
    else:
        # Trouver le lundi et dimanche de cette semaine
        start_date = date_focus - timedelta(days=date_focus.weekday())
        end_date = start_date + timedelta(days=6)
        period_label = f"Semaine du {start_date.strftime('%d/%m')} au {end_date.strftime('%d/%m')}"
        group_col = 'Date' # En vue semaine, on groupe par jour
        
    # 3. Filtrage Données
    mask_focus = (df_horaire['Date'] >= start_date) & (df_horaire['Date'] <= end_date)
    df_focus = consolidate_registers(df_horaire[mask_focus])
    
    # 4. Calcul Benchmark (Habitude sur les 8 dernières semaines mêmes jours)
    # On prend une période large avant
    bench_start = start_date - timedelta(weeks=8)
    mask_bench = (df_horaire['Date'] >= bench_start) & (df_horaire['Date'] < start_date)
    df_bench_raw = consolidate_registers(df_horaire[mask_bench])
    
    # On ne garde que les mêmes jours de la semaine (ex: que les lundis si vue jour, ou tout si vue semaine)
    if view_mode == "Journée":
        target_day = date_focus.day_name()
        df_bench_raw['DayName'] = df_bench_raw['Date'].dt.day_name()
        df_bench = df_bench_raw[df_bench_raw['DayName'] == target_day]
        norm_factor = df_bench['Date'].nunique() # Nombre de lundis trouvés
    else:
        # Pour la semaine, benchmark = moyenne hebdomadaire sur 8 semaines
        df_bench = df_bench_raw
        norm_factor = 8 # On compare à une moyenne sur 8 semaines
        
    if df_focus.empty:
        st.warning(f"Pas de données pour {period_label}")
    else:
        # --- A. 3 LIGNES DE KPIS ---
        # Focus
        ca_f = df_focus['CA TTC'].sum()
        cli_f = df_focus['Nombre de clients'].sum()
        pm_f = ca_f / cli_f if cli_f else 0
        
        # Bench (Moyenne)
        ca_b_total = df_bench['CA TTC'].sum()
        cli_b_total = df_bench['Nombre de clients'].sum()
        
        # Si norm_factor est 0, évite division par zero
        norm_factor = max(1, norm_factor)
        
        ca_b_avg = ca_b_total / norm_factor
        cli_b_avg = cli_b_total / norm_factor
        pm_b_avg = ca_b_avg / cli_b_avg if cli_b_avg else 0
        
        # Affichage
        st.subheader(f"Performance : {period_label}")
        
        c1, c2, c3 = st.columns(3)
        
        def kpi_focus_row(col, title, val_f, val_b, unit=""):
            diff = val_f - val_b
            pct = (diff / val_b * 100) if val_b > 0 else 0
            color = "green" if diff >= 0 else "red"
            sym = "+" if diff >= 0 else ""
            col.markdown(f"""
            <div style="background:#262730; padding:10px; border-radius:5px; border-left:4px solid {color}; margin-bottom:5px;">
                <div style="color:#aaa; font-size:12px;">{title}</div>
                <div style="font-size:20px; font-weight:bold;">{fmt_val(val_f, unit)}</div>
                <div style="font-size:12px; color:{color};">Habitude : {fmt_val(val_b, unit)} ({sym}{pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
        kpi_focus_row(c1, "Chiffre d'Affaires", ca_f, ca_b_avg, "€")
        kpi_focus_row(c2, "Fréquentation", cli_f, cli_b_avg, "")
        kpi_focus_row(c3, "Panier Moyen", pm_f, pm_b_avg, "PM")
        
        st.markdown("---")
        
        # --- B. GRAPHIQUE CHIFFRÉ ---
        # Prép données graph
        if view_mode == "Journée":
            # Par Heure
            chart_data = df_focus.groupby('Heure')['CA TTC'].sum().reset_index()
            chart_bench = df_bench.groupby('Heure')['CA TTC'].mean().reset_index()
            x_col = 'Heure'
        else:
            # Par Jour (Lundi, Mardi...)
            df_focus['Day'] = df_focus['Date'].dt.day_name()
            # Pour trier Lundi -> Dimanche
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fr_days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            df_focus['Day'] = pd.Categorical(df_focus['Day'], categories=days_order, ordered=True)
            chart_data = df_focus.groupby('Day')['CA TTC'].sum().reset_index().sort_values('Day')
            chart_data['Day'] = chart_data['Day'].map(dict(zip(days_order, fr_days)))
            
            # Bench Semaine
            df_bench['Day'] = df_bench['Date'].dt.day_name()
            chart_bench = df_bench.groupby('Day')['CA TTC'].sum() / norm_factor # Moyenne par jour
            chart_bench = chart_bench.reindex(days_order).reset_index()
            chart_bench['Day'] = chart_bench['Day'].map(dict(zip(days_order, fr_days)))
            x_col = 'Day'
            
        fig = go.Figure()
        # Courbe Principale (Focus) AVEC TEXTE
        fig.add_trace(go.Scatter(
            x=chart_data[x_col], y=chart_data['CA TTC'], 
            mode='lines+markers+text', # AJOUT TEXTE
            name='Sélection',
            line=dict(color='#FFD700', width=4),
            text=[f"{v:.0f}" for v in chart_data['CA TTC']], # VALEURS SANS DÉCIMALES
            textposition="top center",
            textfont=dict(color='white', size=12)
        ))
        # Courbe Habitude
        fig.add_trace(go.Scatter(
            x=chart_bench[x_col], y=chart_bench['CA TTC'], 
            mode='lines',
            name='Moyenne Habitude',
            line=dict(color='gray', width=2, dash='dot')
        ))
        fig.update_layout(title="Évolution CA vs Habitude", height=400, hovermode="x unified", yaxis=dict(showgrid=True, gridcolor='#333'))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- C. TABLEAU DÉTAILLÉ AVEC TOTAL ---
        st.subheader("Détail Chiffré")
        
        # Construction du tableau de données
        if view_mode == "Journée":
            # Group by Heure
            tbl = df_focus.groupby('Heure').agg(
                CA=('CA TTC', 'sum'),
                Clients=('Nombre de clients', 'sum')
            ).reset_index()
            # Bench join
            bench_tbl = df_bench.groupby('Heure').agg(
                CA_B=('CA TTC', 'mean')
            ).reset_index()
            tbl = tbl.merge(bench_tbl, on='Heure', how='left')
        else:
            # Group by Date/Day
            tbl = df_focus.groupby('Date').agg(
                CA=('CA TTC', 'sum'),
                Clients=('Nombre de clients', 'sum')
            ).reset_index()
            tbl['Jour'] = tbl['Date'].dt.day_name().map(dict(zip(days_order, fr_days)))
            # Bench join (plus complexe par jour, on simplifie pour l'affichage tableau Focus uniquement)
            # On affiche juste les données réelles
            tbl['Heure'] = tbl['Jour'] # Hack pour affichage uniforme colonne 1
            
        # Calculs colonnes
        tbl['Panier'] = tbl['CA'] / tbl['Clients']
        if 'CA_B' in tbl.columns:
            tbl['Diff vs Hab.'] = tbl['CA'] - tbl['CA_B']
        else:
            tbl['Diff vs Hab.'] = 0 # Pas de diff ligne par ligne en vue semaine (compliqué à mapper)
            
        # Mise en forme
        tbl_display = pd.DataFrame()
        tbl_display['Créneau'] = tbl['Heure'] if view_mode=="Journée" else tbl['Jour']
        tbl_display['CA TTC'] = tbl['CA'].apply(lambda x: f"{x:.0f} €")
        tbl_display['Fréquentation'] = tbl['Clients'].apply(lambda x: f"{x:.0f}")
        tbl_display['Panier Moyen'] = tbl['Panier'].apply(lambda x: f"{x:.2f} €")
        if view_mode == "Journée":
            tbl_display['Diff vs Hab.'] = tbl['Diff vs Hab.'].apply(lambda x: f"{x:+.0f} €")

        # LIGNE TOTAL
        total_row = {
            'Créneau': 'TOTAL',
            'CA TTC': f"{tbl['CA'].sum():.0f} €",
            'Fréquentation': f"{tbl['Clients'].sum():.0f}",
            'Panier Moyen': f"{tbl['CA'].sum()/tbl['Clients'].sum():.2f} €",
            'Diff vs Hab.': ""
        }
        tbl_display = pd.concat([tbl_display, pd.DataFrame([total_row])], ignore_index=True)
        
        # Style table (gras pour total)
        def highlight_total(row):
            if row['Créneau'] == 'TOTAL':
                return ['background-color: #333; font-weight: bold']*len(row)
            return ['']*len(row)

        st.dataframe(tbl_display.style.apply(highlight_total, axis=1), use_container_width=True)


# ==============================================================================
# PAGE 3 : TENDANCES & FAMILLES (REFONTE)
# ==============================================================================
elif page == "📈 Tendances & Familles":
    st.title("📈 Tendances & Familles")
    
    # Filtre
    date_end = st.sidebar.date_input("Fin de période", value=df_horaire['Date'].max())
    date_end = pd.to_datetime(date_end)
    date_start = date_end - pd.DateOffset(months=12)
    
    # Données 12 mois
    mask_12m = (df_horaire['Date'] > date_start) & (df_horaire['Date'] <= date_end)
    df_12m = consolidate_registers(df_horaire[mask_12m])
    
    if not df_12m.empty:
        df_12m['Mois'] = df_12m['Date'].dt.to_period('M').astype(str)
        monthly = df_12m.groupby('Mois').agg({'CA TTC':'sum', 'Nombre de clients':'sum'}).reset_index()
        monthly['Panier'] = monthly['CA TTC'] / monthly['Nombre de clients']
        
        st.subheader("Historique 12 Mois (Avec Moyenne)")
        
        # 3 GRAPHIQUES AVEC VALEURS + MOYENNE
        c1, c2, c3 = st.columns(3)
        
        def plot_trend(col, title, y_col, color):
            avg_val = monthly[y_col].mean()
            fig = px.line(monthly, x='Mois', y=y_col, title=title, markers=True, text=y_col)
            fig.update_traces(line_color=color, textposition="top center", texttemplate='%{text:.0f}')
            # Ligne moyenne
            fig.add_hline(y=avg_val, line_dash="dot", line_color="white", annotation_text="Moy", annotation_position="bottom right")
            col.plotly_chart(fig, use_container_width=True)
            
        plot_trend(c1, "Chiffre d'Affaires", 'CA TTC', '#29B6F6')
        plot_trend(c2, "Fréquentation", 'Nombre de clients', '#66BB6A')
        plot_trend(c3, "Panier Moyen", 'Panier', '#FF7043')
            
        # MIX ACTIVITÉS -> BARRES EMPILÉES (Plus lisible que Area)
        st.subheader("Évolution du Mix Activités (Lisible)")
        mask_act_12m = (df_activite['Date'] > date_start) & (df_activite['Date'] <= date_end)
        df_act_12m = df_activite[mask_act_12m].copy()
        df_act_12m['Mois'] = df_act_12m['Date'].dt.to_period('M').astype(str)
        monthly_act = df_act_12m.groupby(['Mois', 'ACTIVITE'])['CA TTC'].sum().reset_index()
        
        # Assignation couleurs fixes
        fig = px.bar(monthly_act, x='Mois', y='CA TTC', color='ACTIVITE', title="Composition du CA par Mois",
                     color_discrete_map=COLOR_MAP, text_auto='.2s') # text_auto pour valeurs
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
    # --- FAMILLES : MODE CONCRET ---
    st.markdown("---")
    st.subheader("🔍 Analyse Familles : Top & Flop (Concret)")
    
    if not df_famille.empty:
        # Période courante vs N-1
        last_month = df_famille['Date'].max().month
        last_year = df_famille['Date'].max().year
        
        mask_curr = (df_famille['Date'].dt.month == last_month) & (df_famille['Date'].dt.year == last_year)
        mask_prev = (df_famille['Date'].dt.month == last_month) & (df_famille['Date'].dt.year == last_year - 1)
        
        fam_curr = df_famille[mask_curr].groupby('FAMILLE')['CA TTC'].sum()
        fam_prev = df_famille[mask_prev].groupby('FAMILLE')['CA TTC'].sum()
        
        # Merge
        df_comp = pd.DataFrame({'CA N': fam_curr, 'CA N-1': fam_prev}).fillna(0)
        df_comp['Diff'] = df_comp['CA N'] - df_comp['CA N-1']
        df_comp['Evo %'] = (df_comp['Diff'] / df_comp['CA N-1'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Tri par CA N décroissant
        df_comp = df_comp.sort_values('CA N', ascending=False).head(10) # TOP 10
        
        # AFFICHAGE CARTE PAR CARTE
        cols = st.columns(4)
        for idx, (fam, row) in enumerate(df_comp.iterrows()):
            with cols[idx % 4]:
                evo = row['Evo %']
                color_class = "pos" if evo > 0 else "neg"
                sym = "▲" if evo > 0 else "▼"
                
                # Sparkline simplifiée (tendance visuelle)
                # On regarde juste si ça monte ou descend globalement
                trend_color = "#00FF00" if evo > 0 else "#FF4444"
                
                st.markdown(f"""
                <div class="fam-card" style="border-left: 4px solid {trend_color}">
                    <div style="font-weight:bold; font-size:14px; margin-bottom:5px; height:40px; overflow:hidden;">{fam}</div>
                    <div style="font-size:20px; font-weight:bold;">{row['CA N']/1000:.1f} k€</div>
                    <div class="{color_class}" style="font-size:14px; margin-top:5px;">
                        {sym} {abs(evo):.1f}% <span style="color:#aaa; font-size:11px;">vs N-1</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.info("Données Familles non disponibles.")
