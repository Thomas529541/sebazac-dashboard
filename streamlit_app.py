import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# --- CONFIGURATION PAGE & CSS ---
st.set_page_config(page_title="Pilotage Commerce V9", layout="wide", page_icon="🎯")

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
    
    /* KPI GLOBAL STYLES (CENTRÉ) */
    .kpi-container {
        background-color: #262730; border-radius: 8px; padding: 15px;
        border-left: 5px solid #FF4B4B; margin-bottom: 10px;
        text-align: center;
    }
    .kpi-title { font-size: 14px; color: #bbb; font-weight: 500; }
    .kpi-value { font-size: 28px; font-weight: bold; color: white; margin: 5px 0; }
    .kpi-sub { font-size: 13px; margin-top: 5px; display: flex; justify-content: center; gap: 15px; }
    
    /* CARTES DÉTAILLÉES */
    .detail-card {
        background-color: #1E1E1E; border-radius: 8px; padding: 12px;
        border: 1px solid #333; margin-bottom: 10px;
    }
    .detail-header {
        border-bottom: 1px solid #444; padding-bottom: 8px; margin-bottom: 10px;
        font-size: 16px; font-weight: bold; color: #FFD700;
        display: flex; justify-content: space-between;
    }
    .detail-grid {
        display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; }
    .metric-val { font-size: 15px; font-weight: bold; color: white; margin: 2px 0; }
    .metric-delta { font-size: 10px; }
    
    /* KPIS OPERATIONNELS */
    .op-kpi-box {
        background-color: #383838; padding: 15px; border-radius: 8px; 
        margin-bottom: 10px; text-align: center; border: 1px solid #555;
    }
    .op-kpi-title { color: #ddd; font-size: 14px; margin-bottom: 5px; }
    .op-kpi-val { color: #fff; font-size: 24px; font-weight: bold; }
    .op-kpi-bench { color: #aaa; font-size: 14px; margin-top: 5px; font-weight: 500; }

    /* ALERTES */
    .smart-alert {
        background-color: #2b3e50; color: #e0e0e0; padding: 15px; border-radius: 8px;
        border-left: 5px solid #00C851; margin-bottom: 15px; font-size: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .smart-alert-warn {
        background-color: #3e2b2b; color: #e0e0e0; padding: 15px; border-radius: 8px;
        border-left: 5px solid #FF4444; margin-bottom: 15px; font-size: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* CAPTION FOOTER */
    .footer-caption {
        text-align: center; color: #666; font-size: 12px; margin-top: 30px; font-style: italic;
    }

    .pos { color: #00FF00; }
    .neg { color: #FF4444; }
    .neu { color: #888888; }
</style>
""", unsafe_allow_html=True)

# --- 0. PALETTE ---
COLOR_MAP = {
    "Tabac": "#607D8B", "Jeux": "#29B6F6", "Bar Brasserie": "#AB47BC",
    "Presse": "#FF7043", "Divers": "#8D6E63", "Monétique": "#FFCA28", "Autres": "#78909C"
}

# --- 1. CHARGEMENT ---
@st.cache_data
def load_data():
    try:
        fichier = 'donnees.xlsx'
        df_h = pd.read_excel(fichier, sheet_name='ANALYSE HORAIRE')
        df_a = pd.read_excel(fichier, sheet_name='Analyse Activités')
        try: df_f = pd.read_excel(fichier, sheet_name='ANALYSE FAMILLES')
        except: df_f = pd.DataFrame()

        def clean_curr(s):
            if s.dtype == 'object': return s.astype(str).str.replace('€', '').str.replace(' ', '').str.replace(',', '.').astype(float)
            return s

        for df in [df_h, df_a, df_f]:
            if df.empty: continue
            for col in df.columns:
                if 'CA TTC' in col: df[col] = clean_curr(df[col])
                if 'Quantité' in col or 'clients' in col: df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'Période' in df.columns:
                df['DateFull'] = pd.to_datetime(df['Période'], dayfirst=True)
                df['Date'] = df['DateFull'].dt.normalize()
        return df_h, df_a, df_f
    except Exception as e:
        st.error(f"Erreur : {e}"); st.stop()

df_horaire, df_activite, df_famille = load_data()

# --- UTILS ---
def consolidate_registers(df):
    if df.empty: return df
    return df.groupby(['Date', 'Heure']).sum(numeric_only=True).reset_index()

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
# MENU & FILTRES
# ==============================================================================
st.sidebar.title("🎯 Pilotage")
page = st.sidebar.radio("Navigation :", ["🏠 Synthèse Mensuelle", "📅 Focus Jour & Semaine", "📈 Tendances & Familles"])
st.sidebar.markdown("---")

# ==============================================================================
# PAGE 1 : SYNTHÈSE MENSUELLE
# ==============================================================================
if page == "🏠 Synthèse Mensuelle":
    st.sidebar.header("Filtres Cockpit")
    annees = sorted(df_horaire['Date'].dt.year.unique(), reverse=True)
    annee_sel = st.sidebar.selectbox("Année", annees)
    mois_dispo = sorted(df_horaire[df_horaire['Date'].dt.year == annee_sel]['Date'].dt.month.unique(), reverse=True)
    noms_mois = {1:'Jan', 2:'Fév', 3:'Mars', 4:'Avr', 5:'Mai', 6:'Juin', 7:'Juil', 8:'Août', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Déc'}
    mois_sel = st.sidebar.selectbox("Mois", mois_dispo, format_func=lambda x: noms_mois[x])
    waterfall_comp = st.sidebar.radio("Cascade vs :", ["Mois Précédent (M-1)", "Année Précédente (N-1)"])

    # PREP
    df_curr = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == annee_sel) & (df_horaire['Date'].dt.month == mois_sel)])
    df_act_curr = df_activite[(df_activite['Date'].dt.year == annee_sel) & (df_activite['Date'].dt.month == mois_sel)]
    
    if mois_sel == 1: prev_m, prev_y_m = 12, annee_sel - 1
    else: prev_m, prev_y_m = mois_sel - 1, annee_sel
    
    df_m1 = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == prev_y_m) & (df_horaire['Date'].dt.month == prev_m)])
    df_act_m1 = df_activite[(df_activite['Date'].dt.year == prev_y_m) & (df_activite['Date'].dt.month == prev_m)]
    
    df_n1 = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == annee_sel - 1) & (df_horaire['Date'].dt.month == mois_sel)])
    df_act_n1 = df_activite[(df_activite['Date'].dt.year == annee_sel - 1) & (df_activite['Date'].dt.month == mois_sel)]

    st.title(f"Cockpit : {noms_mois[mois_sel]} {annee_sel}")
    
    # 1. KPIs
    ca_c, cli_c = df_curr['CA TTC'].sum(), df_curr['Nombre de clients'].sum()
    pm_c = ca_c/cli_c if cli_c else 0
    ca_m1, cli_m1 = df_m1['CA TTC'].sum(), df_m1['Nombre de clients'].sum()
    pm_m1 = ca_m1/cli_m1 if cli_m1 else 0
    ca_n1, cli_n1 = df_n1['CA TTC'].sum(), df_n1['Nombre de clients'].sum()
    pm_n1 = ca_n1/cli_n1 if cli_n1 else 0

    k1, k2, k3 = st.columns(3)
    def kpi_display(title, val, val_m1, val_n1, unit=""):
        e_m, c_m, s_m = calc_evo(val, val_m1)
        e_n, c_n, s_n = calc_evo(val, val_n1)
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{fmt_val(val, unit)}</div>
            <div class="kpi-sub">
                <span style="color:#aaa;">vs M-1:</span> <span class="{c_m}">{s_m}{abs(e_m):.1f}%</span>
                <span style="color:#aaa;">vs N-1:</span> <span class="{c_n}">{s_n}{abs(e_n):.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k1: kpi_display("Chiffre d'Affaires", ca_c, ca_m1, ca_n1, "€")
    with k2: kpi_display("Fréquentation", cli_c, cli_m1, cli_n1, "")
    with k3: kpi_display("Panier Moyen", pm_c, pm_m1, pm_n1, "PM")

    st.markdown("---")

    # 2. COLONNES
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.subheader("⚖️ Équilibre Journée")
        def get_moment_stats(df, moment):
            if df.empty: return 0, 0, 0
            df = df.copy()
            df['Heure_Int'] = df['Heure'].astype(str).str.slice(0, 2).astype(int)
            d = df[df['Heure_Int'] < 13] if moment == 'Matin' else df[df['Heure_Int'] >= 13]
            return d['CA TTC'].sum(), d['Nombre de clients'].sum(), (d['CA TTC'].sum()/d['Nombre de clients'].sum() if d['Nombre de clients'].sum()>0 else 0)

        for mom, icon in [("Matin", "🌞"), ("Après-Midi", "🌙")]:
            ca, cli, pm = get_moment_stats(df_curr, mom)
            ca_m, cli_m, pm_m = get_moment_stats(df_m1, mom)
            ca_n, cli_n, pm_n = get_moment_stats(df_n1, mom)
            ev_ca_m, cl_ca_m, _ = calc_evo(ca, ca_m); ev_ca_n, cl_ca_n, _ = calc_evo(ca, ca_n)
            ev_cli_m, cl_cli_m, _ = calc_evo(cli, cli_m); ev_cli_n, cl_cli_n, _ = calc_evo(cli, cli_n)
            ev_pm_m, cl_pm_m, _ = calc_evo(pm, pm_m); ev_pm_n, cl_pm_n, _ = calc_evo(pm, pm_n)

            st.markdown(f"""
            <div class="detail-card">
                <div class="detail-header">
                    <span>{icon} {mom.upper()}</span>
                    <span style="font-size:12px; font-weight:normal; color:#aaa;">{(ca/ca_c*100 if ca_c>0 else 0):.0f}% CA</span>
                </div>
                <div class="detail-grid">
                    <div>
                        <div class="metric-label">CA</div>
                        <div class="metric-val">{ca/1000:.1f}k€</div>
                        <div class="metric-delta {cl_ca_m}">M-1 {ev_ca_m:+.0f}%</div>
                        <div class="metric-delta {cl_ca_n}">N-1 {ev_ca_n:+.0f}%</div>
                    </div>
                    <div>
                        <div class="metric-label">Freq.</div>
                        <div class="metric-val">{cli/1000:.1f}k</div>
                        <div class="metric-delta {cl_cli_m}">{ev_cli_m:+.0f}%</div>
                        <div class="metric-delta {cl_cli_n}">{ev_cli_n:+.0f}%</div>
                    </div>
                    <div>
                        <div class="metric-label">Panier</div>
                        <div class="metric-val">{pm:.1f}€</div>
                        <div class="metric-delta {cl_pm_m}">{ev_pm_m:+.0f}%</div>
                        <div class="metric-delta {cl_pm_n}">{ev_pm_n:+.0f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with c_right:
        if "M-1" in waterfall_comp: lbl_prev = f"{noms_mois[prev_m]}"; df_prev = df_act_m1; start_val = ca_m1
        else: lbl_prev = f"{noms_mois[mois_sel]} {annee_sel-1}"; df_prev = df_act_n1; start_val = ca_n1
        st.subheader(f"🌉 Pont CA : {lbl_prev} ➔ Actuel")
        
        if not df_act_curr.empty and not df_prev.empty:
            grp_c = df_act_curr.groupby('ACTIVITE')['CA TTC'].sum()
            grp_p = df_prev.groupby('ACTIVITE')['CA TTC'].sum()
            df_b = pd.DataFrame({'Actuel': grp_c, 'Prev': grp_p}).fillna(0)
            df_b['Delta'] = df_b['Actuel'] - df_b['Prev']
            df_b['Abs'] = df_b['Delta'].abs()
            df_b = df_b.sort_values('Abs', ascending=False)
            
            if len(df_b) > 6:
                main = df_b.head(6); other = df_b.iloc[6:]['Delta'].sum()
                deltas = main['Delta'].to_dict(); deltas['Autres'] = other
            else: deltas = df_b['Delta'].to_dict()

            measures = ["absolute"] + ["relative"] * len(deltas) + ["absolute"]
            x = [lbl_prev] + list(deltas.keys()) + ["Actuel"]
            y = [start_val] + list(deltas.values()) + [ca_c]
            text = [f"{start_val/1000:.0f}k"] + [f"{v/1000:+.1f}k" for v in deltas.values()] + [f"{ca_c/1000:.0f}k"]
            
            run = [start_val]; cur = start_val
            for v in deltas.values(): cur+=v; run.append(cur)
            max_h = max(run + [ca_c]) * 1.15

            fig = go.Figure(go.Waterfall(
                orientation="v", measure=measures, x=x, y=y, text=text, textposition="outside",
                connector={"mode":"between", "line":{"width":1, "color":"#555"}},
                decreasing={"marker":{"color":"#FF4444"}}, increasing={"marker":{"color":"#00C851"}}, 
                totals={"marker":{"color":"#2c3e50"}}
            ))
            fig.update_layout(height=400, showlegend=False, yaxis=dict(range=[0, max_h], title="CA TTC"), margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Données manquantes.")

    st.markdown("---")

    # 3. HEATMAP
    st.subheader("🔥 Heatmap Hebdomadaire & Highlights")
    
    if not df_curr.empty and not df_n1.empty:
        df_curr['Day'] = df_curr['Date'].dt.day_name()
        df_n1['Day'] = df_n1['Date'].dt.day_name()
        stats_curr = df_curr.groupby('Day')['CA TTC'].mean()
        stats_n1 = df_n1.groupby('Day')['CA TTC'].mean()
        fr_map = {'Monday':'Lundi','Tuesday':'Mardi','Wednesday':'Mercredi','Thursday':'Jeudi','Friday':'Vendredi','Saturday':'Samedi','Sunday':'Dimanche'}
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        deltas = {}
        for d in days_order:
            vc = stats_curr.get(d, 0); vn = stats_n1.get(d, 0)
            deltas[d] = ((vc - vn) / vn) * 100 if vn > 0 else 0
        best_d = max(deltas, key=deltas.get); best_val = deltas[best_d]
        worst_d = min(deltas, key=deltas.get); worst_val = deltas[worst_d]
        alerts = []
        if best_val > 5: alerts.append(f"🚀 **Performance Exceptionnelle :** Vos **{fr_map[best_d]}s** surperforment de **+{best_val:.1f}%** vs N-1.")
        if worst_val < -5: alerts.append(f"⚠️ **Point de Vigilance :** Vos **{fr_map[worst_d]}s** décrochent de **{worst_val:.1f}%** vs N-1.")
        if alerts:
            for alert in alerts:
                css = "smart-alert" if "🚀" in alert else "smart-alert-warn"
                st.markdown(f"<div class='{css}'>{alert}</div>", unsafe_allow_html=True)
        else: st.info("✅ Aucune anomalie majeure détectée (Variations < 5%).")

    c_h1, c_h2 = st.columns([2, 8])
    hm_kpi = c_h1.selectbox("Indicateur Heatmap", ["CA TTC", "Clients", "Panier"])
    hm_view = c_h2.selectbox("Type d'analyse", ["Valeur Moyenne", "Évolution vs M-1", "Évolution vs N-1"])

    if not df_curr.empty:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fr_days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        all_h = sorted(df_horaire['Heure'].unique())

        def get_hm(df, ind):
            df['Day'] = df['Date'].dt.day_name()
            if ind == "Panier":
                g = df.groupby(['Day', 'Heure']).agg({'CA TTC':'sum', 'Nombre de clients':'sum'}).reset_index()
                g['Val'] = g['CA TTC'] / g['Nombre de clients']
            else:
                col = 'CA TTC' if ind == "CA TTC" else 'Nombre de clients'
                g = df.groupby(['Day', 'Heure'])[col].mean().reset_index()
                g.rename(columns={col:'Val'}, inplace=True)
            return g.pivot(index='Day', columns='Heure', values='Val').reindex(index=days, columns=all_h, fill_value=0).fillna(0)

        mat_curr = get_hm(df_curr, hm_kpi)
        
        if "Évolution" in hm_view:
            df_ref = df_m1 if "M-1" in hm_view else df_n1
            if not df_ref.empty:
                mat_ref = get_hm(df_ref, hm_kpi)
                c_v, r_v = mat_curr.values, mat_ref.values
                d_pct = np.zeros(c_v.shape); d_txt = np.empty(c_v.shape, dtype=object)
                for r in range(c_v.shape[0]):
                    for c in range(c_v.shape[1]):
                        vc, vr = c_v[r,c], r_v[r,c]
                        if vr==0 and vc==0: d_pct[r,c]=0; d_txt[r,c]="-"
                        elif vr==0: d_pct[r,c]=100; d_txt[r,c]="Ouv."
                        elif vc==0: d_pct[r,c]=-100; d_txt[r,c]="Ferm."
                        else: p=((vc-vr)/vr)*100; d_pct[r,c]=p; d_txt[r,c]=f"{p:+.0f}%"
                z, txt, colors, zmin, zmax = d_pct, d_txt, "RdBu", -100, 100
            else: z[:]=0; txt[:]="-"; colors="Blues"; zmin=None; zmax=None
        else:
            z = mat_curr.values
            txt = np.round(mat_curr.values).astype(int).astype(str)
            colors, zmin, zmax = "Blues", None, None

        fig = go.Figure(go.Heatmap(z=z, x=all_h, y=fr_days, colorscale=colors, zmin=zmin, zmax=zmax, zmid=0 if zmin else None,
                                   xgap=2, ygap=2, text=txt, texttemplate="%{text}", textfont={"size":11}, showscale=False))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), yaxis_autorange='reversed')
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 2 : FOCUS JOUR & SEMAINE
# ==============================================================================
elif page == "📅 Focus Jour & Semaine":
    st.sidebar.header("Filtres Opérationnels")
    view_mode = st.sidebar.radio("Vue :", ["Journée", "Semaine Complète"], horizontal=True)
    min_date = df_horaire['Date'].min()
    max_date = df_horaire['Date'].max()
    date_focus = st.sidebar.date_input("Date Pivot", value=max_date, min_value=min_date, max_value=max_date)
    date_focus = pd.to_datetime(date_focus)
    
    st.title("📅 Analyse Opérationnelle")

    if view_mode == "Journée":
        start_date = date_focus; end_date = date_focus
        lbl_per = f"Journée du {date_focus.strftime('%d/%m/%Y')}"
    else:
        start_date = date_focus - timedelta(days=date_focus.weekday())
        end_date = start_date + timedelta(days=6)
        lbl_per = f"Semaine du {start_date.strftime('%d/%m')} au {end_date.strftime('%d/%m')}"
    
    mask_f = (df_horaire['Date'] >= start_date) & (df_horaire['Date'] <= end_date)
    df_f = consolidate_registers(df_horaire[mask_f])
    
    bench_start = start_date - timedelta(weeks=8)
    mask_b = (df_horaire['Date'] >= bench_start) & (df_horaire['Date'] < start_date)
    df_b_raw = consolidate_registers(df_horaire[mask_b])
    
    if view_mode == "Journée":
        target = date_focus.day_name()
        df_b_raw['D'] = df_b_raw['Date'].dt.day_name()
        df_b = df_b_raw[df_b_raw['D'] == target]
        norm = max(1, df_b['Date'].nunique())
    else:
        df_b = df_b_raw; norm = 8

    if not df_f.empty:
        ca_f = df_f['CA TTC'].sum(); ca_b = df_b['CA TTC'].sum()/norm
        cli_f = df_f['Nombre de clients'].sum(); cli_b = df_b['Nombre de clients'].sum()/norm
        pm_f = ca_f/cli_f if cli_f else 0; pm_b = ca_b/cli_b if cli_b else 0
        
        c1, c2, c3 = st.columns(3)
        def op_kpi(col, title, val, bench, unit=""):
            diff = val - bench; pct = (diff/bench*100) if bench>0 else 0
            color = "green" if diff>=0 else "red"
            col.markdown(f"""
            <div class="op-kpi-box" style="border-left:5px solid {color}">
                <div class="op-kpi-title">{title}</div>
                <div class="op-kpi-val">{fmt_val(val, unit)}</div>
                <div class="op-kpi-bench" style="color:{color}">Habitude : {fmt_val(bench, unit)} ({diff:+.0f})</div>
            </div>
            """, unsafe_allow_html=True)
            
        op_kpi(c1, "Chiffre d'Affaires", ca_f, ca_b, "€")
        op_kpi(c2, "Fréquentation", cli_f, cli_b, "")
        op_kpi(c3, "Panier Moyen", pm_f, pm_b, "PM")
        
        st.markdown("---")
        
        if view_mode == "Journée":
            chart_d = df_f.groupby('Heure')['CA TTC'].sum().reset_index()
            chart_b = df_b.groupby('Heure')['CA TTC'].mean().reset_index()
            x_col = 'Heure'
        else:
            df_f['D'] = df_f['Date'].dt.day_name()
            orders = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            frs = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']
            df_f['D'] = pd.Categorical(df_f['D'], categories=orders, ordered=True)
            chart_d = df_f.groupby('D')['CA TTC'].sum().reset_index().sort_values('D')
            chart_d['D'] = chart_d['D'].map(dict(zip(orders, frs)))
            
            df_b['D'] = df_b['Date'].dt.day_name()
            chart_b = df_b.groupby('D')['CA TTC'].sum()/norm
            chart_b = chart_b.reindex(orders).reset_index(); chart_b['D'] = chart_b['D'].map(dict(zip(orders, frs)))
            x_col = 'D'
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_d[x_col], y=chart_d['CA TTC'], mode='lines+markers+text', name='Actuel',
            line=dict(color='#FFD700', width=4),
            text=[f"{v:,.0f}".replace(",", " ") for v in chart_d['CA TTC']],
            textposition="top center", 
            textfont=dict(size=13, weight='bold')
        ))
        fig.add_trace(go.Scatter(
            x=chart_b[x_col], y=chart_b['CA TTC'], mode='lines', name='Habitude (Moy)',
            line=dict(color='#888', width=2, dash='dot')
        ))
        fig.update_layout(title="Comparaison CA vs Habitude", height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Détail Chiffré")
        if view_mode == "Journée":
            t = df_f.groupby('Heure').agg(CA=('CA TTC','sum'), Cli=('Nombre de clients','sum')).reset_index()
            b = df_b.groupby('Heure').agg(CA_B=('CA TTC','mean'), Cli_B=('Nombre de clients','mean')).reset_index()
            t = t.merge(b, on='Heure', how='left')
            disp_col = 'Heure'
        else:
            t = df_f.groupby('Date').agg(CA=('CA TTC','sum'), Cli=('Nombre de clients','sum')).reset_index()
            # MAPPING JOURS POUR LE TABLEAU
            fr_days_map = {'Monday':'Lundi','Tuesday':'Mardi','Wednesday':'Mercredi','Thursday':'Jeudi','Friday':'Vendredi','Saturday':'Samedi','Sunday':'Dimanche'}
            t['DayName'] = t['Date'].dt.day_name().map(fr_days_map)
            t['DateStr'] = t['Date'].dt.strftime('%d/%m')
            t['Label'] = t['DayName'] + " " + t['DateStr']
            
            # Bench par jour de semaine pour la semaine
            df_b['D_Name'] = df_b['Date'].dt.day_name()
            b_day = df_b.groupby('D_Name').agg(CA_B=('CA TTC','mean'), Cli_B=('Nombre de clients','mean'))/ (norm/8 if norm>1 else 1) # Approx bench weekly avg per day
            # Pour simplifier, on recalcule le bench exact par nom de jour
            b_clean = df_b.groupby('D')['CA TTC'].sum().reset_index() # Déjà normé plus haut dans chart_b
            
            # Merge complexe par jour, on simplifie pour l'affichage en mettant 0 diff si compliqué
            t['CA_B'] = 0; t['Cli_B'] = 0
            disp_col = 'Label'

        t['PM'] = t['CA']/t['Cli']; 
        # Gestion division par zero
        t['PM_B'] = t.apply(lambda row: row['CA_B']/row['Cli_B'] if row['Cli_B']>0 else 0, axis=1)
        
        t['Diff CA'] = t['CA'] - t['CA_B']
        t['Diff Cli'] = t['Cli'] - t['Cli_B']
        t['Diff PM'] = t['PM'] - t['PM_B']
        
        disp = pd.DataFrame()
        disp['Créneau'] = t[disp_col]
        disp['CA'] = t['CA'].apply(lambda x: f"{x:,.0f} €".replace(","," "))
        disp['Diff CA'] = t['Diff CA'].apply(lambda x: f"{x:+,.0f} €".replace(","," ")) if view_mode=="Journée" else "-"
        disp['Freq'] = t['Cli'].apply(lambda x: f"{x:.0f}")
        disp['Diff Freq'] = t['Diff Cli'].apply(lambda x: f"{x:+.0f}") if view_mode=="Journée" else "-"
        disp['Panier'] = t['PM'].apply(lambda x: f"{x:.2f} €")
        disp['Diff PM'] = t['Diff PM'].apply(lambda x: f"{x:+.2f} €") if view_mode=="Journée" else "-"
        
        row_tot = {
            'Créneau': 'TOTAL', 
            'CA': f"{t['CA'].sum():,.0f} €".replace(","," "), 'Diff CA': "-",
            'Freq': f"{t['Cli'].sum():,.0f}", 'Diff Freq': "-",
            'Panier': f"{t['CA'].sum()/t['Cli'].sum():.2f} €", 'Diff PM': "-"
        }
        disp = pd.concat([disp, pd.DataFrame([row_tot])], ignore_index=True)
        
        def style_rows(row):
            if row['Créneau'] == 'TOTAL': return ['background-color: #444; color: white; font-weight: bold'] * len(row)
            return [''] * len(row)
            
        st.dataframe(disp.style.apply(style_rows, axis=1), use_container_width=True, hide_index=True)
        
        st.markdown("<div class='footer-caption'>ℹ️ Les habitudes sont calculées sur la moyenne des 8 dernières semaines équivalentes.</div>", unsafe_allow_html=True)

# ==============================================================================
# PAGE 3 : TENDANCES & FAMILLES
# ==============================================================================
elif page == "📈 Tendances & Familles":
    st.sidebar.header("Filtres")
    date_end = st.sidebar.date_input("Fin Période", value=df_horaire['Date'].max())
    date_end = pd.to_datetime(date_end)
    date_start = date_end - pd.DateOffset(months=12)
    
    # 1. TENDANCE 12 MOIS
    mask_12m = (df_horaire['Date'] > date_start) & (df_horaire['Date'] <= date_end)
    df_12m = consolidate_registers(df_horaire[mask_12m])
    
    st.title("📈 Tendances & Familles")
    
    if not df_12m.empty:
        df_12m['Mois'] = df_12m['Date'].dt.to_period('M').astype(str)
        monthly = df_12m.groupby('Mois').agg({'CA TTC':'sum', 'Nombre de clients':'sum'}).reset_index()
        monthly['Panier'] = monthly['CA TTC'] / monthly['Nombre de clients']
        
        st.subheader("Historique 12 Mois")
        c1, c2, c3 = st.columns(3)
        def plot_trend(col, title, y_col, color):
            avg = monthly[y_col].mean()
            fig = px.line(monthly, x='Mois', y=y_col, title=title, markers=True, text=y_col)
            fig.update_traces(line_color=color, textposition="top center", texttemplate='%{text:.0f}')
            fig.add_hline(y=avg, line_dash="dot", line_color="#555", annotation_text="Moy")
            col.plotly_chart(fig, use_container_width=True)
        plot_trend(c1, "Chiffre d'Affaires", 'CA TTC', '#FFD700')
        plot_trend(c2, "Fréquentation", 'Nombre de clients', '#00C851')
        plot_trend(c3, "Panier Moyen", 'Panier', '#29B6F6')
        
        st.markdown("---")
        
        # 2. ZOOM QUOTIDIEN DU MOIS
        st.subheader(f"🔍 Zoom Quotidien : {date_end.strftime('%B %Y')}")
        mask_month = (df_horaire['Date'].dt.month == date_end.month) & (df_horaire['Date'].dt.year == date_end.year)
        df_daily = consolidate_registers(df_horaire[mask_month])
        
        if not df_daily.empty:
            df_daily['Jour'] = df_daily['Date'].dt.day
            daily_stats = df_daily.groupby('Jour').agg({'CA TTC':'sum', 'Nombre de clients':'sum'}).reset_index()
            daily_stats['Panier'] = daily_stats['CA TTC'] / daily_stats['Nombre de clients']
            
            d1, d2, d3 = st.columns(3)
            def plot_daily(col, title, y_col, color):
                avg = daily_stats[y_col].mean()
                fig = px.line(daily_stats, x='Jour', y=y_col, title=title, markers=True)
                fig.update_traces(line_color=color)
                fig.add_hline(y=avg, line_dash="dot", line_color="#555", annotation_text="Moy")
                col.plotly_chart(fig, use_container_width=True)
            
            plot_daily(d1, "CA Journalier", 'CA TTC', '#FFD700')
            plot_daily(d2, "Fréq Journalière", 'Nombre de clients', '#00C851')
            plot_daily(d3, "Panier Journalier", 'Panier', '#29B6F6')
        else:
            st.warning("Pas de données pour ce mois précis.")

        st.markdown("---")
        
        # 3. MIX ACTIVITÉS
        c_mix1, c_mix2 = st.columns(2)
        mask_act = (df_activite['Date'] > date_start) & (df_activite['Date'] <= date_end)
        df_act_12m = df_activite[mask_act].copy()
        df_act_12m['Mois'] = df_act_12m['Date'].dt.to_period('M').astype(str)
        m_act = df_act_12m.groupby(['Mois', 'ACTIVITE'])['CA TTC'].sum().reset_index()
        
        with c_mix1:
            st.caption("Valeur (€)")
            fig1 = px.bar(m_act, x='Mois', y='CA TTC', color='ACTIVITE', color_discrete_map=COLOR_MAP, text_auto='.2s')
            st.plotly_chart(fig1, use_container_width=True)
        with c_mix2:
            st.caption("Poids (%)")
            m_act['Total'] = m_act.groupby('Mois')['CA TTC'].transform('sum')
            m_act['Pct'] = (m_act['CA TTC'] / m_act['Total'] * 100)
            fig2 = px.bar(m_act, x='Mois', y='Pct', color='ACTIVITE', color_discrete_map=COLOR_MAP, text_auto='.0f')
            st.plotly_chart(fig2, use_container_width=True)
            
    st.markdown("---")
    st.subheader("🔍 Performance Familles (Top 10)")
    
    if not df_famille.empty:
        # LOGIQUE CALENDRAIRE STRICTE POUR FAMILLES
        sel_m = date_end.month; sel_y = date_end.year
        
        # CURR (Mois sélectionné)
        mask_c = (df_famille['Date'].dt.month == sel_m) & (df_famille['Date'].dt.year == sel_y)
        f_curr = df_famille[mask_c].groupby('FAMILLE').agg({'CA TTC':'sum', 'Quantité':'sum'})
        f_curr['PM'] = f_curr['CA TTC']/f_curr['Quantité']
        
        # PREV M-1
        prev_date = date_end - pd.DateOffset(months=1)
        prev_m = prev_date.month; prev_my = prev_date.year
        mask_pm = (df_famille['Date'].dt.month == prev_m) & (df_famille['Date'].dt.year == prev_my)
        f_m1 = df_famille[mask_pm].groupby('FAMILLE')['CA TTC'].sum()
        
        # PREV N-1
        mask_pn = (df_famille['Date'].dt.month == sel_m) & (df_famille['Date'].dt.year == sel_y - 1)
        f_n1 = df_famille[mask_pn].groupby('FAMILLE')['CA TTC'].sum()
        
        # Assemblage
        df_r = f_curr.copy().sort_values('CA TTC', ascending=False).head(10)
        df_r['CA M-1'] = f_m1
        df_r['CA N-1'] = f_n1
        
        # Calcul Evos
        df_r['Evo M-1'] = ((df_r['CA TTC'] - df_r['CA M-1'])/df_r['CA M-1']*100).fillna(0)
        df_r['Evo N-1'] = ((df_r['CA TTC'] - df_r['CA N-1'])/df_r['CA N-1']*100).fillna(0)
        
        cols = st.columns(4)
        for i, (fam, row) in enumerate(df_r.iterrows()):
            with cols[i%4]:
                em = row['Evo M-1']; en = row['Evo N-1']
                cm = "pos" if em>=0 else "neg"; cn = "pos" if en>=0 else "neg"
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-header"><span>{fam}</span></div>
                    <div class="detail-grid">
                        <div><div class="metric-label">CA</div><div class="metric-val">{row['CA TTC']/1000:.1f}k€</div><div class="metric-delta {cm}">M-1 {em:+.0f}%</div><div class="metric-delta {cn}">N-1 {en:+.0f}%</div></div>
                        <div><div class="metric-label">Vol.</div><div class="metric-val">{row['Quantité']:.0f}</div></div>
                        <div><div class="metric-label">Prix</div><div class="metric-val">{row['PM']:.1f}€</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.info("Données Familles non disponibles.")
