import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# --- CONFIGURATION PAGE & CSS ---
st.set_page_config(page_title="Cockpit Commerce V2", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    /* RESET ET MARGES */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0rem !important;
    }
    
    /* KPI GLOBAL */
    .kpi-box {
        background-color: #262730; border-radius: 8px; padding: 10px;
        text-align: center; border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 10px;
    }
    .kpi-val { font-size: 24px; font-weight: bold; color: white; margin: 2px 0; }
    .kpi-label { font-size: 13px; color: #aaaaaa; }
    
    /* CARTES DÉTAIL */
    .act-card {
        background-color: #1E1E1E; border-radius: 10px; padding: 12px;
        border: 1px solid #333; margin-bottom: 15px;
    }
    .act-title { 
        font-size: 16px; font-weight: bold; color: #FFD700; 
        border-bottom: 1px solid #333; padding-bottom: 8px; margin-bottom: 10px;
        display: flex; justify-content: space-between;
    }
    .act-share { font-size: 12px; color: #aaa; font-weight: normal; margin-top: 2px; }
    .act-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px; text-align: center; }
    .act-metric { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 2px; }
    .act-val { font-size: 17px; font-weight: bold; color: #fff; margin-bottom: 4px; }
    .act-delta { font-size: 11px; display: block; line-height: 1.2; }
    
    /* ALERTE INTELLIGENTE */
    .smart-alert {
        background-color: #2b3e50; color: #e0e0e0; padding: 10px; border-radius: 8px;
        border-left: 5px solid #FFD700; margin-bottom: 15px; font-size: 14px;
    }

    /* COULEURS VARIATIONS */
    .pos { color: #00FF00; }
    .neg { color: #FF4444; }
    .neu { color: #888888; }
</style>
""", unsafe_allow_html=True)

# --- 1. CHARGEMENT ROBUSTE ---
@st.cache_data
def load_data():
    try:
        fichier = 'donnees.xlsx'
        # On charge les 3 onglets
        df_h = pd.read_excel(fichier, sheet_name='ANALYSE HORAIRE')
        df_a = pd.read_excel(fichier, sheet_name='Analyse Activités')
        # On essaie de charger Familles, sinon dataframe vide
        try:
            df_f = pd.read_excel(fichier, sheet_name='ANALYSE FAMILLES')
        except:
            df_f = pd.DataFrame()

        def clean_curr(s):
            if s.dtype == 'object':
                return s.astype(str).str.replace('€', '').str.replace(' ', '').str.replace(',', '.').astype(float)
            return s

        # Nettoyage HORAIRE
        cols_h = {'CA TTC': clean_curr, 'Nombre de clients': pd.to_numeric}
        for col, func in cols_h.items():
            if col in df_h.columns: df_h[col] = func(df_h[col])
        if 'Période' in df_h.columns: 
            df_h['DateFull'] = pd.to_datetime(df_h['Période'], dayfirst=True)
            df_h['Date'] = df_h['DateFull'].dt.normalize()

        # Nettoyage ACTIVITE
        cols_a = {'CA TTC': clean_curr, 'Quantité': pd.to_numeric}
        for col, func in cols_a.items():
            if col in df_a.columns: df_a[col] = func(df_a[col])
        if 'Période' in df_a.columns: 
            df_a['DateFull'] = pd.to_datetime(df_a['Période'], dayfirst=True)
            df_a['Date'] = df_a['DateFull'].dt.normalize()
            
        # Nettoyage FAMILLE
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
st.sidebar.title("📱 Navigation")
page = st.sidebar.radio("Aller vers :", ["🏠 Synthèse Mensuelle", "📅 Focus Jour & Semaine", "📈 Tendances & Familles"])
st.sidebar.markdown("---")

# ==============================================================================
# PAGE 1 : SYNTHÈSE MENSUELLE (COCKPIT)
# ==============================================================================
if page == "🏠 Synthèse Mensuelle":
    
    # --- FILTRES ---
    st.sidebar.header("Filtres Cockpit")
    annees = sorted(df_horaire['Date'].dt.year.unique(), reverse=True)
    annee_sel = st.sidebar.selectbox("Année", annees)
    mois_dispo = sorted(df_horaire[df_horaire['Date'].dt.year == annee_sel]['Date'].dt.month.unique(), reverse=True)
    noms_mois = {1:'Jan', 2:'Fév', 3:'Mars', 4:'Avr', 5:'Mai', 6:'Juin', 7:'Juil', 8:'Août', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Déc'}
    mois_sel = st.sidebar.selectbox("Mois", mois_dispo, format_func=lambda x: noms_mois[x])
    waterfall_comp = st.sidebar.radio("Cascade vs :", ["Mois Précédent (M-1)", "Année Précédente (N-1)"])

    # --- DATA PREP ---
    # Actuel
    df_curr = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == annee_sel) & (df_horaire['Date'].dt.month == mois_sel)])
    df_act_curr = df_activite[(df_activite['Date'].dt.year == annee_sel) & (df_activite['Date'].dt.month == mois_sel)]
    
    # Comparatifs
    if mois_sel == 1: prev_m, prev_y_m = 12, annee_sel - 1
    else: prev_m, prev_y_m = mois_sel - 1, annee_sel
    
    df_m1 = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == prev_y_m) & (df_horaire['Date'].dt.month == prev_m)])
    df_act_m1 = df_activite[(df_activite['Date'].dt.year == prev_y_m) & (df_activite['Date'].dt.month == prev_m)]
    
    df_n1 = consolidate_registers(df_horaire[(df_horaire['Date'].dt.year == annee_sel - 1) & (df_horaire['Date'].dt.month == mois_sel)])
    df_act_n1 = df_activite[(df_activite['Date'].dt.year == annee_sel - 1) & (df_activite['Date'].dt.month == mois_sel)]

    # --- KPI HEADER ---
    st.title(f"Cockpit : {noms_mois[mois_sel]} {annee_sel}")
    
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
        <div class="kpi-box">
            <div class="kpi-label">{title}</div>
            <div class="kpi-val">{fmt_val(val, unit)}</div>
            <div class="kpi-delta">
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

    # --- CORPS DE PAGE ---
    c_left, c_right = st.columns([1, 2])

    # GAUCHE : CARTES MATIN / SOIR
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
            ev_cli, cl_cli, _ = calc_evo(cli, cli_m)
            ev_cli_n, cl_cli_n, _ = calc_evo(cli, cli_n)
            ev_pm, cl_pm, _ = calc_evo(pm, pm_m)
            ev_pm_n, cl_pm_n, _ = calc_evo(pm, pm_n)

            st.markdown(f"""
            <div class="act-card">
                <div class="act-title"><span>{icon} {mom.upper()}</span><span class="act-share">{share:.0f}% du CA</span></div>
                <div class="act-grid">
                    <div><div class="act-metric">CA</div><div class="act-val">{ca/1000:.1f}k€</div><div class="act-delta {cl_ca}">M-1 {ev_ca:+.0f}%</div><div class="act-delta {cl_ca_n}">N-1 {ev_ca_n:+.0f}%</div></div>
                    <div><div class="act-metric">Visites</div><div class="act-val">{cli/1000:.1f}k</div><div class="act-delta {cl_cli}"> {ev_cli:+.0f}%</div><div class="act-delta {cl_cli_n}"> {ev_cli_n:+.0f}%</div></div>
                    <div><div class="act-metric">Panier</div><div class="act-val">{pm:.1f}€</div><div class="act-delta {cl_pm}"> {ev_pm:+.0f}%</div><div class="act-delta {cl_pm_n}"> {ev_pm_n:+.0f}%</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # DROITE : CASCADE PONT CA
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
            
            # Top 6 + Autres
            if len(df_b) > 6:
                main = df_b.head(6); other = df_b.iloc[6:]['Delta'].sum()
                deltas = main['Delta'].to_dict(); deltas['Autres'] = other
            else:
                deltas = df_b['Delta'].to_dict()

            measures = ["absolute"] + ["relative"] * len(deltas) + ["absolute"]
            x = [lbl_prev] + list(deltas.keys()) + [lbl_curr]
            y = [start_val] + list(deltas.values()) + [ca_c]
            text = [f"{start_val/1000:.0f}k"] + [f"{v/1000:+.1f}k" for v in deltas.values()] + [f"{ca_c/1000:.0f}k"]
            
            # Scale
            run = [start_val]; cur = start_val
            for v in deltas.values(): cur+=v; run.append(cur)
            max_h = max(run + [ca_c]) * 1.15

            # COULEUR TOTAL = #37474F (Bleu Gris Sombre)
            fig = go.Figure(go.Waterfall(
                orientation="v", measure=measures, x=x, y=y, text=text, textposition="outside",
                connector={"mode":"between", "line":{"width":1, "color":"#555"}},
                decreasing={"marker":{"color":"#FF4444"}}, increasing={"marker":{"color":"#00C851"}}, 
                totals={"marker":{"color":"#37474F"}} # <-- Couleur modifiée
            ))
            fig.update_layout(height=450, showlegend=False, yaxis=dict(range=[0, max_h], title="CA TTC"), margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Données manquantes.")

    st.markdown("---")

    # --- HEATMAP INTELLIGENTE ---
    st.subheader("🔥 Analyse Hebdomadaire & Phrases Clés")

    if not df_curr.empty:
        # GENERATION DE PHRASES (Logique simple)
        df_curr['Day'] = df_curr['Date'].dt.day_name()
        day_stats = df_curr.groupby('Day')['CA TTC'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).fillna(0)
        
        best_day = day_stats.idxmax()
        worst_day = day_stats[day_stats > 0].idxmin() if len(day_stats[day_stats>0]) > 0 else "N/A"
        
        # Traduction
        fr_days_map = {'Monday':'Lundi', 'Tuesday':'Mardi', 'Wednesday':'Mercredi', 'Thursday':'Jeudi', 'Friday':'Vendredi', 'Saturday':'Samedi', 'Sunday':'Dimanche'}
        
        # Comparaison N-1
        df_n1['Day'] = df_n1['Date'].dt.day_name()
        day_stats_n1 = df_n1.groupby('Day')['CA TTC'].sum()
        
        # Phrase 1 : Pic
        phrase_1 = f"🏆 Votre meilleure journée est le **{fr_days_map.get(best_day, best_day)}**."
        
        # Phrase 2 : Variation vs N-1 sur le meilleur jour
        try:
            val_n1 = day_stats_n1.get(best_day, 0)
            val_c = day_stats.get(best_day, 0)
            diff = ((val_c - val_n1)/val_n1)*100 if val_n1 > 0 else 0
            tendance = "en hausse" if diff > 0 else "en baisse"
            phrase_2 = f"Ce jour est **{tendance} de {diff:+.1f}%** par rapport à l'an dernier."
        except:
            phrase_2 = ""

        st.markdown(f"""
        <div class="smart-alert">
            {phrase_1} {phrase_2} <br>
            ❄️ Le jour le plus calme est le <b>{fr_days_map.get(worst_day, worst_day)}</b>.
        </div>
        """, unsafe_allow_html=True)

        # HEATMAP
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fr_days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        all_h = sorted(df_horaire['Heure'].unique())

        def get_hm(df_source):
            df_source['Day'] = df_source['Date'].dt.day_name()
            g = df_source.groupby(['Day', 'Heure'])['CA TTC'].mean().reset_index()
            return g.pivot(index='Day', columns='Heure', values='CA TTC').reindex(index=days, columns=all_h, fill_value=0).fillna(0)

        mat_curr = get_hm(df_curr)
        fig = go.Figure(go.Heatmap(z=mat_curr.values, x=all_h, y=fr_days, colorscale="Blues", xgap=2, ygap=2))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), yaxis_autorange='reversed')
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 2 : FOCUS JOUR / SEMAINE
# ==============================================================================
elif page == "📅 Focus Jour & Semaine":
    st.title("📅 Analyse Opérationnelle")
    
    # Sélecteur Date
    min_date = df_horaire['Date'].min()
    max_date = df_horaire['Date'].max()
    date_focus = st.sidebar.date_input("Choisir un jour", value=max_date, min_value=min_date, max_value=max_date)
    date_focus = pd.to_datetime(date_focus)
    
    # Données du jour
    df_day = consolidate_registers(df_horaire[df_horaire['Date'] == date_focus])
    
    if df_day.empty:
        st.warning("Pas de données pour ce jour (Fermé ?)")
    else:
        # Benchmark : Moyenne des mêmes jours sur 3 mois précédents
        day_name = date_focus.day_name()
        start_bench = date_focus - timedelta(days=90)
        df_bench = df_horaire[(df_horaire['Date'] >= start_bench) & (df_horaire['Date'] < date_focus)]
        df_bench['DayName'] = df_bench['Date'].dt.day_name()
        df_bench = df_bench[df_bench['DayName'] == day_name]
        df_bench = consolidate_registers(df_bench) # Important
        
        # Comparaison KPIs
        ca_d = df_day['CA TTC'].sum()
        ca_b = df_bench.groupby('Date')['CA TTC'].sum().mean() # Moyenne des totaux journaliers
        
        delta_d = ((ca_d - ca_b)/ca_b)*100 if ca_b > 0 else 0
        color = "green" if delta_d > 0 else "red"
        
        st.markdown(f"### Performance du {date_focus.strftime('%d/%m/%Y')} ({day_name})")
        st.markdown(f"CA du jour : **{ca_d:,.0f} €** vs Moyenne habitude : **{ca_b:,.0f} €** (<span style='color:{color}'>{delta_d:+.1f}%</span>)", unsafe_allow_html=True)
        
        # Graphique Comparatif Horaire
        hourly_day = df_day.groupby('Heure')['CA TTC'].sum()
        hourly_bench = df_bench.groupby('Heure')['CA TTC'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_day.index, y=hourly_day.values, name="Ce jour", line=dict(color='#FFD700', width=4)))
        fig.add_trace(go.Scatter(x=hourly_bench.index, y=hourly_bench.values, name="Moyenne habitude", line=dict(color='gray', width=2, dash='dot')))
        fig.update_layout(title="Courbe Horaire vs Habitude", height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Mix Activité du jour
        df_act_day = df_activite[df_activite['Date'] == date_focus]
        if not df_act_day.empty:
            top_act = df_act_day.groupby('ACTIVITE')['CA TTC'].sum().sort_values(ascending=False).head(5)
            st.subheader("Top Activités du Jour")
            st.dataframe(top_act.to_frame().style.format("{:.2f} €"))


# ==============================================================================
# PAGE 3 : TENDANCES & FAMILLES
# ==============================================================================
elif page == "📈 Tendances & Familles":
    st.title("📈 Tendances Long Terme & Familles")
    
    # Filtre Fin de période
    date_end = st.sidebar.date_input("Fin de période", value=df_horaire['Date'].max())
    date_end = pd.to_datetime(date_end)
    date_start = date_end - pd.DateOffset(months=12)
    
    # Données 12 mois
    mask_12m = (df_horaire['Date'] > date_start) & (df_horaire['Date'] <= date_end)
    df_12m = consolidate_registers(df_horaire[mask_12m])
    
    if not df_12m.empty:
        # Agrégation Mensuelle
        df_12m['Mois'] = df_12m['Date'].dt.to_period('M').astype(str)
        monthly = df_12m.groupby('Mois').agg({'CA TTC':'sum', 'Nombre de clients':'sum'}).reset_index()
        monthly['Panier'] = monthly['CA TTC'] / monthly['Nombre de clients']
        
        st.subheader("Historique 12 Mois Glissants")
        
        # 3 Graphiques Lignes
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.line(monthly, x='Mois', y='CA TTC', title="Chiffre d'Affaires", markers=True)
            fig.update_traces(line_color='#29B6F6')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(monthly, x='Mois', y='Nombre de clients', title="Fréquentation", markers=True)
            fig.update_traces(line_color='#66BB6A')
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.line(monthly, x='Mois', y='Panier', title="Panier Moyen", markers=True)
            fig.update_traces(line_color='#FF7043')
            st.plotly_chart(fig, use_container_width=True)
            
        # Stacked Area Activités
        st.subheader("Evolution du Mix Activités")
        mask_act_12m = (df_activite['Date'] > date_start) & (df_activite['Date'] <= date_end)
        df_act_12m = df_activite[mask_act_12m].copy()
        df_act_12m['Mois'] = df_act_12m['Date'].dt.to_period('M').astype(str)
        monthly_act = df_act_12m.groupby(['Mois', 'ACTIVITE'])['CA TTC'].sum().reset_index()
        
        fig = px.area(monthly_act, x='Mois', y='CA TTC', color='ACTIVITE', title="Poids des Activités (Cumulé)")
        st.plotly_chart(fig, use_container_width=True)
        
    # --- FOCUS FAMILLES (Option B) ---
    st.markdown("---")
    st.subheader("🔍 Analyse des Familles (Top 5)")
    
    if not df_famille.empty:
        # Comme on n'a pas forcément le lien Activité <-> Famille dans le fichier Famille,
        # On propose un filtrage par texte ou global
        search = st.text_input("Filtrer par nom de famille (ex: Marlboro, Loto...) - Laisser vide pour Top Global")
        
        # Filtrer sur le mois sélectionné dans la sidebar (pour cohérence) ou dernier mois dispo
        # On va utiliser le mois sélectionné dans le cockpit par défaut si possible, sinon dernier mois
        # Simplification : On prend les données du dernier mois complet dispo dans df_famille
        last_month_fam = df_famille['Date'].max().month
        last_year_fam = df_famille['Date'].max().year
        
        mask_fam_month = (df_famille['Date'].dt.month == last_month_fam) & (df_famille['Date'].dt.year == last_year_fam)
        df_fam_curr = df_famille[mask_fam_month]
        
        if search:
            df_fam_curr = df_fam_curr[df_fam_curr['FAMILLE'].str.contains(search, case=False, na=False)]
            
        # Top 5
        top_fam = df_fam_curr.groupby('FAMILLE')['CA TTC'].sum().sort_values(ascending=False).head(5)
        
        c_fam1, c_fam2 = st.columns([1, 2])
        
        with c_fam1:
            st.markdown(f"**Top 5 Familles ({last_month_fam}/{last_year_fam})**")
            st.dataframe(top_fam)
            
        with c_fam2:
            st.markdown("**Tendances 12 mois (Top 5)**")
            # Pour chaque famille du top 5, on trace une sparkline
            for fam_name in top_fam.index:
                # Get history
                mask_fam_hist = (df_famille['FAMILLE'] == fam_name) & (df_famille['Date'] > date_start)
                hist_data = df_famille[mask_fam_hist].groupby(df_famille['Date'].dt.to_period('M'))['CA TTC'].sum()
                
                # Mini chart
                fig = px.line(x=hist_data.index.astype(str), y=hist_data.values)
                fig.update_layout(height=50, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, showlegend=False)
                fig.update_traces(line_color='#FFD700', line_width=2)
                
                col_a, col_b = st.columns([1, 3])
                col_a.write(f"**{fam_name}**")
                col_b.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Pas de données Familles détectées (Onglet 'ANALYSE FAMILLES' vide ou absent).")
