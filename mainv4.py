######################### librairies 
from contextlib import contextmanager
from typing import Generator, Sequence
import pandas as pd
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

######################### Style de page 
st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('<style>div.block-container{padding-top:1px;}</style>', unsafe_allow_html=True)

######################### importation/nettoyage data
data = pd.read_csv('data.csv')

numeric_columns = ['CA VENTE', 'CA ACHAT', 'MARGE']
data[numeric_columns] = data[numeric_columns].replace(',', '.', regex=True)

######################### Widget (filtre)
st.sidebar.header('Filtres')
month_mapping = {
    1: 'Janvier',
    2: 'Février',
    3: 'Mars',
    4: 'Avril',
    5: 'Mai',
    6: 'Juin',
    7: 'Juillet',
    8: 'Août',
    9: 'Septembre',
    10: 'Octobre',
    11: 'Novembre',
    12: 'Décembre'
}
selected_month = st.sidebar.selectbox('Mois', ['ALL'] + list(month_mapping.values()))
selected_year = st.sidebar.selectbox('Année', ['ALL'] + list(data['ANNEE'].dropna().unique().astype(str)))
selected_product = st.sidebar.multiselect('Produit', ['ALL'] + list(data['DESIGNATION'].dropna().unique()), default='ALL')
selected_canal = st.sidebar.multiselect('Organisation Commerciale', ['ALL'] + list(data['CANAL'].dropna().unique()), default='ALL')

if selected_month == 'ALL':
    filtered_data = data.copy()
else:
    selected_month_key = list(month_mapping.keys())[list(month_mapping.values()).index(selected_month)]
    filtered_data = data[data['MOIS'] == selected_month_key]

if 'ALL' not in selected_canal:
    filtered_data = filtered_data[filtered_data['CANAL'].isin(selected_canal)]

selected_central = st.sidebar.multiselect('Central', options=list(filtered_data['CENTRAL'].dropna().unique()))
if selected_central:
    filtered_data = filtered_data[filtered_data['CENTRAL'].isin(selected_central)]

if selected_year != 'ALL':
    filtered_data = filtered_data[filtered_data['ANNEE'] == int(selected_year)]

if 'ALL' not in selected_product:
    filtered_data = filtered_data[filtered_data['DESIGNATION'].isin(selected_product)]

######################### Titre

######################### Titre interactif
selected_month_text = f"du mois de {selected_month}" if selected_month != 'ALL' else "de l'année 2023"
selected_canal_text = f"pour l'organisation commerciale {selected_canal}" if selected_canal != 'ALL' else "pour toutes les organisations commerciales"

st.markdown(f'<h1 style="font-size: 30px;">Analyse des ventes {selected_month_text} {selected_canal_text}</h1>', unsafe_allow_html=True)
add_vertical_space(1)

######################### CA
sales_by_month = filtered_data['CA VENTE'].astype(float).sum()
sales_by_month_formatted = "{:,.0f}".format(sales_by_month).replace(",", " ")

if selected_month == 'ALL':
    previous_month = None
else:
    previous_month_index = list(month_mapping.values()).index(selected_month) - 1
    previous_month = list(month_mapping.values())[previous_month_index]

if previous_month is not None:
    previous_month_sales = data[data['MOIS'].map(month_mapping) == previous_month]['CA VENTE'].astype(float).sum()
    if previous_month_sales != 0:
        variation = (sales_by_month - previous_month_sales) / previous_month_sales * 100
    else:
        variation = 0
else:
    variation = 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<h2 style='font-size: 20px;'>Chiffre d'affaires</h2>", unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size: 40px; color: #BF9D7A;">{sales_by_month_formatted} €</h1>', unsafe_allow_html=True)
    if variation > 0:
        st.markdown(f'<h2 style="font-size: 16px; color: #55efc4;">↑ +{variation:.0f}% <span style="color: #ffffff; font-weight: 100; font-size: 12px;">par rapport au mois précédent</span></h2>', unsafe_allow_html=True)
    elif variation < 0:
        st.markdown(f'<h2 style="font-size: 16px; color: #ff7675;">↓ {variation:.0f}% <span style="color: #ffffff; font-weight: 100; font-size: 12px;">par rapport au mois précédent</span></h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="font-size: 14px;"></h2>', unsafe_allow_html=True)

######################### Prix de revient
sales_by_month_achat = filtered_data['CA ACHAT'].astype(float).sum()
sales_by_month_achat_formatted = "{:,.0f}".format(sales_by_month_achat).replace(",", " ")

if previous_month is not None:
    previous_month_sales_achat = data[data['MOIS'].map(month_mapping) == previous_month]['CA ACHAT'].astype(float).sum()
    if previous_month_sales_achat != 0:
        variation_achat = (sales_by_month_achat - previous_month_sales_achat) / previous_month_sales_achat * 100
    else:
        variation_achat = 0
else:
    variation_achat = 0

with col2:
    st.markdown('<h2 style="font-size: 20px; ">Prix de revient</h2>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size: 40px; color: #D4DCA9;">{sales_by_month_achat_formatted} €</h1>', unsafe_allow_html=True)
    if variation_achat > 0:
        st.markdown(f'<h2 style="font-size: 16px; color: #55efc4;">↑ +{variation_achat:.0f}% <span style="color: #ffffff; font-weight: 100; font-size: 12px;">par rapport au mois précédent</span></h2>', unsafe_allow_html=True)
    elif variation_achat < 0:
        st.markdown(f'<h2 style="font-size: 16px; color: #ff7675;">↓ {variation_achat:.0f}% <span style="color: #ffffff; font-weight: 100; font-size: 12px;">par rapport au mois précédent</span></h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="font-size: 14px;"></h2>', unsafe_allow_html=True)

######################### Marge brute
sales_by_month_marge = filtered_data['MARGE'].astype(float).sum()
sales_by_month_marge_formatted = "{:,.0f}".format(sales_by_month_marge).replace(",", " ")
if previous_month is not None:
    previous_month_sales_marge = data[data['MOIS'].map(month_mapping) == previous_month]['MARGE'].astype(float).sum()
    if previous_month_sales_marge != 0:
        variation_marge = (sales_by_month_marge - previous_month_sales_marge) / previous_month_sales_marge * 100
    else:
        variation_marge = 0
else:
    variation_marge = 0
with col3:
    st.markdown('<h2 style="font-size: 20px;">Marge Brute</h2>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size: 40px; color: #EBF2EA;">{sales_by_month_marge_formatted} €</h1>', unsafe_allow_html=True)
    if variation_marge > 0:
        st.markdown(f'<h2 style="font-size: 16px; color: #55efc4;">↑ +{variation_marge:.0f}% <span style="color: #ffffff; font-weight: 100; font-size: 12px;"> par rapport au mois précédent</span></h2>', unsafe_allow_html=True)
    elif variation_marge < 0:
        st.markdown(f'<h2 style="font-size: 16px; color: #ff7675;">↓ {variation_marge:.0f}% <span style="color: #ffffff; font-weight: 100; font-size: 12px;"> par rapport au mois précédent</span></h2>' , unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="font-size: 14px;"></h2>', unsafe_allow_html=True)

add_vertical_space(1)

#########################  Onglet avec graphique
@contextmanager
def chart_container(
    tabs: Sequence[str] = (
        "Classement des organisations commerciales par chiffre d'affaires",
        "Classement des produits par chiffre d'affaires",
        "Répartition du chiffre d'affaires par organisation commerciale",
        "Évolution du chiffre d'affaires mensuel et prévisions futures",
        "Diagramme de dispersion avec une ligne de régression pour le chiffre d'affaires et la marge"
    ),
) -> Generator:
    tab_1, tab_2, tab_3,tab_4,tab_5 = st.tabs(tabs)

######################### Chart 1 ( qu'on retrouve dans la fonction tout en bas)
    with tab_1:
        yield

######################### Chart 2
    with tab_2:
        def generate_top5_chart2(data):
            data = data.dropna(subset=['CA VENTE'])
            top_produits = data.groupby("DESIGNATION")["CA VENTE"].sum().reset_index()
            top_produits = top_produits.nlargest(5, "CA VENTE")
            top_produits = top_produits.sort_values("CA VENTE", ascending=False)
            chart = alt.Chart(top_produits).mark_bar(color='#eaa56f').encode(
                x=alt.X('DESIGNATION', axis=alt.Axis(title='produits')),
                y=alt.Y('CA VENTE', axis=alt.Axis(title="chiffre d'affaires")),
                tooltip=['DESIGNATION', 'CA VENTE'],
                text="CA VENTE"
            ).properties(
                width=alt.Step(40),
                height=350
            )
            st.altair_chart(chart, use_container_width=True)
        generate_top5_chart2(filtered_data)

######################### Chart 3
    with tab_3:
        def generate_ca_vente_by_canal(data, seuil):
            data = data.dropna(subset=['CA VENTE'])
            ca_vente_by_canal = data.groupby("CANAL")["CA VENTE"].sum().reset_index()
            ca_total = ca_vente_by_canal["CA VENTE"].sum()
            ca_vente_by_canal.loc[ca_vente_by_canal["CA VENTE"] < (seuil * ca_total), "CANAL"] = "Autres"
            colors = ["#eaa56f", "#74b9ff", "#f07070","#0eaf7d"]
            fig = go.Figure(data=go.Pie(labels=ca_vente_by_canal["CANAL"], values=ca_vente_by_canal["CA VENTE"],marker=dict(colors=colors)))
            fig.update_layout(width=400, height=350,margin=dict(l=20, r=20, t=20, b=20))  
            st.plotly_chart(fig, use_container_width=True)
        seuil = 0.05  
        generate_ca_vente_by_canal(filtered_data, seuil)

######################### Chart 4
    with tab_4:
        def generate_ca_vente_by_month(data):
            data['ANNEE'] = pd.to_numeric(data['ANNEE'], errors='coerce')
            data['MOIS'] = pd.to_numeric(data['MOIS'], errors='coerce')
            filtered_data = data[data['MOIS'] <= 5]
            ca_vente_by_month = filtered_data.groupby('MOIS')['CA VENTE'].sum().reset_index()
            X = np.array(ca_vente_by_month['MOIS']).reshape(-1, 1)
            y = np.array(ca_vente_by_month['CA VENTE'])
            reg = LinearRegression().fit(X, y)
            forecast_months = np.arange(6, 13).reshape(-1, 1)
            ca_vente_forecast = reg.predict(forecast_months)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ca_vente_by_month['MOIS'], y=ca_vente_by_month['CA VENTE'], name="chiffre d'affaires",marker=dict(color='#eaa56f')))
            fig.add_trace(go.Scatter(x=ca_vente_by_month['MOIS'], y=reg.predict(X), name='Tendance', mode='lines', marker=dict(color='#0eaf7d')))
            fig.add_trace(go.Bar(x=forecast_months.flatten(), y=ca_vente_forecast, name='Prévision',marker=dict(color='#f07070')))
            fig.update_xaxes(title_text='Mois')
            fig.update_yaxes(title_text="chiffre d'affaires ")
            fig.update_layout(barmode='group', legend=dict(orientation='h'),height=350,margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig)
        generate_ca_vente_by_month(filtered_data)
            
    with tab_5:  # Changez le numéro de l'onglet en fonction de votre code
        def generate_scatterplot(data):
            # Conversion des colonnes en numérique
            data['CA VENTE'] = pd.to_numeric(data['CA VENTE'], errors='coerce')
            data['MARGE'] = pd.to_numeric(data['MARGE'], errors='coerce')

            # Création du diagramme de dispersion
            fig = px.scatter(data, x='CA VENTE', y='MARGE', trendline="ols", hover_data=['DESIGNATION','NUM COMMANDE'])

            # Mise à jour du layout
            fig.update_layout(
                title='Relation entre le chiffre d\'affaires et la marge',
                xaxis_title='Chiffre d\'affaires',
                yaxis_title='Marge',
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig)
        generate_scatterplot(filtered_data)
        
######################### Chart 1
def graph():
    with chart_container():
        def generate_top5_chart(data):
            data['CA VENTE'] = pd.to_numeric(data['CA VENTE'], errors='coerce')
            data = data.dropna(subset=['CA VENTE'])
            top_produits = data.groupby("CENTRAL")["CA VENTE"].sum().reset_index()
            top_produits = top_produits.nlargest(5, "CA VENTE")
            chart = alt.Chart(top_produits).mark_bar(color='#0eaf7d').encode(
                x=alt.X('CENTRAL', axis=alt.Axis(title='organisation commerciale')),
                y=alt.Y('CA VENTE', axis=alt.Axis(title="chiffre d'affaires")),
                tooltip=['CENTRAL', 'CA VENTE']
            ).properties(
                width=alt.Step(40),
                height=350
            )
            st.altair_chart(chart, use_container_width=True)
        generate_top5_chart(filtered_data)
graph()
