import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from data_generator import generate_washington_real_estate_data
from model import train_and_save_model, load_model, predict_price

# Configuration
st.set_page_config(
    page_title="King County Real Estate",
    page_icon="🏠",
    layout="wide"
)

# Titre
st.title("🏠 King County, WA Real Estate Market")
st.markdown("### Analysis of 41 cities in King County, Washington")

# Chargement des données
@st.cache_data
def load_data():
    if os.path.exists('washington_real_estate.csv'):
        df = pd.read_csv('washington_real_estate.csv')
    else:
        with st.spinner('Generating data...'):
            df = generate_washington_real_estate_data(2000)
            df.to_csv('washington_real_estate.csv', index=False)
    
    # Nettoyage
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['age'] = 2024 - df['year_built']
    df['price_per_sqft'] = df['sale_price'] / df['sqft'].replace(0, 1)
    
    return df

# Charger
df = load_data()

# Modèle
@st.cache_resource
def get_model():
    if os.path.exists('models/washington_real_estate_model.pkl'):
        return load_model()
    return None

model = get_model()

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio(
        "Go to:",
        ["Dashboard", "Market Analysis", "City Comparison", "Price Predictor", "Data Explorer"]
    )
    
    st.header("🔍 Filters")
    
    # Villes
    all_cities = sorted(df['city'].unique())
    selected_cities = st.multiselect(
        "Cities",
        all_cities,
        default=['SEATTLE', 'BELLEVUE', 'REDMOND', 'KIRKLAND', 'RENTON']
    )
    
    # Prix
    min_price, max_price = int(df['sale_price'].min()), int(df['sale_price'].max())
    price_range = st.slider(
        "Price Range ($)",
        min_price, max_price,
        (int(df['sale_price'].quantile(0.1)), int(df['sale_price'].quantile(0.9)))
    )
    
    # Superficie
    sqft_range = st.slider(
        "Square Footage",
        int(df['sqft'].min()), int(df['sqft'].max()),
        (1000, 3000)
    )
    
    # Année
    year_range = st.slider(
        "Year Built",
        int(df['year_built'].min()), int(df['year_built'].max()),
        (1950, 2020)
    )
    
    st.header("ℹ️ Info")
    st.info(f"""
    **Data Summary:**
    • {len(df):,} properties
    • {df['city'].nunique()} cities
    • ${df['sale_price'].mean():,.0f} avg price
    • {df['year_built'].min()}-{df['year_built'].max()} year range
    """)

# Filtrer
filtered_df = df[
    (df['city'].isin(selected_cities)) &
    (df['sale_price'].between(price_range[0], price_range[1])) &
    (df['sqft'].between(sqft_range[0], sqft_range[1])) &
    (df['year_built'].between(year_range[0], year_range[1]))
]

# Dashboard
if page == "Dashboard":
    st.subheader("📈 Market Overview")
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = filtered_df['sale_price'].mean()
        st.metric("Average Price", f"${avg_price:,.0f}")
    
    with col2:
        price_psf = filtered_df['price_per_sqft'].mean()
        st.metric("Price per SqFt", f"${price_psf:,.0f}")
    
    with col3:
        avg_size = filtered_df['sqft'].mean()
        st.metric("Avg Size", f"{avg_size:,.0f} sqft")
    
    with col4:
        count = len(filtered_df)
        st.metric("Properties", f"{count:,}")
    
    # Graphiques
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Prix par ville
        city_prices = filtered_df.groupby('city')['sale_price'].mean().sort_values(ascending=False).head(10)
        fig1 = px.bar(
            x=city_prices.index,
            y=city_prices.values,
            title="Top 10 Cities by Average Price",
            labels={'x': 'City', 'y': 'Average Price ($)'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_right:
        # Distribution des prix
        fig2 = px.histogram(
            filtered_df,
            x='sale_price',
            nbins=50,
            title="Price Distribution",
            labels={'sale_price': 'Sale Price ($)'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Carte
    st.subheader("📍 Property Locations")
    
    if len(filtered_df) > 1000:
        map_data = filtered_df.sample(1000)
    else:
        map_data = filtered_df
    
    fig_map = px.scatter_mapbox(
        map_data,
        lat="latitude",
        lon="longitude",
        color="sale_price",
        size="sqft",
        hover_name="city",
        hover_data=["beds", "bath_full", "grade"],
        color_continuous_scale=px.colors.sequential.Viridis,
        zoom=9,
        height=500,
        title="Property Map"
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

# Market Analysis
elif page == "Market Analysis":
    st.subheader("📊 Market Analysis")
    
    # Tendances temporelles
    st.write("### Price Trends Over Time")
    
    if 'sale_year' in filtered_df.columns:
        yearly = filtered_df.groupby('sale_year').agg({
            'sale_price': ['mean', 'count']
        }).round(0)
        yearly.columns = ['avg_price', 'count']
        yearly = yearly.reset_index()
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Scatter(
                x=yearly['sale_year'],
                y=yearly['avg_price'],
                name='Average Price',
                line=dict(color='blue', width=3)
            ),
            secondary_y=False
        )
        
        fig_trend.add_trace(
            go.Bar(
                x=yearly['sale_year'],
                y=yearly['count'],
                name='Sales Count',
                opacity=0.3
            ),
            secondary_y=True
        )
        
        fig_trend.update_layout(
            title='Price Trends and Sales Volume',
            xaxis_title='Year'
        )
        
        fig_trend.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Sales Count", secondary_y=True)
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Facteurs de prix
    st.write("### Price Factors")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        # Taille vs Prix
        fig_size = px.scatter(
            filtered_df,
            x='sqft',
            y='sale_price',
            color='city',
            title='Size vs Price',
            labels={'sqft': 'Square Footage', 'sale_price': 'Price ($)'}
        )
        st.plotly_chart(fig_size, use_container_width=True)
    
    with col_f2:
        # Grade vs Prix
        fig_grade = px.box(
            filtered_df,
            x='grade',
            y='sale_price',
            title='Construction Grade vs Price',
            labels={'grade': 'Grade', 'sale_price': 'Price ($)'}
        )
        st.plotly_chart(fig_grade, use_container_width=True)
    
    # Statistiques
    st.write("### Market Statistics")
    
    stats_cols = ['sale_price', 'sqft', 'sqft_lot', 'beds', 'bath_full', 'grade', 'price_per_sqft']
    stats_df = filtered_df[stats_cols].describe()
    
    st.dataframe(
        stats_df.style.format({
            'sale_price': '${:,.0f}',
            'price_per_sqft': '${:,.0f}',
            'sqft': '{:,.0f}',
            'sqft_lot': '{:,.0f}'
        }),
        use_container_width=True
    )

# City Comparison
elif page == "City Comparison":
    st.subheader("🏙️ City Comparison")
    
    # Sélection
    cities_to_compare = st.multiselect(
        "Select cities to compare:",
        sorted(df['city'].unique()),
        default=['SEATTLE', 'BELLEVUE', 'REDMOND', 'KIRKLAND']
    )
    
    if cities_to_compare:
        compare_df = df[df['city'].isin(cities_to_compare)]
        
        # Métriques par ville
        city_stats = compare_df.groupby('city').agg({
            'sale_price': ['mean', 'median', 'count'],
            'sqft': 'mean',
            'price_per_sqft': 'mean',
            'beds': 'mean',
            'grade': 'mean'
        }).round(0)
        
        city_stats.columns = ['Avg_Price', 'Median_Price', 'Count', 'Avg_SqFt', 'Avg_Price_SqFt', 'Avg_Beds', 'Avg_Grade']
        city_stats = city_stats.reset_index()
        
        # Affichage
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.dataframe(
                city_stats.style.format({
                    'Avg_Price': '${:,.0f}',
                    'Median_Price': '${:,.0f}',
                    'Avg_Price_SqFt': '${:,.0f}',
                    'Avg_SqFt': '{:,.0f}'
                }),
                use_container_width=True
            )
        
        with col_c2:
            fig_compare = px.bar(
                city_stats,
                x='city',
                y='Avg_Price',
                color='Count',
                title='Average Price by City',
                labels={'Avg_Price': 'Average Price ($)'}
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # Box plots
        st.write("### Price Distribution Comparison")
        
        fig_box = px.box(
            compare_df,
            x='city',
            y='sale_price',
            title='Price Distribution by City',
            points=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Scatter matrix
        st.write("### Property Characteristics")
        
        fig_scatter = px.scatter_matrix(
            compare_df,
            dimensions=['sale_price', 'sqft', 'beds', 'grade', 'year_built'],
            color='city',
            title='Property Characteristics Matrix'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# Price Predictor
elif page == "Price Predictor":
    st.subheader("🤖 Price Predictor")
    
    if model is None:
        st.warning("Model not trained. Training model...")
        with st.spinner("Training model..."):
            result = train_and_save_model()
            if result:
                model = result['model']
                st.success("Model trained successfully!")
            else:
                st.error("Failed to train model")
                st.stop()
    
    st.markdown("""
    Estimate the price of a property based on its characteristics.
    """)
    
    # Formulaire
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox("City", sorted(df['city'].unique()))
            sqft = st.number_input("Square Footage", 500, 10000, 2000, 100)
            sqft_lot = st.number_input("Lot Size (sqft)", 0, 100000, 5000, 500)
            beds = st.slider("Bedrooms", 1, 6, 3)
            bath_full = st.slider("Full Bathrooms", 1, 4, 2)
        
        with col2:
            bath_3qtr = st.slider("3/4 Bathrooms", 0, 2, 0)
            bath_half = st.slider("Half Bathrooms", 0, 2, 1)
            grade = st.slider("Construction Grade (1-13)", 1, 13, 7)
            condition = st.slider("Condition (1-5)", 1, 5, 3)
            year_built = st.number_input("Year Built", 1800, 2024, 1990)
        
        # Options
        col3, col4 = st.columns(2)
        with col3:
            greenbelt = st.checkbox("Greenbelt View")
            view_rainier = st.checkbox("Mt. Rainier View")
        with col4:
            view_water = st.checkbox("Water View")
            stories = st.slider("Stories", 1, 4, 2)
        
        submit = st.form_submit_button("Estimate Price")
    
    if submit:
        # Données d'entrée
        input_data = {
            'city': city,
            'sqft': sqft,
            'sqft_lot': sqft_lot,
            'beds': beds,
            'bath_full': bath_full,
            'bath_3qtr': bath_3qtr,
            'bath_half': bath_half,
            'grade': grade,
            'condition': condition,
            'year_built': year_built,
            'stories': stories,
            'greenbelt': 1 if greenbelt else 0,
            'view_rainier': 1 if view_rainier else 0,
            'view_otherwater': 1 if view_water else 0
        }
        
        # Coordonnées par défaut
        if city == 'SEATTLE':
            input_data['latitude'], input_data['longitude'] = 47.6062, -122.3321
        elif city == 'BELLEVUE':
            input_data['latitude'], input_data['longitude'] = 47.6101, -122.2015
        else:
            input_data['latitude'], input_data['longitude'] = 47.5, -122.3
        
        # Prédiction
        try:
            predicted_price = predict_price(model, input_data)
            
            # Résultat
            st.success(f"## Estimated Price: **${predicted_price:,.0f}**")
            
            # Détails
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                price_psf = predicted_price / sqft
                st.metric("Price per SqFt", f"${price_psf:,.0f}")
            
            with col_r2:
                city_avg = df[df['city'] == city]['sale_price'].mean()
                diff = ((predicted_price - city_avg) / city_avg) * 100
                st.metric(
                    "vs City Average",
                    f"${city_avg:,.0f}",
                    delta=f"{diff:.1f}%"
                )
            
            with col_r3:
                if predicted_price < 500000:
                    segment = "Budget"
                elif predicted_price < 1000000:
                    segment = "Mid-Range"
                elif predicted_price < 2000000:
                    segment = "Premium"
                else:
                    segment = "Luxury"
                st.metric("Market Segment", segment)
            
            # Propriétés similaires
            st.subheader("📊 Similar Properties")
            
            similar = df[
                (df['city'] == city) &
                (df['sqft'].between(sqft * 0.8, sqft * 1.2)) &
                (df['beds'] == beds)
            ].head(5)
            
            if len(similar) > 0:
                st.dataframe(
                    similar[['city', 'sqft', 'beds', 'bath_full', 'grade', 'year_built', 'sale_price']]
                    .style.format({
                        'sale_price': '${:,.0f}',
                        'sqft': '{:,.0f}'
                    }),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Data Explorer
elif page == "Data Explorer":
    st.subheader("🔍 Data Explorer")
    
    # Statistiques
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    
    with col_e1:
        st.metric("Total Properties", f"{len(df):,}")
    
    with col_e2:
        st.metric("Cities", f"{df['city'].nunique()}")
    
    with col_e3:
        st.metric("Avg Price", f"${df['sale_price'].mean():,.0f}")
    
    with col_e4:
        st.metric("Date Range", f"{df['sale_year'].min()}-{df['sale_year'].max()}")
    
    # Aperçu
    st.write("### Data Preview")
    
    # Sélection colonnes
    all_cols = df.columns.tolist()
    default_cols = ['city', 'sale_price', 'sqft', 'beds', 'bath_full', 'grade', 'year_built', 'sale_date']
    selected_cols = st.multiselect("Select columns:", all_cols, default=default_cols)
    
    if selected_cols:
        # Tri
        sort_col = st.selectbox("Sort by:", selected_cols)
        sort_asc = st.checkbox("Ascending", value=True)
        
        # Données
        preview = df[selected_cols].sort_values(sort_col, ascending=sort_asc).head(100)
        
        # Formatage
        def format_val(val):
            if isinstance(val, (int, float)):
                if val > 1000:
                    return f"${val:,.0f}" if 'price' in str(val).lower() or 'val' in str(val).lower() else f"{val:,.0f}"
            return val
        
        # Affichage
        st.dataframe(preview, use_container_width=True)
        
        # Téléchargement
        csv = preview.to_csv(index=False)
        st.download_button(
            "Download as CSV",
            csv,
            "king_county_data.csv",
            "text/csv"
        )
    
    # Distribution villes
    st.write("### City Distribution")
    
    city_counts = df['city'].value_counts().reset_index()
    city_counts.columns = ['City', 'Count']
    
    fig_cities = px.bar(
        city_counts.head(20),
        x='City',
        y='Count',
        title='Top 20 Cities by Property Count'
    )
    st.plotly_chart(fig_cities, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("King County Real Estate Analytics • v1.0")