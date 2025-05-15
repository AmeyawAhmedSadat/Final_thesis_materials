import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from PIL import Image

# Used species classes in the model
class_names = [
    "badger", "boar", "brown_bear", "hare", "lynx", "musk_deer", "otter",
  "raccoon", "red_fox", "roe_deer_female", "roe_deer_male", "sable",
  "sika_deer_female", "sika_deer_male", "tiger", "ussuri_bear", "wild_cat",
  "yellow_marten"
]

# Load data
@st.cache_data
def load_data():
    data_path = "preprocessed_predictions.csv"  # Relative path to the file
    df = pd.read_csv(data_path)
    # Rest of your preprocessing code...
    
    # Preprocess data
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df['month'] = df['date'].dt.month_name()
    df['hour'] = pd.to_datetime(df['time'].astype(str)).dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    
    
    for class_name in class_names:
        if class_name not in df.columns:
            df[class_name] = 0
    
    # Exclude non-animal columns
    non_animal_columns = [
        "name", "shape", "metadata", "date", "time", "temperature",
        "detected_animals", "animal_counts", "confidence_scores",
        "source", "total_animals", "avg_confidence", "timestamp",
        "month", "hour", "day_of_week"
    ]
    
    # Remove "no animal detected" from species count
    if "no animal detected" in df.columns:
        non_animal_columns.append("no animal detected")
    
    animal_columns = [col for col in class_names if col in df.columns]
    
    return df, animal_columns, non_animal_columns

df, animal_columns, non_animal_columns = load_data()

# Load wildcat image as sideline cover
try:
    wildcat_img = Image.open("wildcat.jpg")
except FileNotFoundError:
    wildcat_img = None

# Add the GSOM image to the sidebar
st.sidebar.image("GSOM.jpg", caption="Master's students of GSOM", use_container_width=True)

# Optional: Add a hyperlink below the image if needed
st.sidebar.markdown("[Click here to learn more about Kedrovaya Pad Nature Reserve](https://leopard-land.ru/territory/kedrpad?lang=en)")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "Overview Dashboard",
    "Species Analysis",
    "Temporal Patterns",
    "Environmental Factors",
    "Population Estimation"
])

# Common filters in sidebar
st.sidebar.header("Global Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=[df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

selected_species = st.sidebar.multiselect(
    "Select Species",
    options=animal_columns,
    default=animal_columns[:3]
)



# Temperature filter (only for Environmental Factors page)
temp_range = None
if page == "Environmental Factors":
    min_temp = float(df['temperature'].min())
    max_temp = float(df['temperature'].max())
    temp_range = st.sidebar.slider(
        "Temperature Range (°C)",
        min_value=min_temp,
        max_value=max_temp,
        value=(min_temp, max_temp)
)

# Apply filters
filtered_df = df[
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
]

# Apply temperature filter if on Environmental Factors page
if page == "Environmental Factors" and temp_range:
    filtered_df = filtered_df[
        (filtered_df['temperature'] >= temp_range[0]) & 
        (filtered_df['temperature'] <= temp_range[1])
]

# Add wildcat image to sidebar
if wildcat_img:
    st.sidebar.image(wildcat_img, caption="Wildcat - Kedrovaya Pad Nature Reserve", use_container_width=True)

# Helper function to convert Interval to string for JSON serialization
def interval_to_str(interval):
    if pd.api.types.is_interval(interval):
        return f"{interval.left:.1f} to {interval.right:.1f}"
    return str(interval)

# Page 1: Overview Dashboard
if page == "Overview Dashboard":
    st.title("🌍 Kedrovaya Pad Nature Reserve Wildlife Monitoring")
    st.subheader("Made by Ahmed & Rakhat")
    
    # Header with park info
    st.markdown("""
    **Kedrovaya Pad Nature Reserve** is a protected area in the Russian Far East known for its biodiversity, 
    particularly its Amur leopard and other rare species populations.
    """)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", len(filtered_df))
    with col2:
        st.metric("Total Detections", filtered_df['total_animals'].sum())
    with col3:
        st.metric("Unique Species", len([col for col in animal_columns if filtered_df[col].sum() > 0]))
    with col4:
        no_detect = filtered_df['no animal detected'].sum() if 'no animal detected' in filtered_df.columns else 0
        st.metric("Empty Images", int(no_detect))
    
    # Detection vs Non-detection chart
    st.subheader("Detection Rate")
    if 'no animal detected' in df.columns:
        detection_data = pd.DataFrame({
            'Type': ['With Animals', 'No Animals'],
            'Count': [
                len(filtered_df) - filtered_df['no animal detected'].sum(),
                filtered_df['no animal detected'].sum()
            ]
        })
        fig_detect = px.bar(
            detection_data,
            x='Type',
            y='Count',
            color='Type',
            labels={'Count': 'Number of Images', 'Type': ''},
            height=400
        )
        st.plotly_chart(fig_detect, use_container_width=True)
    else:
        st.warning("No detection data available for empty images")
    
    # Top row charts
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Species Distribution")
        species_counts = filtered_df[selected_species].sum().sort_values(ascending=False)
        fig1 = px.bar(
            species_counts,
            orientation='h',
            color=species_counts.values,
            color_continuous_scale='Viridis',
            labels={'value': 'Count', 'index': 'Species'},
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Detection Confidence")
        conf_data = filtered_df.melt(
            id_vars=['timestamp'],
            value_vars=selected_species,
            var_name='species',
            value_name='count'
        )
        conf_data = conf_data[conf_data['count'] > 0]
        if not conf_data.empty:
            fig2 = px.box(
                conf_data,
                x='species',
                y='count',
                color='species',
                labels={'species': 'Species', 'count': 'Count'},
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No detection data for selected species")
    
    # Bottom row charts
    st.subheader("Detection Timeline")
    timeline_data = filtered_df.groupby('date')['total_animals'].sum().reset_index()
    fig3 = px.line(
        timeline_data,
        x='date',
        y='total_animals',
        labels={'total_animals': 'Daily Detections', 'date': 'Date'},
        height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

# Page 2: Species Analysis
elif page == "Species Analysis":
    st.title("🦌 Species-Specific Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Species Distribution", "Rare Species", "Co-occurrence"])
    
    with tab1:
        st.subheader("Species Frequency Distribution")
        species_data = filtered_df[selected_species].sum().reset_index()
        species_data.columns = ['Species', 'Count']
        
        fig = px.pie(
            species_data,
            names='Species',
            values='Count',
            hole=0.3,
            color='Species'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            species_data.sort_values('Count', ascending=False),
            hide_index=True,
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Rare Species Detection")
        # Define rare species as those with < 5% of total detections
        total_detections = filtered_df[animal_columns].sum().sum()
        rare_threshold = 0.05 * total_detections
        rare_species = [sp for sp in animal_columns if filtered_df[sp].sum() <= rare_threshold]
        
        if rare_species:
            rare_data = filtered_df.melt(
                id_vars=['date'],
                value_vars=rare_species,
                var_name='species',
                value_name='count'
            )
            rare_data = rare_data[rare_data['count'] > 0]
            
            fig = px.scatter(
                rare_data,
                x='date',
                y='species',
                size='count',
                color='species',
                labels={'species': 'Species', 'date': 'Date'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add rarity score
            st.subheader("Rarity Score")
            rarity_scores = filtered_df[rare_species].sum().reset_index()
            rarity_scores.columns = ['Species', 'Detections']
            rarity_scores['Rarity Score'] = (1 - (rarity_scores['Detections'] / rarity_scores['Detections'].sum())) * 100
            st.dataframe(
                rarity_scores.sort_values('Rarity Score', ascending=False),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No rare species detected in selected timeframe")
    
    with tab3:
        st.subheader("Species Co-occurrence")
        if len(selected_species) >= 2:
            # Create co-occurrence matrix only for selected species
            cooccurrence = filtered_df[selected_species].T.dot(filtered_df[selected_species])
            np.fill_diagonal(cooccurrence.values, 0)  # Remove self-counts
            
            fig = px.imshow(
                cooccurrence,
                labels=dict(x="Species", y="Species", color="Co-occurrences"),
                x=cooccurrence.columns,
                y=cooccurrence.index,
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Network graph of co-occurrence
            st.subheader("Species Interaction Network")
            try:
                import networkx as nx
                G = nx.Graph()
                
                
                for i in range(len(cooccurrence)):
                    for j in range(i+1, len(cooccurrence)):
                        if cooccurrence.iloc[i,j] > 0:
                            G.add_edge(
                                cooccurrence.columns[i],
                                cooccurrence.columns[j],
                                weight=cooccurrence.iloc[i,j]
                            )
                
                if len(G.edges()) > 0:
                    pos = nx.spring_layout(G)
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines')
                    
                    node_x = []
                    node_y = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        hoverinfo='text',
                        marker=dict(
                            showscale=True,
                            colorscale='YlGnBu',
                            size=10,
                            colorbar=dict(
                                thickness=15,
                                title='Node Connections',
                                xanchor='left',
                                title_side='right'
                            )
                        ),
                        text=list(G.nodes())
                    )
                    
                    fig_net = go.Figure(data=[edge_trace, node_trace],
                                     layout=go.Layout(
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20,l=5,r=5,t=40),
                                        height=400))
                    st.plotly_chart(fig_net, use_container_width=True)
                else:
                    st.warning("No co-occurrences found between selected species")
            except ImportError:
                st.warning("Network graph requires networkx package. Install with: pip install networkx")
        else:
            st.warning("Select at least 2 species to analyze co-occurrence")

# Page 3: Temporal Patterns
elif page == "Temporal Patterns":
    st.title("⏰ Temporal Detection Patterns")
    
    tab1, tab2, tab3 = st.tabs(["Daily Patterns", "Seasonal Trends", "Weekly Patterns"])
    
    with tab1:
        st.subheader("Hourly Activity Patterns")
        hourly_data = filtered_df.groupby('hour')[selected_species].sum().reset_index()
        
        fig = px.line(
            hourly_data.melt(id_vars='hour', var_name='species', value_name='count'),
            x='hour',
            y='count',
            color='species',
            labels={'hour': 'Hour of Day', 'count': 'Detections'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add circadian rhythm analysis
        st.subheader("Activity Period Analysis")
        if len(selected_species) > 0:
            circadian_data = []
            for species in selected_species:
                night_hours = list(range(18, 24)) + list(range(0, 6))
                day_hours = list(range(6, 18))
                
                night_detections = filtered_df[filtered_df['hour'].isin(night_hours)][species].sum()
                day_detections = filtered_df[filtered_df['hour'].isin(day_hours)][species].sum()
                
                circadian_data.append({
                    'Species': species,
                    'Day Detections': day_detections,
                    'Night Detections': night_detections,
                    'Nocturnality Index': night_detections / (day_detections + night_detections) if (day_detections + night_detections) > 0 else 0
                })
            
            circadian_df = pd.DataFrame(circadian_data)
            st.dataframe(
                circadian_df.sort_values('Nocturnality Index', ascending=False),
                hide_index=True,
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Monthly Detection Trends")
        monthly_data = filtered_df.groupby('month')[selected_species].sum().reset_index()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
        monthly_data['month'] = pd.Categorical(monthly_data['month'], categories=month_order, ordered=True)
        monthly_data = monthly_data.sort_values('month')
        
        fig = px.bar(
            monthly_data.melt(id_vars='month', var_name='species', value_name='count'),
            x='month',
            y='count',
            color='species',
            barmode='group',
            labels={'month': 'Month', 'count': 'Detections'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Day of Week Patterns")
        dow_data = filtered_df.groupby('day_of_week')[selected_species].sum().reset_index()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data['day_of_week'] = pd.Categorical(dow_data['day_of_week'], categories=dow_order, ordered=True)
        dow_data = dow_data.sort_values('day_of_week')
        
        fig = px.bar(
            dow_data.melt(id_vars='day_of_week', var_name='species', value_name='count'),
            x='day_of_week',
            y='count',
            color='species',
            barmode='group',
            labels={'day_of_week': 'Day of Week', 'count': 'Detections'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 4: Environmental Factors
elif page == "Environmental Factors":
    st.title("🌡️ Environmental Correlations")
    
    tab1, tab2 = st.tabs(["Temperature Effects", "Detection Heatmap"])
    
    with tab1:
        st.subheader("Temperature vs. Species Presence")
        temp_data = filtered_df[['temperature'] + selected_species]
        
        fig = px.scatter(
            temp_data.melt(id_vars='temperature', var_name='species', value_name='count'),
            x='temperature',
            y='count',
            color='species',
            facet_col='species',
            facet_col_wrap=3,
            labels={'temperature': 'Temperature (°C)', 'count': 'Detections'},
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature preference analysis
        st.subheader("Temperature Preferences")
        temp_pref_data = []
        for species in selected_species:
            species_detections = filtered_df[filtered_df[species] > 0]
            if len(species_detections) > 0:
                avg_temp = species_detections['temperature'].mean()
                min_temp = species_detections['temperature'].min()
                max_temp = species_detections['temperature'].max()
                temp_pref_data.append({
                    'Species': species,
                    'Average Temperature': avg_temp,
                    'Min Temperature': min_temp,
                    'Max Temperature': max_temp,
                    'Detection Count': len(species_detections)
                })
        
        if temp_pref_data:
            temp_pref_df = pd.DataFrame(temp_pref_data)
            st.dataframe(
                temp_pref_df.sort_values('Average Temperature'),
                hide_index=True,
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Detection Heatmap by Time and Temperature")
        heatmap_data = filtered_df.copy()
        
        # Convert temperature to bins and then to strings for JSON serialization
        heatmap_data['temp_bin'] = pd.cut(heatmap_data['temperature'], bins=10)
        heatmap_data['temp_bin_str'] = heatmap_data['temp_bin'].apply(interval_to_str)
        
        heatmap_agg = heatmap_data.groupby(['hour', 'temp_bin_str'])['total_animals'].sum().reset_index()
        
        fig = px.density_heatmap(
            heatmap_agg,
            x='hour',
            y='temp_bin_str',
            z='total_animals',
            histfunc="sum",
            nbinsx=24,
            labels={'hour': 'Hour of Day', 'temp_bin_str': 'Temperature Range (°C)', 'total_animals': 'Detections'},
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 5: Population Estimation
elif page == "Population Estimation":
    st.title("🔢 Population Estimation Algorithm")
    
    st.markdown("""
    This algorithm attempts to estimate the minimum number of unique individuals by:
    1. Considering detections of the same species within a short time window as likely the same individual
    2. Adjusting the time threshold to account for animal movement patterns
    3. Using minimum detection requirements to filter false positives
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        time_threshold = st.slider(
            "Time threshold (minutes)",
            min_value=1,
            max_value=120,
            value=30,
            help="How many minutes between detections to consider as a new individual"
        )
    with col2:
        distance_threshold = st.slider(
            "Minimum detections to count",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum number of detections to count as a valid individual"
        )
    
    # Advanced algorithm for population estimation
    def estimate_population(df, time_threshold_min, min_detections):
        results = {}
        
        for species in selected_species:
            # Get all detections of this species
            detections = df[df[species] > 0][['timestamp', species]].copy()
            
            if len(detections) == 0:
                results[species] = 0
                continue
                
            # Sort by timestamp
            detections = detections.sort_values('timestamp')
            
            # Initialize tracking
            individuals = []
            current_individual = {
                'first_seen': detections.iloc[0]['timestamp'],
                'last_seen': detections.iloc[0]['timestamp'],
                'count': 1
            }
            
            for i in range(1, len(detections)):
                time_diff = (detections.iloc[i]['timestamp'] - current_individual['last_seen']).total_seconds() / 60
                
                if time_diff <= time_threshold_min:
                    # Same individual
                    current_individual['last_seen'] = detections.iloc[i]['timestamp']
                    current_individual['count'] += 1
                else:
                    # New individual
                    if current_individual['count'] >= min_detections:
                        individuals.append(current_individual)
                    current_individual = {
                        'first_seen': detections.iloc[i]['timestamp'],
                        'last_seen': detections.iloc[i]['timestamp'],
                        'count': 1
                    }
            
            # Add the last individual
            if current_individual['count'] >= min_detections:
                individuals.append(current_individual)
            
            results[species] = len(individuals)
        
        return pd.DataFrame.from_dict(results, orient='index', columns=['estimated_population']).reset_index().rename(columns={'index': 'species'})
    
    population_estimate = estimate_population(filtered_df, time_threshold, distance_threshold)
    
    st.subheader("Population Estimates")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = px.bar(
            population_estimate,
            x='species',
            y='estimated_population',
            color='species',
            labels={'species': 'Species', 'estimated_population': 'Estimated Individuals'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            population_estimate.sort_values('estimated_population', ascending=False),
            hide_index=True,
            use_container_width=True
        )
    
    st.subheader("Detection Clusters")
    st.markdown("Visualization of detection clusters that were counted as individuals")
    
    for species in selected_species:
        detections = filtered_df[filtered_df[species] > 0][['timestamp', species]].copy()
        if len(detections) > 0:
            detections = detections.sort_values('timestamp')
            detections['time_diff'] = detections['timestamp'].diff().dt.total_seconds() / 60
            detections['cluster'] = (detections['time_diff'] > time_threshold).cumsum()
            
            fig = px.scatter(
                detections,
                x='timestamp',
                y=species,
                color='cluster',
                title=f"{species} Detection Clusters",
                labels={'timestamp': 'Time', 'cluster': 'Cluster ID'},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Kedrovaya Pad Nature Reserve Wildlife Monitoring")
st.sidebar.markdown(f"Data last updated: {datetime.now().strftime('%Y-%m-%d')}")
