import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry
import joblib
st.set_page_config(page_title="Healthcare Data Visualization", layout="wide")

if 'data_loaded' not in st.session_state or st.session_state.data_loaded == False:
    st.session_state.df = pd.read_csv(r"Cleaned.csv")
    
    cols = list(st.session_state.df.columns)
    cols.insert(0, cols.pop(cols.index('Under-five mortality rate (probability of dying by age 5 per 1000 live births)')))
    st.session_state.df = st.session_state.df[cols]
    
    st.session_state.model = joblib.load(r"rfr_healthcare.joblib")

    
df = st.session_state.df

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Overview", "Graphs", "Model"))

if page == "Overview":
    
    st.markdown("Under-five mortality remains a critical public health issue, particularly in low- and middle-income countries. Despite global efforts, millions of children under five die each year due to preventable causes. Understanding the relationship between the under-five mortality rate and other factors is essential to achieving the UN 2030 SDG agenda, particularly SDG 3, specifically SDG 3.2. Analyzing these factors is vital for identifying and targeting countries that lag behind, allowing for the allocation of necessary resources and the implementation of targeted interventions and recommendations. This dashboard serves as an important tool for policymakers, enabling them to visualize the relationship between the under-five mortality rate and other factors. It also provides an opportunity for interactive predictions using the random forest model. Our dataset consists of 4,246 observations, retrieved from WHO and Vizhub.")
    st.dataframe(df)
    
    st.markdown('Mohamad EL Moussawi, Jana EL Masri, Rola EL Jeaid')
elif page == "Graphs":
    st.sidebar.header("Filters")
    countries = st.sidebar.multiselect("Select Countries", options=df['Location'].unique())
    years = st.sidebar.slider("Select Year Range", min_value=int(df['Period'].min()), max_value=int(df['Period'].max()), value=(int(df['Period'].min()), int(df['Period'].max())))

    # Filter the data based on user input
    filtered_df = df[(df['Location'].isin(countries)) & (df['Period'].between(years[0], years[1]))]

    # Get columns for selectboxes, excluding 'Location'
    selectable_columns = [col for col in df.columns if col != 'Location']

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time Series Plot")
        y_column = st.selectbox("Select Y-axis variable", options=selectable_columns)
        fig = px.line(filtered_df, x='Period', y=y_column, color='Location')
        st.plotly_chart(fig)

   
    with col2:
        st.subheader("Box Plot")
        box_x = st.selectbox("Select variable for boxplot", options=selectable_columns, key='box')
        fig = px.box(filtered_df, x=box_x, color='Location')
        st.plotly_chart(fig)



    st.subheader("Scatter Plot with Trendline")
    trend_x = st.selectbox("Select X-axis variable for trendline plot", options=selectable_columns, key = 'x2')
    trend_y = st.selectbox("Select Y-axis variable for trendline plot", options=selectable_columns, key = 'y2')
    fig = px.scatter(filtered_df, 
                     x=trend_x,
                     y=trend_y,
                     color='Location', 
                     size='Period', 
                     hover_name='Location',
                     size_max=8,
                     trendline="ols")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig)


    # Choropleth map
    st.subheader("Choropleth Map")
    map_variable = st.selectbox("Select variable for map", options=selectable_columns)
    map = px.choropleth(data_frame=df,
                    locations='iso',
                    color=map_variable,
                    color_continuous_scale='Viridis',
                    projection='natural earth',
                    labels={'Value': 'Value'},
                    width=2200, height=1000,
                    hover_name='Location',  # Add this line to show Location on hover
                    hover_data=[map_variable])  # Add this line to show the selected variable value on hover

    map.update_layout(title_text=f'{map_variable} by Country')
    st.plotly_chart(map)


    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].drop(columns=['Period']).corr()

    heatmap = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.index,
                        y=corr_matrix.columns,
                        color_continuous_scale='Viridis')

    heatmap.update_layout(
        height=1500,
        xaxis=dict(tickangle=90, tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18))
    )

    st.plotly_chart(heatmap)



elif page == "Model":
    st.title("Healthcare Data Model")
    st.write("Enter values for the following features to get a prediction:")

    # Create input fields for the selected features
    features = [
        'Prevalence of underweight among adults, BMI < 18 (age-standardized estimate) (%)',
        'Prevalence of anaemia in children aged 6-59 months (%)_x',
        'Hepatitis B (HepB3) immunization coverage among 1-year-olds (%)',
        'Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)',
        'Polio (Pol3) immunization coverage among 1-year-olds (%)',
        'Prevalence of anaemia in pregnant women (aged 15-49) (%)',
        'Prevalence of anaemia in children aged 6-59 months (%)_y',
        'HIV/AIDS',
        'Diarrheal diseases',
        'Measles'
    ]

    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(feature, value=0.0)

    # Create a button to trigger the prediction
    if st.button("Predict"):
        # Create a DataFrame with the input values
        input_df = pd.DataFrame([input_data])

        # Make prediction using the loaded model
        prediction = st.session_state.model.predict(input_df)

        # Display the prediction
        st.write(f"Predicted Under-five mortality rate: {prediction[0]:.2f}")




