import streamlit as st

def about_func():
    st.markdown("# About 🌱")
    st.sidebar.markdown("# About 🌱")
    #st.write("This is a simple Streamlit application with login authentication.")
    #st.subheader("Smart Plant Health")
    #st.subheader("Automated Plant Disease Detection Application With Chatbot")

    st.markdown("""
    ## Overview
    **Smart Plant Health** is an advanced application designed to automate the detection and classification of plant diseases using images of leaves. Developed by R.K.A. Oshadha Wijerathne from the Department of Computer Science at the University of Moratuwa, this innovative solution leverages machine learning and image analysis to assist farmers and agricultural professionals in identifying and managing plant diseases promptly and efficiently.

    ## Key Features
    1. **Automated Disease Detection**:
       - Utilizes a robust machine learning model, likely a Convolutional Neural Network (CNN), trained on a comprehensive dataset of labeled leaf images.
       - Capable of accurately identifying various plant diseases, enabling timely interventions to minimize crop damage.

    2. **User-Friendly Web Application**:
       - Allows users to upload images of plant leaves directly from their devices.
       - Provides real-time disease classification results from the machine learning model.
       - Designed for intuitive use, making it accessible to farmers and agricultural professionals.

    3. **Interactive Chatbot**:
       - Enhances user interaction by providing information and solutions related to plant diseases.
       - Answers questions about disease symptoms, causes, and offers recommendations for treatment and prevention.
       - Facilitates text-based conversations for easy access to information and support.

    ## Project Objectives
    - **Develop a Robust Machine Learning Model**: Create an accurate CNN model trained on a large dataset of labeled leaf images to detect and classify plant diseases.
    - **Create a User-Friendly Web Application**: Design an intuitive application for users to upload leaf images and receive disease classification results.
    - **Implement a Chatbot Feature**: Integrate a chatbot to provide personalized assistance and enhance user interaction.

    ## Significance
    The significance of Smart Plant Health lies in its ability to facilitate early disease detection, reducing crop losses and improving agricultural productivity. By automating the disease detection process, the application helps farmers take prompt action to prevent the spread of diseases, ultimately contributing to sustainable farming practices and enhanced food security.

    ## Impact
    Smart Plant Health empowers users with timely and accurate disease identification, enabling informed decision-making for disease control and treatment strategies. By reducing reliance on chemical pesticides and promoting environmentally friendly practices, the application plays a crucial role in advancing sustainable agriculture and ensuring food security.
    """)

    st.markdown("---")
    st.markdown("For more information on Smart Plant Health, please contact:")
    st.markdown("""
    **R.K.A. Oshadha Wijerathne**  
    Department of Computer Science  
    University of Moratuwa
    """)
