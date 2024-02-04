import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import smtplib
from email.mime.text import MIMEText

# Load the dataset
file_path = 'D:\\hackathon nminms 2.0\\classData.csv'
data = pd.read_csv(file_path)

# Combine fault indicator columns to create 'Fault_Type'
data['Fault_Type'] = data[['G', 'C', 'B', 'A']].astype(str).agg(''.join, axis=1)
fault_types = {
    '0000': 'No Fault',
    '1000': 'Ground Fault',
    '0100': 'Fault in Line A',
    '0010': 'Fault in Line B',
    '0001': 'Fault in Line C',
    '1001': 'LG Fault (Between Phase A and Ground)',
    '1010': 'LG Fault (Between Phase B and Ground)',
    '1100': 'LG Fault (Between Phase C and Ground)',
    '0011': 'LL Fault (Between Phase B and Phase A)',
    '0110': 'LL Fault (Between Phase C and Phase B)',
    '0101': 'LL Fault (Between Phase C and Phase A)',
    '1100': 'LG Fault (Between Phase C and Ground)',
    '1010': 'LG Fault (Between Phase B and Ground)',
    '1001': 'LG Fault (Between Phase A and Ground)',
    '1011': 'LLG Fault (Between Phases A, B, and Ground)',
    '1110': 'LLG Fault (Between Phases A, C, and Ground)',
    '1101': 'LLG Fault (Between Phases C, B, and Ground)',
    '0111': 'LLL Fault (Between all three phases)',
    '1111': 'LLLG Fault (Three-phase symmetrical fault)',
    '1011': 'Line A Line B to Ground Fault'
}

data['Fault_Type'] = data['Fault_Type'].map(fault_types)

# Feature selection
features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
X = data[features]
y = LabelEncoder().fit_transform(data['Fault_Type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training and evaluation with Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Fault Detection")

# User Input
st.sidebar.header("User Input Features")

# Collect user input for feature values using sliders
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.slider(f"Select {feature}", min_value=float(X[feature].min()),
                                            max_value=float(X[feature].max()), value=float(X[feature].median()))

# Transform user input into a DataFrame
user_df = pd.DataFrame([user_input])

# Make prediction with the model
prediction = model.predict(user_df)[0]

# Display Prediction
st.subheader("Prediction:")
st.write(f"The predicted fault type is: {prediction}")

# Create tabs for different sections
tabs = ["Visualizations", "Fault Information"]
selected_tab = st.sidebar.radio("Select Section", tabs)

if selected_tab == "Visualizations":
    # Visualizations
    st.header("Data Analysis and Visualizations")

    # Pie chart of Fault Types distribution in the dataset
    fault_counts = data['Fault_Type'].value_counts()
    fig_pie = px.pie(fault_counts, values=fault_counts.values, names=fault_counts.index, title='Fault Type Distribution')
    st.plotly_chart(fig_pie)

    # Feature Importance Plot
    feature_importance = model.feature_importances_
    fig_feature_importance = px.bar(x=features, y=feature_importance, title='Feature Importance')
    st.plotly_chart(fig_feature_importance)

elif selected_tab == "Fault Information":
    # Information about the selected fault type
    st.header("Fault Information")

    fault_info = f"Fault Type: {prediction}"
    st.write(fault_info)

 # Email notification
    st.subheader("Email Notification")

    email_address = st.text_input("Enter your email address:")
    if st.button("Send Email Notification"):
        try:
            # Configure your email server and credentials
            smtp_server = "your_smtp_server"
            smtp_port = 587
            email_sender = "your_email@gmail.com"
            email_password = "your_email_password"

            # Create message
            subject = "Fault Detected Notification"
            body = f"Fault Type Detected: {prediction}\nDescription: {fault_types.get(prediction, 'Unknown')}"
            message = MIMEText(body)
            message["Subject"] = subject
            message["From"] = email_sender
            message["To"] = email_address

            # Connect to the SMTP server and send the email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(email_sender, email_password)
                server.sendmail(email_sender, [email_address], message.as_string())

            st.success("Email notification sent successfully!")

        except Exception as e:
            st.error(f"An error occurred while sending the email: {e}")
