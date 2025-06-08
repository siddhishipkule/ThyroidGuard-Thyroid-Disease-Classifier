import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# App setup
st.set_page_config(page_title="ThyroidGuard", layout="wide")
st.title("🧠 ThyroidGuard: Thyroid Disease Classifier")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("new-thyroid.data", header=None)
    df.columns = ['Class', 'T3-resin uptake test', 'Total serum thyroxin',
                  'Total serum triiodothyronine', 'Basal TSH', 'Maximal TSH']
    return df

df = load_data()

# Prepare data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Page: Home
if page == "Home":
    st.subheader("🏠 Welcome to ThyroidGuard")
    st.markdown("""
        This Streamlit app uses a Decision Tree Classifier to predict thyroid diseases 
        based on patient test results.
        
        Navigate using the left sidebar to explore the dataset, view summary stats, graphs, or make predictions!
    """)

# Page: Dataset
elif page == "Dataset":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

# Page: Summary
elif page == "Summary":
    st.subheader("📋 Dataset Summary")
    st.write(df.describe())

    st.subheader("📊 Model Performance")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Page: Graphs
elif page == "Graphs":
    st.subheader("📈 Class Distribution")
    fig, ax = plt.subplots()
    df['Class'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel("Thyroid Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Page: Predict
elif page == "Predict":
    st.subheader("🔮 Make a Prediction")

    input_data = []
    for feature in X.columns:
        val = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
        input_data.append(val)

    if st.button("Predict"):
        result = model.predict([input_data])
        st.success(f"Predicted Thyroid Class: {int(result[0])}")
