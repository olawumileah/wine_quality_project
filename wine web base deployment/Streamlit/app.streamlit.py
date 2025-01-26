import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model_XGBClassifier.h5')

#Define the app
st.title('Wine Quality Classifier')
st.write('Enter the wine composition below to predict the quality')

#input fields for user data
fixed_acidity = st.number_input("fixed_acidity", min_value= 3.0, max_value= 16.0)
volatile_acidity = st.number_input("volatile_acidity",  min_value= 0.05, max_value= 1.63 )
critic_acid = st.number_input("critic_acid", min_value= 0.00, max_value= 1.76)
residual_sugar = st.number_input("residual_sugar",  min_value= 0.5, max_value= 66.8 )
chlorides = st.number_input("chlorides",  min_value= 0.008, max_value= 0.617 )
free_sulfurdioxide = st.number_input("free_sulfurdioxide",  min_value= 1.0, max_value= 290.0)
total_sulfurdioxide = st.number_input("total_sulfurdioxide",  min_value= 5.5 , max_value= 447.0)
density = st.number_input('density',  min_value= 0.98711, max_value= 1.03898 )
pH = st.number_input('pH',  min_value= 2.70 , max_value= 4.02)
sulphate = st.number_input("sulphate",  min_value= 0.20 , max_value= 2.5 )
alcohol = st.number_input("alcohol",  min_value= 7.5, max_value= 15.0)
type = st.number_input("type",  min_value= 0, max_value= 1)

#predict button
if st.button("Predict"):
    #Prepare input data
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, critic_acid, residual_sugar, chlorides, free_sulfurdioxide, total_sulfurdioxide, density, pH, sulphate, alcohol, type ]],
                             columns= ['fixed_acidity', 'volatile_acidity', 'critic_acid', 'residual_sugar', 'chlorides', 'free_sulfurdioxide', 'total_sulfurdioxide', 'density', 'pH', 'sulphate', 'alcohol', 'type' ])
    #Make prediction
    prediction = model.predict(input_data)
    quality = [3, 4, 5, 6, 7, 8, 9]
    st.success(f"The predicted quality is: {quality[prediction[0]]}")