import streamlit as st
from model import model
import requests
import pandas as pd

def main():
    co_df = pd.DataFrame()
    pm_df = pd.DataFrame()
    co_read_key = "ZKSZLC0A13S5Y2BV"
    co_channel = 2303832
    co_field = 1
    co_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(co_channel, co_field, co_read_key)
    pm_read_key = "PS8QAYOFT2YUUF11"
    pm_channel = 2253626
    pm_field = 3
    pm_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(pm_channel, pm_field, pm_read_key)
    st.title("This will be the dashboard and stuff")

    # accept age 
    age = st.number_input("Enter your age: ",step=1,min_value=0,max_value=100)
    # Choose your gender
    gender = st.radio(label="gender",options=["Male","Female"])
    dust = st.slider("How allergic to dust?", 1, 8)
    hazard = st.slider("How allergic to hazard?", 1, 8)
    gene = st.slider("How allergic to gene?", 1, 8)
    lung_disesa = st.slider("How allergic to lung?", 1, 7)
    smokin = st.slider("How allergic to smok?", 1, 8)
    pass_smok = st.slider("How allergic to pass smoke?", 1, 8)
    nails = st.slider("How allergic to nail?", 1, 9)
    cold = st.slider("How allergic to cold?", 1, 7)
    a = st.button("Submit")
    if a or st.session_state['values']:
        st.session_state['values']=True
        st.text("Age: "+str(age)+"\nGender: "+str(gender)+ "\ndust"+str(dust)+"\n hazard"+str(hazard)+"\n gene"+str(gene)+"\n lung_disesa"+str(lung_disesa)+"\n smokin"+str(smokin)+"\n pass_smok"+str(pass_smok)+"\n nails"+str(nails)+"\n cold"+str(cold))

        if st.button("Run Model"):
            with st.spinner("Running Model..."):
                result = model()
            st.success("Model finished running!")
            st.write("Result from model: ",result)

        if st.button("Get data from thingspeak"):
            with st.spinner("Getting data..."):
                co_data = requests.get(co_url).json()
                co_data = co_data['feeds']
                co_df = pd.DataFrame(co_data)
                
                pm_data = requests.get(pm_url).json()
                pm_data = pm_data['feeds']
                pm_df = pd.DataFrame(pm_data)

        if co_df.empty == False and pm_df.empty == False:
            co_col, pm_col = st.columns(2)

            with co_col:
                st.title("CO Data")
                st.dataframe(co_df)
                
            with pm_col:
                st.title("PM Data")
                st.dataframe(pm_df)

if __name__ == "__main__":
    main()
