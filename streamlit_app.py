import streamlit as st
from model import model
import requests
import pandas as pd

def main():
    co_df = pd.DataFrame()
    pm_df = pd.DataFrame()
    st.title("This will be the dashboard and stuff")
    if st.button("Run Model"):
        with st.spinner("Running Model..."):
            result = model()
        st.success("Model finished running!")
        st.write("Result from model: ",result)
    if st.button("Get data from thingspeak"):
        with st.spinner("Getting data..."):
            co_read = "ZKSZLC0A13S5Y2BV"
            co_channel = 2303832
            co_field = 1
            co_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(co_channel, co_field, co_read)
            co_data = requests.get(co_url).json()
            co_data = co_data['feeds']
            co_df = pd.DataFrame(co_data)
            pm_read = "PS8QAYOFT2YUUF11"
            pm_channel = 2253626
            pm_field = 3
            pm_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(pm_channel, pm_field, pm_read)
            pm_data = requests.get(pm_url).json()
            pm_data = pm_data['feeds']
            pm_df = pd.DataFrame(pm_data)
    if co_df.empty == False and pm_df.empty == False:
        co_col, pm_col = st.columns(2)
        with co_col:
            st.dataframe(co_df)
        with pm_col:
            st.dataframe(pm_df)

if __name__ == "__main__":
    main()
