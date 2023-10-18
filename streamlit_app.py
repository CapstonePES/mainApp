import streamlit as st
from model import model
import requests
import pandas as pd

def main():
    st.title("This will be the dashboard and stuff")
    if st.button("Run Model"):
        with st.spinner("Running Model..."):
            result = model()
        st.success("Model finished running!")
        st.write("Result from model: ",result)
    if st.button("Get data from thingspeak changed"):
        with st.spinner("Getting data..."):
            co_read = "ZKSZLC0A13S5Y2BV"
            co_channel = 2303832
            co_field = 1
            co_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(co_channel, co_field, co_read)
            co_data = requests.get(co_url).json()
            co_data = co_data['feeds']
            co_df = pd.DataFrame(co_data)
            st.dataframe(co_df)


if __name__ == "__main__":
    main()
