import streamlit as st
from model import model
from arima import arima
from lstm import lstm
import requests
import pandas as pd


def main():
    if "values" not in st.session_state:
        st.session_state["values"] = False
    co_df = pd.DataFrame()
    pm_df = pd.DataFrame()
    co_read_key = "ZKSZLC0A13S5Y2BV"
    co_channel = 2303832
    co_field = 1
    co_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(
        co_channel, co_field, co_read_key
    )
    pm_read_key = "G1PW99IDKTRCSMZM"
    pm_channel = 2253626
    pm_field = 3
    pm_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(
        pm_channel, pm_field, pm_read_key
    )
    st.title("Get your lung cancer predictions here!")
    age = st.number_input("Please enter your age: ", step=1, min_value=0, max_value=100)
    gender = st.radio(label="Please select your gender?", options=["Male", "Female"])
    dust = st.slider(
        "On a Scale of 0 to 10, how allergic are you to dust particles?", 0, 10
    )
    hazard = st.slider(
        "On a scale of 0 to 10, how would you classify your occupational hazards?", 0, 10
    )
    gene = st.slider(
        "On a scale of 0 to 10, how would you classify your genetic risk of lung cancer?",
        0,
        10,
    )
    lung_disesa = st.slider("Do you currently have any chronic lung disease? If yes how drastic?", 0, 10)
    smokin = st.slider("On a scale of 1-10 how often do you smoke?", 0, 10)
    pass_smok = st.slider(
        "On a scale of 0 to 10, what would be your exposure to cigarette smoke?", 0, 10
    )
    nails = st.slider(
        "Have you noticed any clubbing of finger nails? If yes how extreme is it?", 0, 10
    )
    cold = st.slider(
        "On a scale of 0 to 10, how frequently do you contract a cold?", 0, 10
    )
    a = st.button("Submit")
    if a or st.session_state["values"]:
        st.session_state["values"] = True
        if st.button("Click on me to check your risk."):
            with st.spinner("Fetching data from your local station and streamlit."):
                co_data = requests.get(co_url).json()
                co_data = co_data['feeds']
                co_df = pd.DataFrame(co_data)
                pm_data = requests.get(pm_url).json()
                pm_data = pm_data['feeds']
                pm_df = pd.DataFrame(pm_data)
            st.success("Data from your station retrieved! Running ARIMA...")
            with st.spinner("Fetching forcasts from ARIMA..."):
                result = arima(co_df,pm_df)
            st.write(result)
            st.success("AQI forecast ready! Running LSTM...")
            with st.spinner("Running LSTM..."):
                result = lstm([result,age/10, dust, hazard, gene, lung_disesa, smokin, pass_smok, nails, cold])/100
            st.write("Model has finished running.")
            if result >0.4 and result > 0.75:
                st.error("Your lung cancer incidence rate is: "+str(result))
            elif result<0.4:
                st.success("Your lung cancer incidence rate is: "+str(result))
            else:
                st.warning("Your lung cancer incidence rate is: "+str(result))

        if st.button("Get data from thingspeak"):
            with st.spinner("Getting data..."):
                co_data = requests.get(co_url).json()
                st.write(co_data)
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
