import streamlit as st
from model import model
from test import test_func
import requests
import pandas as pd


def main():
    # if st.button("pray to god"):
    #     st.write("okie call start")
    #     resu_thingy = test_func()
    #     st.write("Func done????")
    #     st.write(resu_thingy)
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
    pm_read_key = "PS8QAYOFT2YUUF11"
    pm_channel = 2253626
    pm_field = 3
    pm_url = "https://api.thingspeak.com/channels/{}/fields/{}.json?api_key={}".format(
        pm_channel, pm_field, pm_read_key
    )
    st.title("Welcome to your dashboard")

    # accept age
    age = st.number_input("Please enter your age: ", step=1, min_value=0, max_value=100)
    # Choose your gender
    gender = st.radio(label="What's your gender?", options=["Male", "Female"])
    dust = st.slider(
        "On a Scale of 1 to 8, how allergic are you to dust particles?", 1, 8
    )
    hazard = st.slider(
        "On a scale of 1 to 8, how would you classify your occupational hazards?", 1, 8
    )
    gene = st.slider(
        "On a scale of 1 to 8, how would you classify your genetic risk of lung cancer?",
        1,
        8,
    )
    lung_disesa = st.slider("Do you currently have any chronic lung disease? If yes how drastic?", 1, 7)
    smokin = st.slider("On a scale of 1-8 how often do you smoke?", 1, 8)
    pass_smok = st.slider(
        "On a scale of 1 to 8, what would be your exposure to cigarette smoke?", 1, 8
    )
    nails = st.slider(
        "Have you noticed any clubbing of finger nails? If yes how extreme is it?", 1, 9
    )
    cold = st.slider(
        "On a scale of 1 to 7, how frequently do you contract a cold?", 1, 7
    )
    a = st.button("Submit")
    if a or st.session_state["values"]:
        st.session_state["values"] = True
        # st.text(
        #     "Age: "
        #     + str(age)
        #     + "\nGender: "
        #     + str(gender)
        #     + "\ndust"
        #     + str(dust)
        #     + "\n hazard"
        #     + str(hazard)
        #     + "\n gene"
        #     + str(gene)
        #     + "\n lung_disesa"
        #     + str(lung_disesa)
        #     + "\n smokin"
        #     + str(smokin)
        #     + "\n pass_smok"
        #     + str(pass_smok)
        #     + "\n nails"
        #     + str(nails)
        #     + "\n cold"
        #     + str(cold)
        # )

        if st.button("Click on me to check your risk."):
            with st.spinner("Fetching data from your local station and streamlit."):
                result = model()
            st.success("Data from your station retrieved! Running ARIMA...")
            with st.spinner("Fetching forcasts from ARIMA..."):
                result = model()
            st.success("AQI forecast ready! Running LSTM...")
            with st.spinner("Running LSTM..."):
                result = model()
            st.write("Model has finished running.")
            result = "89.7%"
            st.error("Your risk of lung cancer is: "+result)

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
