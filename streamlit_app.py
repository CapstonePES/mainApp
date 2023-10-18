import streamlit as st
from model import model

def main():
    st.title("This will be the dashboard and stuff")
    if st.button("Run Model"):
        with st.spinner("Running Model..."):
            result = model()
        st.success("Model finished running!")
        st.write("Result from model: ",result)

if __name__ == "__main__":
    main()
