import streamlit as st

def loadStyles():
    st.markdown("""
        <style>
            ::-webkit-scrollbar {{
                display: none;
            }}
                
            .stApp {
                padding: 1rem 1rem 1rem 1rem !important;
                margin: 0 !important;
            }
        
            .stMarkdown h1 {
                margin-top: 0rem !important;
                padding-top: 0 !important;

            }
                        
            .stMarkdown p {
                margin-bottom:0;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 1rem 3rem;
                font-size: 1rem;
            }
        
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                width: 100% !important;
            }
                
            div.stContainer > div:first-child {
                padding-top: 0.5rem !important;
            }
            
            div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
                background-color: rgb(255, 255, 255);
            }            
            
            .stSlider, .stSelectbox {
                margin-top: -0.2rem !important;
                margin-bottom: -0.1rem !important;
            }
            
            div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
                background-color: rgb(255, 255, 255);
            }   
            
        </style>
    """, unsafe_allow_html=True)
