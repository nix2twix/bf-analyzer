import streamlit as st

def loadStyles():
    st.markdown("""
        <style>

            hr {
                margin: 0.5rem 0 !important;
            }
            
            h3 {
                margin-top: 0.5rem !important;
                margin-bottom: 0.5rem !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }

                       
            div.stSlider {
                margin-bottom: 0.5rem !important;
            }
            
            div.stSlider > div {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }
            
            [data-testid="column"] {
                padding: 0 0.5rem !important;
            }
            
            .stDownloadButton {
                margin-top: 0.25rem !important;
                margin-bottom: 0.25rem !important;
            }
            
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
            }
            
            ::-webkit-scrollbar {
                display: none;
            } 
                
            .stApp {
                padding: 1rem 1rem 1rem 1rem !important;
                margin: 0 !important;
            }
        
            .stMarkdown h1 {
                margin-top: 0rem !important;
                padding-top: 0 !important;
            }
                        
            .stMarkdown p {
                margin-bottom: 0;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 1rem 2rem;   
            }
        
            .stTabs [data-baseweb="tab-list"] {
                width: 100% !important;
            }
                
            div.stContainer > div:first-child {
                padding-top: 0.5rem !important;
            }
            
            div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
                background-color: rgb(255, 255, 255);
            }       
                
            div.stSlider label, div.stSlider div[data-testid="stMarkdownContainer"] {
                font-size: 0.8rem; 
                margin-bottom: 0rem;
            }
            
            .stSelectbox {
                margin-top: -0.2rem !important;
                margin-bottom: 1rem !important;
            }
            
            [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p {
                font-size: 0.9rem;
            }
            
            .stButton {
                margin-top: 0rem !important;
                margin-bottom: 0rem !important;
            }
            
            .stFileUploader {
                margin-bottom: 0.5rem !important;
            }

            :root {
                --small-text-size: 0.9rem;
                --small-text-weight: 400;
            }

            /* Вкладки (st.tabs) - текст */
            .stTabs [data-baseweb="tab"] {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
                padding: 0.8rem 1.5rem !important;
            }
            
            .stTabs [data-baseweb="tab"] p {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
                margin: 0 !important;
            }

            /* Кнопки (st.button) */
            .stButton button {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
                padding: 0.4rem 1rem !important;
            }

            /* Segmented Control */
            .stSegmentedControl [role="radio"] {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
                padding: 0.4rem 0.8rem !important;
            }
            
            .stSegmentedControl [role="radio"] span {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Markdown (обычный текст) */
            .stMarkdown p,
            .stMarkdown li,
            .stMarkdown div:not(.st-emotion-cache-*) {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Текст в виджетах */
            .stSelectbox label,
            .stNumberInput label,
            .stTextInput label,
            .stTextArea label,
            .stDateInput label,
            .stTimeInput label,
            .stCheckbox label,
            .stRadio label {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Текст в выпадающих списках и полях ввода */
            .stSelectbox div[data-baseweb="select"] span,
            .stNumberInput input,
            .stTextInput input,
            .stTextArea textarea,
            .stDateInput input,
            .stTimeInput input {
                font-size: var(--small-text-size) !important;
            }

            /* Expander */
            .streamlit-expanderHeader {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Metric */
            .stMetric label {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Информационные сообщения */
            .stAlert p {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Tabs вложенные */
            .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Заголовки */
            .stMarkdown h1 {
                font-size: 2rem !important;
                font-weight: 700 !important;
                margin-top: 0rem !important;
                padding-top: 0 !important;
            }
            
            .stMarkdown h2 {
                font-size: 1.5rem !important;
                font-weight: 600 !important;
            }
            
            .stMarkdown h3 {
                font-size: 1.2rem !important;
                font-weight: 600 !important;
                margin-top: 0.5rem !important;
                margin-bottom: 0.5rem !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }

            /* Адаптация для мобильных устройств */
            @media (max-width: 768px) {
                :root {
                    --small-text-size: 0.8rem;
                }
                
                .stTabs [data-baseweb="tab"] {
                    padding: 0.5rem 0.8rem !important;
                }
                
                .stButton button {
                    padding: 0.3rem 0.8rem !important;
                }
            }

        </style>
    """, unsafe_allow_html=True)

def loadFooter():
    st.markdown("""
    <style>
        .footer > a:link, a:visited {
            color: #24b353;
            background-color: transparent;
            text-decoration: none;
        }
        .footer > a:hover, a:active {
            color: #24b353;
            background-color: transparent;
            text-decoration: none;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0E1117;
            color: white;
            text-align: center;
            padding: 0px 0;
            z-index: 999;
        }
        
        .main .block-container {
            padding-bottom: 50px !important;
        }
    </style>
    <div class="footer">
        <p><a href="https://tulsu.ru/molodezhnye-nauchnye-laboratorii/laboratoriya-kognitivnyh-tehnologij-i-simulyacionnyh-sistem" 
              target="_blank" 
              style='display: block; text-align: center; color: #24b353; text-decoration: none;'>
            &copy; 2026 BioChemTech, TulSU
        </a></p>
    </div>
    """, unsafe_allow_html=True)