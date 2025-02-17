import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import statsapi
import requests
import pybaseball as pyb
import polars as pl
import re
from PIL import Image
from io import BytesIO
from pyfonts import load_font
from bs4 import BeautifulSoup
import time
import math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
import api_scraper
from api_scraper import MLB_Scrape
import streamlit as st
import OSU_Dashboard as dashboard

# Display the app title and description
st.markdown("""
## OSU Dashboard App
##### By: Olav Moeller ([Twitter](https://x.com/OlavMoeller), [LinkedIn](https://linkedin.com/in/olavmoeller))
##### Code: [GitHub Repo](https://github.com/olavmoeller/OSU-Pitching-Graphics/tree/main/Dashboard%20and%20MLB%20Comparison)
#### About
This Streamlit app creates a summary graphic for a selected OSU pitcher, based on their games in StatCast equipped parks.                            
"""
)

all_games = dashboard.get_stat_data(dashboard.osu_games[2024] + dashboard.osu_games[2025])
osu_pitches = all_games[all_games['pitcher_team'] == 'OSU']
osu_pitches.loc[:, 'name_year'] = osu_pitches['pitcher_name'] + ' - ' + osu_pitches['year'].astype(str)
options_list = pd.Series(osu_pitches['name_year'].unique()).sort_values().tolist()

selected_pitcher = st.selectbox('Select pitcher and year', options_list)

pitcher_name = selected_pitcher.split(' - ')[0]
pitcher_year = int(selected_pitcher.split(' - ')[1])

# Button to update plot
if st.button('Update Plot'):
    st.session_state.update_plot = True
    dashboard.pitching_dashboard(pitcher_name, pitcher_year)
