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
import google_colab_selenium as gs
import api_scraper
from api_scraper import MLB_Scrape

stratum = load_font(font_url="https://github.com/ccheney/chromotion/blob/master/assets/fonts/stratum2-medium-webfont.ttf?raw=true")

# Set the theme for seaborn plots
sns.set_theme(style='whitegrid',
              palette='deep',
              font='DejaVu Sans',
              font_scale=1.5,
              color_codes=True,
              rc=None)

plt.rcParams['figure.dpi'] = 300

### PITCH COLORS ###
pitch_colors = {
    ## Fastballs ##
    'FF': {'color': '#FF007D', 'name': '4-Seam Fastball'},
    'FA': {'color': '#FF007D', 'name': 'Fastball'},
    'SI': {'color': '#98165D', 'name': 'Sinker'},
    'FC': {'color': '#BE5FA0', 'name': 'Cutter'},

    ## Offspeed ##
    'CH': {'color': '#F79E70', 'name': 'Changeup'},
    'FS': {'color': '#FE6100', 'name': 'Splitter'},
    'SC': {'color': '#F08223', 'name': 'Screwball'},
    'FO': {'color': '#FFB000', 'name': 'Forkball'},

    ## Sliders ##
    'SL': {'color': '#67E18D', 'name': 'Slider'},
    'ST': {'color': '#1BB999', 'name': 'Sweeper'},
    'SV': {'color': '#376748', 'name': 'Slurve'},

    ## Curveballs ##
    'KC': {'color': '#311D8B', 'name': 'Knuckle Curve'},
    'CU': {'color': '#3025CE', 'name': 'Curveball'},
    'CS': {'color': '#274BFC', 'name': 'Slow Curve'},
    'EP': {'color': '#648FFF', 'name': 'Eephus'},

    ## Others ##
    'KN': {'color': '#867A08', 'name': 'Knuckleball'},
    'PO': {'color': '#472C30', 'name': 'Pitch Out'},
    'UN': {'color': '#9C8975', 'name': 'Unknown'},
}

# Create a dictionary mapping pitch types to their colors
dict_color = dict(zip(pitch_colors.keys(), [pitch_colors[key]['color'] for key in pitch_colors]))

# Create a dictionary mapping pitch types to their names
dict_pitch = dict(zip(pitch_colors.keys(), [pitch_colors[key]['name'] for key in pitch_colors]))

# Create a list of OSU game ids
osu_games = {
    2024: [763702,763704,763697,763701],
    2025: [796298,796296,796293,796291,795107,795103,795104,791896,791894,791892]}

# Creating a function that can return the full dataframe for any set of games
def get_stat_data(gamelist):
    # Activating the scraper
    scraper = MLB_Scrape()

    # Getting the game data for the requested games, making it a pandas dataframe
    game_data = scraper.get_data(game_list_input=gamelist)
    data_df = scraper.get_data_df(data_list=game_data)
    df = data_df.to_pandas()

    # Adding columns for relevant pitching results
    df['in_zone'] = (df['zone'] < 10)
    df['out_zone'] = (df['zone'] > 10)
    df['chase'] = (df.in_zone==False) & (df.is_swing)

    # Adding a year column
    df['year'] = pd.to_datetime(df['game_date']).dt.year

    return df

# Creating a function that gets only pitches thrown by a selected pitcher over a selected year
def player_year_data(playername, year):
    year_df = get_stat_data(osu_games[year])
    return year_df[year_df['pitcher_name'] == playername]

# Aggregating relevant metrics for our OSU pitcher to find pitch classification averages
def gen_grouping(df):
    group_df = df.groupby(['pitcher_name','pitcher_hand','year','pitch_type']).agg(
                        pitch = ('pitch_type','count'),  # Count of pitches
                        start_speed = ('start_speed','mean'),  # Average start speed
                        ivb = ('ivb','mean'),  # Average vertical movement
                        hb = ('hb','mean'),  # Average horizontal movement
                        spin_rate = ('spin_rate','mean'),  # Average spin rate
                        spin_axis = ('spin_direction','mean'),  # Average spin axis
                        x0 = ('x0','mean'),  # Average horizontal release position
                        z0 = ('z0','mean'),  # Average vertical release position
                        extension = ('extension','mean'),  # Average release extension
                        swing = ('is_swing','sum'),  # Total swings
                        whiff = ('is_whiff','sum'),  # Total whiffs
                        in_zone = ('in_zone','sum'),  # Total in-zone pitches
                        out_zone = ('out_zone','sum'),  # Total out-of-zone pitches
                        chase = ('chase','sum'),  # Total chases
                        ).reset_index()
    return group_df

# Importing the data from 2020-2024 mlb pitch level data
mlbpd = pd.read_csv('mlb_pitch_data_2020_2024.csv')

# Adding a year column to the dataframe
mlbpd['year'] = pd.to_datetime(mlbpd['game_date']).dt.year

# Adding columns for relevant pitching results
mlbpd['in_zone'] = (mlbpd['zone'] < 10)
mlbpd['out_zone'] = (mlbpd['zone'] > 10)
mlbpd['chase'] = (mlbpd.in_zone==False) & (mlbpd.is_swing)

# Aggregating the relevant metrics for all MLB pitchers by year to account for pitch changes between years
mlbpdall = gen_grouping(mlbpd)

# Defining a command that will return our selected pitcher's OSU roster page
def get_player_link(playername, year):
    # URL of the OSU Beavers baseball roster page
    url = 'https://osubeavers.com/sports/baseball/roster/' + str(year) + '/'

    # Defining a function using Selenium to scroll the page, so it loads every player (original URL stops after loading 30)
    driver = gs.Chrome()
    def scroll_and_scrape(url, scroll_pause_time=1):
        driver.get(url)
        # Get scroll height
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load page
            time.sleep(scroll_pause_time)
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        # Now that the page is fully scrolled, grab the source
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup
    soup = scroll_and_scrape(url)

    link = url.removesuffix('/sports/baseball/roster/' + str(year) + '/') + soup.find(attrs={"aria-label": re.compile(playername)})['href']
    return link

# Defining a function that will return our selected pitcher's OSU player ID
def get_player_id(playername, link):
    id = link.removeprefix('https://osubeavers.com/sports/baseball/roster/' + playername.lower().split(' ',)[0] + '-' + playername.lower().split(' ',)[1] + '/')
    return id

def get_headshot(link, ax):
    # Using the players link to create a soup object
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Finding the headshot on the page
    pic_link = soup.find(loading="eager", class_="block aspect-[2/3] h-full w-full max-w-[120px] md:max-w-[180px]")['src']
    # Making the headshot a plottable image
    pic_response = requests.get(pic_link)
    img = Image.open(BytesIO(pic_response.content))
    # Creating the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    ax.imshow(img, extent=[0, 1, 0, 1.5], origin='upper')
    ax.axis('off')

def player_bio(playername, year, link, ax):
    # Using the players link to create a soup object
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Determining pitcher handedness
    if soup.find("dt", string="Position: ").find_parent().get_text().split(': ')[1].split('-')[0] == "Right":
      pitcher_hand = 'RHP'
    else:
      pitcher_hand = 'LHP'
    # Calling pitcher class
    pitcher_class = soup.find("dt", string="Class: ").find_parent().get_text()
    # Calling height/weight
    height = soup.find("dt", string="Height: ").find_parent().get_text().split(': ')[1]
    weight = soup.find("dt", string="Weight: ").find_parent().get_text().split(': ')[1]
    # Display the graphic
    ax.text(0.5, 1, f'{playername}', va='top', ha='center', fontsize=56, font=stratum)
    ax.text(0.5, 0.70, f'{pitcher_hand}, {pitcher_class}, {height}/{weight}', va='top', ha='center', fontsize=30, font=stratum)
    ax.text(0.5, 0.50, f'Season Pitching Summary', va='top', ha='center', fontsize=50, font=stratum)
    ax.text(0.5, 0.25, f'{year} NCAA D1 Baseball Season', va='top', ha='center', fontsize=30, fontstyle='italic', font=stratum)
    ax.text(0.5, 0.1, f'OSU Statcast Data From 2/21 - 2/25/2024', va='top', ha='center', fontsize=20, font=stratum)
    ax.axis('off')

def logo(ax):
    # Using the logo from the baseball website, but storing it here so we don't have to scrape as it will be the same for each player
    logo_link = 'https://dxbhsrqyrr690.cloudfront.net/sidearm.nextgen.sites/oregonstate.sidearmsports.com/images/logos/site/site.png'
    # Making the logo a plottable image
    logo_response = requests.get(logo_link)
    img = Image.open(BytesIO(logo_response.content))
    # Creating the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.imshow(img, extent=[0, 1, 0, 1], origin='upper')
    ax.axis('off')

def break_plot(playername, year, ax):
    # Defining our dataframe by the selected pitcher
    df = player_year_data(playername, year)

    # Check if the pitcher throws with the right hand
    if df['pitcher_hand'].values[0] == 'R':
        sns.scatterplot(ax=ax,
                        x=df['hb'],
                        y=df['ivb'],
                        hue=df['pitch_type'],
                        palette=dict_color,
                        ec='black',
                        alpha=1,
                        zorder=2)

    # Check if the pitcher throws with the left hand
    if df['pitcher_hand'].values[0] == 'L':
        sns.scatterplot(ax=ax,
                        x=df['hb']*-1,
                        y=df['ivb'],
                        hue=df['pitch_type'],
                        palette=dict_color,
                        ec='black',
                        alpha=1,
                        zorder=2)

    # Draw horizontal and vertical lines at y=0 and x=0 respectively
    ax.axhline(y=0, color='#808080', alpha=0.5, linestyle='--', zorder=1)
    ax.axvline(x=0, color='#808080', alpha=0.5, linestyle='--', zorder=1)

    # Set the labels for the x and y axes
    ax.set_xlabel('Horizontal Break (in)', font=stratum, fontsize=16)
    ax.set_ylabel('Induced Vertical Break (in)', font=stratum, fontsize=16)

    # Set the title of the plot
    ax.set_title("Pitch Breaks", font=stratum, fontsize=20)

    # Remove the legend
    ax.get_legend().remove()

    # Set the tick positions and labels for the x and y axes
    ax.set_xticks(range(-20, 21, 10))
    ax.set_xticklabels(range(-20, 21, 10), font=stratum, fontsize=15)
    ax.set_yticks(range(-20, 21, 10))
    ax.set_yticklabels(range(-20, 21, 10), font=stratum, fontsize=15)

    # Set the limits for the x and y axes
    ax.set_xlim((-25, 25))
    ax.set_ylim((-25, 25))

    # Add text annotations based on the pitcher's throwing hand
    if df['pitcher_hand'].values[0] == 'R':
        ax.text(-21.5, -24.2, s='Glove Side', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), font=stratum, fontsize=10, zorder=3)
        ax.text(-24.2, -24.2, s='← ', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=9, zorder=3)
        ax.text(21.5, -24.2, s='Arm Side', fontstyle='italic', ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), font=stratum, fontsize=10, zorder=3)
        ax.text(22.7, -24.2, s=' →', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=9, zorder=3)

    if df['pitcher_hand'].values[0] == 'L':
        ax.invert_xaxis()
        ax.text(21.5, -24.2, s='Arm Side', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), font=stratum, fontsize=10, zorder=3)
        ax.text(24.2, -24.2, s='← ', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=9, zorder=3)
        ax.text(-21.5, -24.2, s='Glove Side', fontstyle='italic', ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), font=stratum, fontsize=10, zorder=3)
        ax.text(-22.7, -24.2, s=' →', fontstyle='italic', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black'), fontsize=9, zorder=3)

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Format the x and y axis tick labels as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

# Defining a function that finds the pitch after any given pitch, and determines whether a strike or ball was thrown if the at bat continued
def after(df, balls, strikes):
    # Calling all pitches in a specified count
    pitch = df[(df['strikes'] == strikes ) & (df['balls'] == balls)].reset_index()
    # Calling the pitch after the original pitch for the specified count
    after_index = [x+1 for x in df[(df['strikes'] == strikes ) & (df['balls'] == balls)].index]
    # If the inning ended on a certain count, the next pitch would be out of our data, as it wasn't by our OSU pitcher. This loop controls for that
    try:
        pitch_after = df.loc[after_index].reset_index()
    except KeyError:
        same_ab_index = [x for x in after_index if x in df.index]
        pitch_after = df.loc[same_ab_index].reset_index()
    ball_after = 0
    strike_after = 0
    # Running a loop to sum up the number of strikes and balls after our selected count
    for x in pitch_after.index:
        if pitch_after['strikes'].loc[x] == pitch_after['balls'].loc[x] == 0:
            pass
        elif pitch_after['strikes'].loc[x] == pitch['strikes'].loc[x] + 1:
            strike_after = strike_after + 1
        elif pitch_after['balls'].loc[x] == pitch['balls'].loc[x] + 1:
            ball_after = ball_after + 1
        else:
            pass
    return pd.DataFrame({'ball': [ball_after], 'strike': [strike_after]})

# Defining a function that plots a pie chart based on a specified count, determining how many of each pitch type was thrown
def pitch_pie(df, balls, strikes, ax):
    # Finding the ratio of each pitch type
    pitch = df[(df['strikes'] == strikes ) & (df['balls'] == balls)]['pitch_type'].value_counts(normalize=True)
    # If the specified count never happened, display a gray pie chart
    if len(pitch) == 0:
        patches, lists = ax.pie([1], colors=['#808080'])
        [p.set_zorder(10) for p in patches]
    # Otherwise, make the pie chart have colors corresponding to each pitch
    else:
        patches, lists = ax.pie(pitch.to_list(), colors=pitch.index.map(dict_color))
        [p.set_zorder(10) for p in patches]
    # Making the titles for each count
    if balls + strikes == 3:
        ax.set_title(f'{balls}-{strikes}', font=stratum, fontsize=20, loc='left')
    else:
        ax.set_title(f'{balls}-{strikes}', font=stratum, fontsize=20)
    ax.axis('equal')

# Creating the function that calls the chart
def plinko_chart(playername, year, fig, ax, gs, gs_x, gs_y):
    # Assigning our dataframe by relevant pitcher
    df = player_year_data(playername, year)
    # Creating a grid for the pie charts to be placed in
    inner_grid = gridspec.GridSpecFromSubplotSpec(8, 5, subplot_spec=gs[gs_x[0]:gs_x[-1], gs_y[0]:gs_y[-1]])
    # Making a dictionary of where to plot each pie chart
    count_plot_loc = {
        ## Top Row
        (0,0): fig.add_subplot(inner_grid[1, 2]),
        ## Second Row
        (0,1): fig.add_subplot(inner_grid[2, 1]),
        (1,0): fig.add_subplot(inner_grid[2, 3]),
        ## Third Row
        (0,2): fig.add_subplot(inner_grid[3, 0]),
        (1,1): fig.add_subplot(inner_grid[3, 2]),
        (2,0): fig.add_subplot(inner_grid[3, 4]),
        ## Fourth Row
        (1,2): fig.add_subplot(inner_grid[5, 0]),
        (2,1): fig.add_subplot(inner_grid[5, 2]),
        (3,0): fig.add_subplot(inner_grid[5, 4]),
        ## Fifth Row
        (2,2): fig.add_subplot(inner_grid[6, 1]),
        (3,1): fig.add_subplot(inner_grid[6, 3]),
        ## Sixth Row
        (3,2): fig.add_subplot(inner_grid[7, 2])}
    # Defining every possible count
    count_list = [(balls,strikes) for balls in [0,1,2,3] for strikes in [0,1,2]]
    # Creating an empty list that will contain each line between plots
    line_list = []
    # Defining the style of each line
    kw = dict(linestyle="-", color="black", zorder=5)
    # Noting the total number of atbats not ending in the first pitch, as a baseline for how big our lines should be
    tot_abs = after(df,0,0).sum(axis=1,numeric_only=True)[0]
    # Iterating the creation of the pie charts and lines along each possible count
    for (balls, strikes) in count_list:
        pitch_pie(df, balls, strikes, count_plot_loc[(balls,strikes)])
        if (balls,strikes+1) in count_list:
           line_list.append(ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA=count_plot_loc[(balls,strikes)].transData, coordsB=count_plot_loc[(balls,strikes+1)].transData, **kw, linewidth=10*after(df, balls,strikes)['strike'].loc[0]/tot_abs))
        if (balls+1,strikes) in count_list:
           line_list.append(ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA=count_plot_loc[(balls,strikes)].transData, coordsB=count_plot_loc[(balls+1,strikes)].transData, **kw, linewidth=10*after(df, balls,strikes)['ball'].loc[0]/tot_abs))
        for line in line_list:
            ax.add_artist(line)
    # Hiding axis text
    ax.axis('off')
    # Setting the title
    ax.set_title('Pitch Sequencing', font=stratum, fontsize=20)
    # Set a label underneath the plot
    count_plot_loc[(3,2)].set_xlabel('Line Thickness = Amount of Pitches',fontsize=15, font=stratum)

def velocity_chart(playername, year, fig, ax, gs, gs_x, gs_y):
    # Assigning the dataframe relevant to our selected pitcher
    df = player_year_data(playername, year)

    # Get the count of each pitch type and sort them in descending order
    sorted_value_counts = df['pitch_type'].value_counts().sort_values(ascending=False)

    # Get the list of pitch types ordered from most to least frequent
    items_in_order = sorted_value_counts.index.tolist()

    # Turn off the axis and set the title for the main plot
    ax.axis('off')
    ax.set_title('Pitch Velocity Distribution', font=stratum, fontsize=20)

    # Create a grid for the inner subplots
    inner_grid_1 = gridspec.GridSpecFromSubplotSpec(len(items_in_order), 1, subplot_spec=gs[gs_x[0]:gs_x[-1], gs_y[0]:gs_y[-1]])
    ax_top = []

    # Create subplots for each pitch type
    for inner in inner_grid_1:
        ax_top.append(fig.add_subplot(inner))
    ax_number = 0
    for i in items_in_order:
        # Check if all release speeds for the pitch type are the same
        if np.unique(df[df['pitch_type'] == i]['start_speed']).size == 1:
            # Plot a single line if all values are the same
            ax_top[ax_number].plot([np.unique(df[df['pitch_type'] == i]['start_speed']),
                              np.unique(df[df['pitch_type'] == i]['start_speed'])], [0, 1], linewidth=4,
                              color=dict_color[df[df['pitch_type'] == i]['pitch_type'].values[0]], zorder=20)
        else:
            # Plot the KDE for the release speeds
            sns.kdeplot(df[df['pitch_type'] == i]['start_speed'], ax=ax_top[ax_number], fill=True,
                  clip=(df[df['pitch_type'] == i]['start_speed'].min(), df[df['pitch_type'] == i]['start_speed'].max()),
                  color=dict_color[df[df['pitch_type'] == i]['pitch_type'].values[0]])

        # Plot the mean release speed for the OSU data
        df_average = df[df['pitch_type'] == i]['start_speed']
        ax_top[ax_number].plot([df_average.mean(), df_average.mean()],
                      [ax_top[ax_number].get_ylim()[0], ax_top[ax_number].get_ylim()[1]],
                      color=dict_color[df[df['pitch_type'] == i]['pitch_type'].values[0]],
                      linestyle='--')

        # Plot the mean release speed for the 2020-2024 MLB Average Data
        df_average = mlbpd[mlbpd['pitch_type'] == i]['start_speed']
        ax_top[ax_number].plot([df_average.mean(), df_average.mean()],
                      [ax_top[ax_number].get_ylim()[0], ax_top[ax_number].get_ylim()[1]],
                      color=dict_color[df[df['pitch_type'] == i]['pitch_type'].values[0]],
                      linestyle=':')

        # Set the x-axis limits
        ax_top[ax_number].set_xlim(math.floor(df['start_speed'].min() / 5) * 5, math.ceil(df['start_speed'].max() / 5) * 5)
        ax_top[ax_number].set_xlabel('')
        ax_top[ax_number].set_ylabel('')

        # Hide the top, right, and left spines for all but the last subplot
        if ax_number < len(items_in_order) - 1:
            ax_top[ax_number].spines['top'].set_visible(False)
            ax_top[ax_number].spines['right'].set_visible(False)
            ax_top[ax_number].spines['left'].set_visible(False)
            ax_top[ax_number].tick_params(axis='x', colors='none')

        # Set the x-ticks and y-ticks
        ax_top[ax_number].set_xticks(range(math.floor(df['start_speed'].min() / 5) * 5, math.ceil(df['start_speed'].max() / 5) * 5, 5))
        ax_top[ax_number].set_yticks([])
        ax_top[ax_number].grid(axis='x', linestyle='--')
        for label in ax_top[ax_number].get_xticklabels():
            label.set_fontproperties(stratum)

        # Add text label for the pitch type
        ax_top[ax_number].text(-0.01, 0.5, i, transform=ax_top[ax_number].transAxes,
                      fontsize=20, va='center', ha='right', font=stratum)
        ax_number += 1

    # Hide the top, right, and left spines for the last subplot
    ax_top[-1].spines['top'].set_visible(False)
    ax_top[-1].spines['right'].set_visible(False)
    ax_top[-1].spines['left'].set_visible(False)

    # Set the x-ticks and x-label for the last subplot
    ax_top[-1].set_xticks(list(range(math.floor(df['start_speed'].min() / 5) * 5, math.ceil(df['start_speed'].max() / 5) * 5, 5)))
    ax_top[-1].set_xlabel('Velocity (mph)',fontsize=20, font=stratum)

# Defining a function that will turn our player's season stats into a dataframe
def get_player_stats(playername, year, link):
    # Using the osu stats API, with the previous functions to find the player's stats
    response = requests.get('https://osubeavers.com/api/v2/stats/bio?rosterPlayerId=' + get_player_id(playername, link) + '&sport=baseball&year=' + str(year)).json()

    # Converting it to a pandas dataframe with just the total pitching stats
    df = pd.DataFrame(response).loc['pitchingStatsTotal', 'currentStats']

    # Defining factors necessary for the stats table
    hits = int(df['hitsAllowed'])
    walks = int(df['walksAllowed'])
    hbp = int(df['hitBatters'])
    strikeouts = int(df['strikeouts'])
    homeruns = int(df['homeRunsAllowed'])
    outs_recorded = (float(df['inningsPitched'])*10-round(float(df['inningsPitched']),0)*7)
    innings_math = outs_recorded/3

    # Defining the league FIP constant (Based on the PAC-12 in 2024, could be automatic later on with scraping)
    cFIP = 5.31-(((13*663)+(3*(2530+173))-(2*5289))/5465)

    # Creating data table
    stats_data = { 'IP': [df['inningsPitched']],
                  'TBF': [hits+walks+hbp+outs_recorded],
                  'WHIP': [(walks+hits)/innings_math],
                  'ERA': [df['earnedRunAverage']],
                  'FIP': [((13*homeruns)+(3*(walks+hbp))-(2*strikeouts))/innings_math + cFIP]}
    stats_df = pd.DataFrame(stats_data)

    stats_df['K%'] = strikeouts/stats_df['TBF']
    stats_df['BB%'] = walks/stats_df['TBF']
    stats_df['K-BB%'] = stats_df['K%']-stats_df['BB%']
    stats_df = stats_df.astype(float)
    return stats_df

format_stats_dict = {
    'IP':{'table_header':'$\\bf{IP}$','format':'.1f',} ,
    'TBF':{'table_header':'$\\bf{PA}$','format':'.0f',} ,
    'AVG':{'table_header':'$\\bf{AVG}$','format':'.3f',} ,
    'K/9':{'table_header':'$\\bf{K\/9}$','format':'.2f',} ,
    'BB/9':{'table_header':'$\\bf{BB\/9}$','format':'.2f',} ,
    'K/BB':{'table_header':'$\\bf{K\/BB}$','format':'.2f',} ,
    'HR/9':{'table_header':'$\\bf{HR\/9}$','format':'.2f',} ,
    'K%':{'table_header':'$\\bf{K\%}$','format':'.1%',} ,
    'BB%':{'table_header':'$\\bf{BB\%}$','format':'.1%',} ,
    'K-BB%':{'table_header':'$\\bf{K-BB\%}$','format':'.1%',} ,
    'WHIP':{'table_header':'$\\bf{WHIP}$','format':'.2f',} ,
    'BABIP':{'table_header':'$\\bf{BABIP}$','format':'.3f',} ,
    'LOB%':{'table_header':'$\\bf{LOB\%}$','format':'.1%',} ,
    'xFIP':{'table_header':'$\\bf{xFIP}$','format':'.2f',} ,
    'FIP':{'table_header':'$\\bf{FIP}$','format':'.2f',} ,
    'H':{'table_header':'$\\bf{H}$','format':'.0f',} ,
    '2B':{'table_header':'$\\bf{2B}$','format':'.0f',} ,
    '3B':{'table_header':'$\\bf{3B}$','format':'.0f',} ,
    'R':{'table_header':'$\\bf{R}$','format':'.0f',} ,
    'ER':{'table_header':'$\\bf{ER}$','format':'.0f',} ,
    'HR':{'table_header':'$\\bf{HR}$','format':'.0f',} ,
    'BB':{'table_header':'$\\bf{BB}$','format':'.0f',} ,
    'IBB':{'table_header':'$\\bf{IBB}$','format':'.0f',} ,
    'HBP':{'table_header':'$\\bf{HBP}$','format':'.0f',} ,
    'SO':{'table_header':'$\\bf{SO}$','format':'.0f',} ,
    'OBP':{'table_header':'$\\bf{OBP}$','format':'.0f',} ,
    'SLG':{'table_header':'$\\bf{SLG}$','format':'.0f',} ,
    'ERA':{'table_header':'$\\bf{ERA}$','format':'.2f',} ,
    'wOBA':{'table_header':'$\\bf{wOBA}$','format':'.3f',} ,
    'bWAR':{'table_header':'$\\bf{bWAR}$','format':'.1f',} ,
    'G':{'table_header':'$\\bf{G}$','format':'.0f',} }

# A function for the table
def player_stats_table(playername, year, link, ax, fontsize:int=20):
    # calling the dataframe with our stats
    df = get_player_stats(playername, year, link)
    # assigning labels for the table, from the names of the stats
    stats = df.columns.to_list()
    # Formatting the values in the table
    df.loc[0] = [format(df[x][0], format_stats_dict[x]['format']) if df[x][0] != '---' else '---' for x in df]

    # creating the table
    table_fg = ax.table(cellText=df.values, colLabels=stats, cellLoc='center',
                    bbox=[0.00, 0.0, 1, 1])
    # setting font size
    table_fg.set_fontsize(fontsize)
    # hiding axis text
    ax.axis('off')
    # mapping the format to the table
    new_column_names = [format_stats_dict[x]['table_header'] if x in df else '---' for x in stats]
    # #new_column_names = ['Pitch Name', 'Pitch%', 'Velocity', 'Spin Rate','Exit Velocity', 'Whiff%', 'CSW%']
    for i, col_name in enumerate(new_column_names):
        table_fg.get_celld()[(0, i)].get_text().set_text(col_name)

def table_df(playername, year):
    df = player_year_data(playername, year)
    # Remaking the osu_df dataframe we made earlier, but this allows us to still work with the raw data for averaging
    df_group = gen_grouping(df)

    # Map pitch types to their descriptions
    df_group['pitch_description'] = df_group['pitch_type'].map(dict_pitch)

    # Calculate pitch usage as a percentage of total pitches
    df_group['pitch_usage'] = df_group['pitch'] / df_group['pitch'].sum()

    # Calculate whiff rate as the ratio of whiffs to swings, and have a process in place if there were no swings
    df_group['whiff_rate'] = df_group['whiff'].astype('float') / df_group['swing'].astype('float')

    # Calculate in-zone rate as the ratio of in-zone pitches to total pitches
    df_group['in_zone_rate'] = df_group['in_zone'] / df_group['pitch']

    # Calculate chase rate as the ratio of chases to out-of-zone pitches, with backup in case there were no out of zone pitches
    df_group['chase_rate'] = df_group['chase'] / df_group['out_zone'].astype('float')

    # Map pitch types to their colors
    df_group['color'] = df_group['pitch_type'].map(dict_color)

    # Sort the DataFrame by pitch usage in descending order
    df_group = df_group.sort_values(by='pitch_usage', ascending=False)
    color_list = df_group['color'].tolist()

    # Making a row for totals of each pitch, to have at the bottom of the table
    plot_table_all = pd.DataFrame(data={
                'pitch_type': 'All',
                'pitch_description': 'All',  # Description for the summary row
                'pitch': df['pitch_type'].count(),  # Total count of pitches
                'pitch_usage': 1,  # Usage percentage for all pitches (100%)
                'start_speed': np.nan,  # Placeholder for release speed
                'ivb': np.nan,  # Placeholder for vertical movement
                'hb': np.nan,  # Placeholder for horizontal movement
                'spin_rate': np.nan,  # Placeholder for spin rate
                'x0': np.nan,  # Placeholder for horizontal release position
                'x0': np.nan,  # Placeholder for vertical release position
                'extension': df['extension'].mean(),  # Placeholder for release extension
                'whiff_rate': df['is_whiff'].sum() / df['is_swing'].sum(),  # Whiff rate
                'in_zone_rate': df['in_zone'].sum() / df['pitch_type'].count(),  # In-zone rate
                'chase_rate': df['chase'].sum() / df['out_zone'].sum(),  # Chase rate
            }, index=[0])

    # Merging the group DataFrame with the total row DataFrame
    df_plot = pd.concat([df_group, plot_table_all], ignore_index=True)

    return df_plot, color_list

pitch_stats_dict = {
    'pitch': {'table_header': '$\\bf{Count}$', 'format': '.0f'},
    'start_speed': {'table_header': '$\\bf{Velocity}$', 'format': '.1f'},
    'ivb': {'table_header': '$\\bf{iVB}$', 'format': '.1f'},
    'hb': {'table_header': '$\\bf{HB}$', 'format': '.1f'},
    'spin_rate': {'table_header': '$\\bf{Spin}$', 'format': '.0f'},
    'x0': {'table_header': '$\\bf{hRel}$', 'format': '.1f'},
    'z0': {'table_header': '$\\bf{vRel}$', 'format': '.1f'},
    'extension': {'table_header': '$\\bf{Ext.}$', 'format': '.1f'},
    'xwoba': {'table_header': '$\\bf{xwOBA}$', 'format': '.3f'},
    'pitch_usage': {'table_header': '$\\bf{Pitch\%}$', 'format': '.1%'},
    'pitch_usage_r': {'table_header': '$\\bf{Pitch\% vs. R}$', 'format': '.1%'},
    'pitch_usage_l': {'table_header': '$\\bf{Pitch\% vs. L}$', 'format': '.1%'},
    'whiff_rate': {'table_header': '$\\bf{Whiff\%}$', 'format': '.1%'},
    'in_zone_rate': {'table_header': '$\\bf{Zone\%}$', 'format': '.1%'},
    'chase_rate': {'table_header': '$\\bf{Chase\%}$', 'format': '.1%'},
    'delta_run_exp_per_100': {'table_header': '$\\bf{RV\//100}$', 'format': '.1f'},
    'spin_axis': {'table_header': '$\\bf{Spin Axis}$', 'format': '.1f'},
    'euclidean': {'table_header': '$\\bf{Euclidean}$', 'format': '.3f'}
    }

table_columns = [ 'pitch_description',
            'pitch',
            'pitch_usage',
            'start_speed',
            'ivb',
            'hb',
            'spin_rate',
            'x0',
            'z0',
            'extension',
            'whiff_rate',
            'in_zone_rate',
            'chase_rate',
            ]

def plot_pitch_format(df, table):
    # Create a DataFrame for the summary row with aggregated statistics for all pitches
    df_group = df[table].fillna('—')

    # Apply the formats to the DataFrame
    # Iterate over each column in pitch_stats_dict
    for column, props in pitch_stats_dict.items():
        # Check if the column exists in df_plot
        if column in df_group.columns:
            # Apply the specified format to the column values
            df_group[column] = df_group[column].apply(lambda x: format(x, props['format']) if isinstance(x, (int, float)) else x)
    return df_group

# Defining MLB averages by pitch type
mlbpd_pt_avg = mlbpd.groupby(['pitch_type']).agg(
                        pitch = ('pitch_type','count'),  # Count of pitches
                        start_speed = ('start_speed','mean'),  # Average start speed
                        ivb = ('ivb','mean'),  # Average vertical movement
                        hb = ('hb','mean'),  # Average horizontal movement
                        spin_rate = ('spin_rate','mean'),  # Average spin rate
                        x0 = ('x0','mean'),  # Average horizontal release position
                        z0 = ('z0','mean'),  # Average vertical release position
                        extension = ('extension','mean'),  # Average release extension
                        swing = ('is_swing','sum'),  # Total swings
                        whiff = ('is_whiff','sum'),  # Total whiffs
                        in_zone = ('in_zone','sum'),  # Total in-zone pitches
                        out_zone = ('out_zone','sum'),  # Total out-of-zone pitches
                        chase = ('chase','sum'),  # Total chases
                        spin_axis = ('spin_direction','mean'),  # Average spin axis
                        ).reset_index()

# Calculate whiff rate as the ratio of whiffs to swings, and have a process in place if there were no swings
mlbpd_pt_avg['whiff_rate'] = mlbpd_pt_avg['whiff'].astype('float') / mlbpd_pt_avg['swing'].astype('float')

# Calculate in-zone rate as the ratio of in-zone pitches to total pitches
mlbpd_pt_avg['in_zone_rate'] = mlbpd_pt_avg['in_zone'] / mlbpd_pt_avg['pitch']

# Calculate chase rate as the ratio of chases to out-of-zone pitches, with backup in case there were no out of zone pitches
mlbpd_pt_avg['chase_rate'] = mlbpd_pt_avg['chase'] / mlbpd_pt_avg['out_zone'].astype('float')

# Add a row at the bottom, for MLB average extension, whiff rate, inzone rate, and chase rate across all pitches
mlbpd_pt_avg_totals = pd.DataFrame(data={
                'pitch_type': 'All',
                'extension': mlbpd['extension'].mean(),  # Placeholder for release extension
                'whiff_rate': mlbpd['is_whiff'].sum() / mlbpd['is_swing'].sum(),  # Whiff rate
                'in_zone_rate': mlbpd_pt_avg['in_zone'].sum() / mlbpd_pt_avg['pitch'].sum(),  # In-zone rate
                'chase_rate': mlbpd_pt_avg['chase'].sum() / mlbpd_pt_avg['out_zone'].sum(),  # Chase rate
            }, index=[0])

# Joining the totals row with the averages database to be used in the code
mlb_averages = pd.concat([mlbpd_pt_avg, mlbpd_pt_avg_totals], ignore_index=True)

# Define color maps
cmap_sum = mcolors.LinearSegmentedColormap.from_list("", ['#325aa1','#FFFFFF','#c91f26'])
cmap_sum_r = mcolors.LinearSegmentedColormap.from_list("", ['#c91f26','#FFFFFF','#325aa1'])

# List of statistics to color
color_stats = ['start_speed', 'extension', 'whiff_rate', 'in_zone_rate', 'chase_rate']

### get colors ###
def get_color(value, normalize, cmap_sum):
    color = cmap_sum(normalize(value))
    return mcolors.to_hex(color)

def get_cell_colors(df_group: pd.DataFrame,
                     df_statcast_group: pd.DataFrame,
                     color_stats: list,
                     cmap_sum: mcolors.LinearSegmentedColormap,
                     cmap_sum_r: mcolors.LinearSegmentedColormap):
    color_list_df = []
    for pt in df_group.pitch_type.unique():
        color_list_df_inner = []
        select_df = df_statcast_group[df_statcast_group['pitch_type'] == pt]
        df_group_select = df_group[df_group['pitch_type'] == pt]

        for tb in table_columns:

            if tb in color_stats and type(df_group_select[tb].values[0]) == np.float64:
                if np.isnan(df_group_select[tb].values[0]):
                    color_list_df_inner.append('#ffffff')
                elif tb == 'start_speed':
                    normalize = mcolors.Normalize(vmin=(pd.to_numeric(select_df[tb], errors='coerce')).mean() * 0.95,
                                                  vmax=(pd.to_numeric(select_df[tb], errors='coerce')).mean() * 1.05)
                    color_list_df_inner.append(get_color((pd.to_numeric(df_group_select[tb], errors='coerce')).mean(), normalize, cmap_sum))
                else:
                    normalize = mcolors.Normalize(vmin=(pd.to_numeric(select_df[tb], errors='coerce')).mean() * 0.7,
                                                  vmax=(pd.to_numeric(select_df[tb], errors='coerce')).mean() * 1.3)
                    color_list_df_inner.append(get_color((pd.to_numeric(df_group_select[tb], errors='coerce')).mean(), normalize, cmap_sum))
            else:
                color_list_df_inner.append('#ffffff')
        color_list_df.append(color_list_df_inner)
    return color_list_df

def pitch_table(playername, year, ax, fontsize:int=20):
    # Defining our dataframe by selected pitcher
    df = player_year_data(playername, year)

    # Defining what table we want for our pitch formatting function
    table = table_columns

    # Performing operations on our dataframe
    df_group, color_list = table_df(playername, year)
    df_plot = plot_pitch_format(df_group, table)
    color_list_df = get_cell_colors(df_group, mlb_averages, color_stats, cmap_sum, cmap_sum_r)

    # Create a table plot with the DataFrame values and specified column labels
    table_plot = ax.table(cellText=df_plot.values, colLabels=table_columns, cellLoc='center',
                        bbox=[0, -0.1, 1, 1],
                        colWidths=[2.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        cellColours=color_list_df)

    # Disable automatic font size adjustment and set the font size
    table_plot.auto_set_font_size(False)

    table_plot.set_fontsize(fontsize)

    # Scale the table plot
    table_plot.scale(1, 0.5)

    # Correctly format the new column names using LaTeX formatting
    new_column_names = ['$\\bf{Pitch\\ Name}$'] + [pitch_stats_dict[x]['table_header'] if x in pitch_stats_dict else '---' for x in table_columns[1:]]

    # Update the table headers with the new column names
    for i, col_name in enumerate(new_column_names):
        table_plot.get_celld()[(0, i)].get_text().set_text(col_name)

    # Bold the first column in the table
    for i in range(len(df_plot)):
        table_plot.get_celld()[(i+1, 0)].get_text().set_fontweight('bold')

    # Set the color for the first column, all rows except header and last
    for i in range(1, len(df_plot)):
        # Check if the pitch type is in the specified list
        if table_plot.get_celld()[(i, 0)].get_text().get_text() in ['Split-Finger', 'Slider', 'Changeup']:
            table_plot.get_celld()[(i, 0)].set_text_props(color='#000000', fontweight='bold')
        else:
            table_plot.get_celld()[(i, 0)].set_text_props(color='#FFFFFF')
        # Set the background color of the cell
        table_plot.get_celld()[(i, 0)].set_facecolor(color_list[i-1])

    # Remove the axis
    ax.axis('off')

def pitching_dashboard(playername, year):
    # Create a 20 by 20 figure
    df = player_year_data(playername, year)
    fig = plt.figure(figsize=(20, 20))

    # Create a gridspec layout with 8 columns and 6 rows
    # Include border plots for the header, footer, left, and right
    gs = gridspec.GridSpec(6, 8,
                        height_ratios=[2,20,9,36,36,7],
                        width_ratios=[1,18,18,18,18,18,18,1])

    # Define the positions of each subplot in the grid
    ax_headshot = fig.add_subplot(gs[1,1:3])
    ax_bio = fig.add_subplot(gs[1,3:5])
    ax_logo = fig.add_subplot(gs[1,5:7])

    ax_season_table = fig.add_subplot(gs[2,1:7])

    ax_plot_1 = fig.add_subplot(gs[3,1:3])
    ax_plot_2 = fig.add_subplot(gs[3,3:5])
    ax_plot_3 = fig.add_subplot(gs[3,5:7])

    ax_table = fig.add_subplot(gs[4,1:7])

    ax_footer = fig.add_subplot(gs[-1,1:7])
    ax_header = fig.add_subplot(gs[0,1:7])
    ax_left = fig.add_subplot(gs[:,0])
    ax_right = fig.add_subplot(gs[:,-1])

    # Hide axes for footer, header, left, and right
    ax_footer.axis('off')
    ax_header.axis('off')
    ax_left.axis('off')
    ax_right.axis('off')

    # Define the player's link that can be called for the functions
    link = get_player_link(playername=playername, year=year)
    
    # Call the functions
    fontsize = 16
    player_stats_table(playername=playername, year=year, link=link, ax=ax_season_table, fontsize=20)
    pitch_table(playername=playername, year=year, ax=ax_table, fontsize=fontsize)

    get_headshot(link=link, ax=ax_headshot)
    player_bio(playername=playername, year=year, link=link, ax=ax_bio)
    logo(ax=ax_logo)

    velocity_chart(playername=playername, year=year, fig=fig, ax=ax_plot_1, gs=gs, gs_x=[3,4], gs_y=[1,3])
    plinko_chart(playername=playername, year=year, fig=fig, ax=ax_plot_2, gs=gs, gs_x=[3,4], gs_y=[3,5])
    break_plot(playername=playername, year=year, ax=ax_plot_3)

    # Add footer text
    ax_footer.text(0, 1, 'By: Olav Moeller\nInspired by: @TJStats', ha='left', va='top', fontsize=24, font=stratum)
    ax_footer.text(0.5, 1, 'Color Coding Compares to League Average By Pitch', ha='center', va='top', fontsize=16, font=stratum)
    ax_footer.text(1, 1, 'Data: MLB, Fangraphs, OSU Baseball\nImages: OSU Baseball\nStatcast Data from 2/21-2/25/2024', ha='right', va='top', fontsize=24, font=stratum)

    # Adjust the spacing between subplots
    plt.tight_layout()

