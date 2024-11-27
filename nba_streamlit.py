import pandas as pd
import numpy as np
import random
#from nba_api import *
from nba_api.stats.endpoints import playergamelog
from nba_api.stats import endpoints
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players
#import matplotlib.pyplot as plt
import datetime as dt
from nba_api.stats.endpoints import commonplayerinfo
import streamlit as st
import pickle

season_mapping = {
    2023: "2023-24",
    2022: "2022-23",
    2021: "2021-22",
    2020: "2020-21",
    2019: "2019-20",
    2018: "2018-19",
    2017: "2017-18",
    2016: "2016-17",
    2015: "2015-16",
    2014: "2014-15",
    2013: "2013-14",
    2012: "2012-13",
    2011: "2011-12",
}

def get_opponent_abbrev(team_name):


    teams = {
    'GSW'	: 'Golden State Warriors',
    'LAC' :	'Los Angeles Clippers',
    'LAL'	: 'Los Angeles Lakers',
    'NOP'	: 'New Orleans Pelicans',
    'NYK'	: 'New York Knicks',
    'BKN'	: 'Brooklyn Nets',
    'OKC'	: 'Oklahoma City Thunder',
    'UTH'	: 'Utah Jazz',
    'SAS'   : 'San Antonio Spurs'
    }

    if team_name in teams.values():
        return ([i for i in teams if teams[i]==team_name][0])

    else:
        return (team_name[0:3].upper())


def get_player_id(player_name):
    player_list = players.find_players_by_full_name(player_name)
    if player_list:
        return player_list[0]['id']
    else:
        return None

def get_player_team_abbrev(player_id):
    team_abbrev = endpoints.commonplayerinfo.CommonPlayerInfo(player_id)
    if team_abbrev:
        return team_abbrev.get_normalized_dict().get('CommonPlayerInfo')[0].get('TEAM_ABBREVIATION')
    else:
        return None

numGames=5

def main():
    st.title("NBA Player Points Prediction Dashboard")

    # User input for player
    player_name = st.text_input("Enter Player Name (e.g., LeBron James):", "LeBron James")

    #INPUTS
    playerName = player_name
    

    #Get the Opponent Team's game averages and advanced metrics for this season using the TeamEstimatedMetrics function
    opp = endpoints.TeamEstimatedMetrics(season=SeasonAll.current_season).get_data_frames()[0]
    #Append the abbreviation column
    opp['TEAM_ABBREV'] = [get_opponent_abbrev(team_name) for team_name in opp['TEAM_NAME']]

    #find the game logs for the player in question
    player_id = get_player_id(playerName)
    team_abbrev = get_player_team_abbrev(player_id)

    log = endpoints.PlayerGameLog(player_id=player_id, season=SeasonAll.current_season).get_data_frames()[0]
    #logPlays = endpoints.PlayerGameLog(player_id=player_id, season=SeasonAll.current_season, season_type_all_star='Playoffs').get_data_frames()[0]
    #log = pd.concat([logRegSea,])
    log['GAME_DATE'] = pd.to_datetime(log['GAME_DATE'], format='%b %d, %Y')
    log = log.sort_values('GAME_DATE', ascending=False)
    log['OPP_TEAM_ABBREV'] = [matchup[-3:] for matchup in log['MATCHUP']]

    #create merged dataframe
    df = log.merge(opp, left_on = 'OPP_TEAM_ABBREV', right_on = 'TEAM_ABBREV')
    #df['E_OT_MIN'] = df['MIN_y'] - 48*df['GP']
    df['HOMECOURT'] = [0 if '@' in str(matchup) else 1 for matchup in df['MATCHUP']]
    #opp['E_OT_MIN'] = opp['MIN'] - 48*opp['GP']


    import urllib.request, json

    file_name = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    with urllib.request.urlopen(file_name) as url:
        data = json.load(url)

    games = []
    curr_date = str(dt.datetime.today()).split()[0]
    #count = 0
    for date in data.get("leagueSchedule").get("gameDates"):
      for game in date.get('games'):
        if (str(game.get('gameDateEst'))>=curr_date) & (team_abbrev in game.get('gameCode')):
          games.append(game)


    # load gradient boost model
    with open('NBA_grad_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    #Perform training on all previous games in the season (regular season and playoffs)
    #best_model, best_params, best_score = gradient_boosting_tune(df.loc[:,['E_OREB_PCT', 'E_DREB_PCT', 'E_PACE', 'E_OT_MIN', 'HOMECOURT']], df.loc[:, 'PTS'])

    preds = pd.DataFrame()
    dates = []
    #predict points scored for each game
    for game in games:
      #get game date
      dates.append(game.get('gameDateEst')[0:10])
      #get opponent team abbreviation and if home team or not
      opp_abbrev = game.get('gameCode')[-3:]
      isAway = 0 if team_abbrev == game.get('homeTeam').get('teamTricode') else 1

      #gather the input data to predict the stats for the game in the future (for the player in question)
      curr_opp = opp.loc[opp['TEAM_ABBREV']==opp_abbrev,:].reset_index(drop=True)
      curr_opp.loc[:,'HOMECOURT'] = 1 if isAway==0 else 0
      curr_opp['AVG_PTS'] = df['PTS'].mean()
      attributes = ['E_DEF_RATING','E_OREB_PCT', 'E_DREB_PCT', 'E_PACE', 'W_RANK', 'HOMECOURT']
      new_game_inputs = curr_opp.loc[:,attributes]
      # Make predictions on the test set
      predicted_points = best_model.predict(new_game_inputs)
      curr_opp.loc[:,'PTS_Pred'] = predicted_points + curr_opp['AVG_PTS']
      preds = pd.concat([preds, curr_opp])
    
    #finalize predictions dataframe
    preds['GAME_DATE'] = dates
    predictions = preds

    # Show the last 5 games' points and predicted points for the next game
    st.subheader(f"Projected Points for Next Five Games for {player_name}")
    prediction_df = preds[["GAME_DATE", "PTS_Pred"]]

    st.write(prediction_df.tail(5))

    # Plot the past and predicted points
    st.subheader(f"Points Over Time for {player_name}")
    print_df = df.loc[0:5, ['GAME_DATE','PTS']]
    for pred in preds:
      print_df.loc[len(print_df)] = [pd.to_datetime(pred['GAME_DATE']),pred['PTS_Pred']]
    st.line_chart(print_df.set_index('GAME_DATE'))

    st.subheader("Predicted Points for Next Game")
    next_game_pred = prediction_df.iloc[-1]["PTS_Pred"]
    st.write(f"Predicted Points for Next Game: {next_game_pred:.2f}")

if __name__ == "__main__":
    main()
