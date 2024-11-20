import streamlit as st
from nba_api.stats.endpoints import playergamelog
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.master("local").appName("LeBron Points Prediction").getOrCreate()

# Function to fetch player's game data using nba_api
def fetch_player_data(player_id, season="2023-24"):
    game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    game_data = game_log.get_data_frames()[0]
    return game_data

# Function to train the model
def train_model(game_data):
    # Preprocess the data
    game_data = spark.createDataFrame(game_data)
    game_data = game_data.select("GAME_DATE", "PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "MIN")
    
    # Create a feature for the previous game's points (shifted)
    game_data = game_data.withColumn("prev_pts", F.lag(game_data["PTS"], 1).over(Window.orderBy("GAME_DATE")))
    
    # Drop the first row (since it has no previous game data)
    game_data = game_data.filter(game_data["prev_pts"].isNotNull())
    
    # Create the feature vector for prediction (using previous points, assists, and other stats)
    assembler = VectorAssembler(inputCols=["prev_pts", "AST", "REB", "STL", "BLK", "FG_PCT", "MIN"], outputCol="features")
    
    # Split the data into training and testing sets
    train_data, test_data = game_data.randomSplit([0.8, 0.2], seed=1234)
    
    # Train a linear regression model
    lr = LinearRegression(featuresCol="features", labelCol="PTS")
    
    # Create a pipeline
    pipeline = Pipeline(stages=[assembler, lr])
    
    # Fit the model
    model = pipeline.fit(train_data)
    
    return model, test_data

# Function to make predictions using the trained model
def predict_points(model, test_data):
    predictions = model.transform(test_data)
    return predictions

# Streamlit UI
def main():
    st.title("NBA Player Points Prediction Dashboard")
    
    # User input for player
    player_name = st.text_input("Enter Player Name (e.g., LeBron James):", "LeBron James")
    
    if player_name:
        # Fetch player data
        player_id = 2544  # LeBron James' ID (replace with a dynamic ID lookup if needed)
        game_data = fetch_player_data(player_id)
        
        # Train the model on the data
        model, test_data = train_model(game_data)
        
        # Make predictions
        predictions = predict_points(model, test_data)
        
        # Show the last 5 games' points and predicted points for the next game
        st.subheader(f"Past Game Points and Projected Points for {player_name}")
        prediction_df = predictions.select("GAME_DATE", "PTS", "prediction").toPandas()
        
        st.write(prediction_df.tail(5))
        
        # Plot the past and predicted points
        st.subheader(f"Points Over Time for {player_name}")
        st.line_chart(prediction_df[['GAME_DATE', 'PTS']].set_index('GAME_DATE'))
        
        st.subheader("Predicted Points for Next Game")
        next_game_pred = prediction_df.iloc[-1]["prediction"]
        st.write(f"Predicted Points for Next Game: {next_game_pred:.2f}")
    
if __name__ == "__main__":
    main()
