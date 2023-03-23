DSC 80 Project #5 - Model Building for the Recipe and Ratings Dataset

By: Sahana Narayanan (sanarayanan@ucsd.edu)

# Framing the Problem

**Introduction**: This dataset contains information about recipes and the reviews and ratings submitted for them. The recipes dataset contains basic information about the name of the recipe, the cooking time, nutrition information, steps and a description. The ratings dataset contains information about the specific reviews given.

**Prediction Problem:** Can we predict the average rating of a recipe based on information about its nutritional values and cooking time?

The type of this prediction problem will be regression, and the response variable is the average ratings of recipes. I chose it as the response variable because I wanted to investigate how the ratings users give to recipes change based on the nutritional values (calories, sugar, etc) and the amount of time taken to prepare the recipe. In Project 3, I specifically explored the relationship between cooking time and the average ratings, and I wanted to expand more on this in Project 5. All of the values used to make predictions would be known at the time of prediction

The metric I will be using to evaluate my model is root mean squared error, using the `mean_squared_error` function from `sklearn`. I think this metric will be useful as it will indicate how much error there was in the predictions and will give a good idea of how reasonable the predictions were. Additionally, I will be able to use this metric to evaulate the best hyperparameters as well as evaluating how well the model generalizes to unseen data. 

**Data Cleaning**

It was necessary to perform certain data cleaning steps before creating the model, which were very similar to the cleaning steps I performed in Project 3 for this dataset - the following steps and explanations were taken from my website for Project 3. First, the recipes and ratings datasets were merged together. Next, the ratings with values of 0 were replaced with `np.nan`. This was a reasonable step as we can predict that instead of giving a recipe a 0 rating, it is more likely that the user did not submit their rating for that recipe and it had just been filled with 0 instead. Then, the average ratings per recipe were calculated. The 'nutrition' column, which previously contained a list of numbers representing each nutritional value, was split up into different parts such that there would be one column for each value. 

Next, while examining the ‘minutes’ column, I realized that several values were very large, the maximum such number being 1051200 minutes. These numbers are an unreasonable value for cooking time and it would not be logical to include values above a certain threshold in the analysis. Therefore, I decided to set the range to 600 minutes, or 10 hours. Any value larger than this would be replaced with `np.nan` as it is not reasonable a recipe would take significantly longer than this to prepare.

Finally, the remaining columns were dropped since they were not relevant to my specific analysis, and all the rows containing missing values for the average ratings and minutes columns were dropped as well, as it would be unhelpful to have missing values when making the predictions. 

Here is the `head` of the cleaned dataframe:

|   minutes |   calories |   total fat |   sugar |   sodium |   protein |   saturated fat |   carbohydrates |   avg_rating |
|----------:|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|-------------:|
|        40 |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 |            4 |
|        45 |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 |            5 |
|        40 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |            5 |
|        40 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |            5 |
|        40 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |            5 |

# Baseline Model

Describe your model and state the features in your model, including how many are quantitative, ordinal, and nominal, and how you performed any necessary encodings. Report the performance of your model and whether or not you believe your current model is “good” and why.

I used all the possible features to create my model - all of the nutritional values (calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates) and the cooking time in minutes. All 8 of these these features were quantitative, therefore no feature engineering or encodings were necessary. The regressor I used in my pipeline was the `DecisionTreeRegressor` in `sklearn` with an initial `max_depth` of 4 - this is a hyperparameter I would be tuning in the following step. 

After fitting the pipeline with the relevant `X` and `y` data and producing an array of predicted `avg_rating` values, I calculated a RMSE value of 0.24. Since this value is on the lower side, it indicates that the model generally made reasonable predictions and was good at predicting the rating values given the features that were passed in. Therefore, I concluded that this current model is "good". 

I also wanted to evaulate the model's ability to generalize to unseen data - to do this, I used the `train_test_split` function and fit the pipeline with the `x_train` and `y_train` data. Then, I used the pipeline to predict values for the `y_train` and `y_test` data and found the RMSE of each. The RMSE of the training data was 0.494 and for the test data it was 0.4975. Therefore, since the RMSE values of both are very similar, I concluded that the model is not overfitting to the data and it would be able to generalize well to unseen data. 

# Final Model

