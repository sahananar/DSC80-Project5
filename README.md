DSC 80 Project #5 - Model Building for the Recipe and Ratings Dataset

By: Sahana Narayanan (sanarayanan@ucsd.edu)

# Framing the Problem

**Introduction**: This dataset contains information about recipes and the reviews and ratings submitted for them. The recipes dataset contains basic information about the name of the recipe, the cooking time, nutrition information, steps and a description. The ratings dataset contains information about the specific reviews given.

**Prediction Problem:** Can we predict the average rating of a recipe based on information about its nutritional values and cooking time?

The type of this prediction problem will be regression, and the response variable is the average ratings of recipes. I chose it as the response variable because I wanted to investigate how the ratings users give to recipes change based on the nutritional values (calories, sugar, etc) and the amount of time taken to prepare the recipe. In Project 3, I specifically explored the relationship between cooking time and the average ratings, and I wanted to expand more on this in Project 5. 

The metric I will be using to evaluate my model is accuracy, which will indicate the proportion of predictions that are correct. I find that this is a suitable metric to use in regression problems and it will allow a clear comparison between the baseline and final models. 

**Data Cleaning**

It was necessary to perform certain data cleaning steps before creating the model, which were very similar to the cleaning steps I performed in Project 3 for this dataset - the following steps and explanations were taken from my website for Project 3. First, the recipes and ratings datasets were merged together. Next, the ratings with values of 0 were replaced with `np.nan`. This was a reasonable step as we can predict that instead of giving a recipe a 0 rating, it is more likely that the user did not submit their rating for that recipe and it had just been filled with 0 instead. Then, the average ratings per recipe were calculated. The 'nutrition' column, which previously contained a list of numbers representing each nutritional value, was split up into different parts such that there would be one column for each value. 

Next, while examining the ‘minutes’ column, I realized that several values were very large, the maximum such number being 1051200 minutes. These numbers are an unreasonable value for cooking time and it would not be logical to include values above a certain threshold in the analysis. Therefore, I decided to set the range to 600 minutes, or 10 hours. Any value larger than this would be replaced with `np.nan` as it is not reasonable a recipe would take significantly longer than this to prepare.

Finally, the remaining columns were dropped since they were not relevant to my specific analysis, and all the rows containing missing values for the average ratings and minutes columns were dropped as well, as it would be unhelpful to have missing values when making the predictions. 

Here is the `head` of the cleaned dataframe:

|   minutes |   avg_rating |   calories |   total fat |   sugar |   sodium |   protein |   saturated fat |   carbohydrates |
|----------:|-------------:|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|
|        40 |            4 |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 |
|        45 |            5 |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 |
|        40 |            5 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |
|        40 |            5 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |
|        40 |            5 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |

# Baseline Model

