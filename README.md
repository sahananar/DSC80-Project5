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

I used all the possible features to create my model - all of the nutritional values (calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates) and the cooking time in minutes. All 8 of these these features were quantitative, therefore no feature engineering or encodings were necessary. The regressor I used in my pipeline was the `DecisionTreeRegressor` in `sklearn` with an initial `max_depth` of 4 - this is a hyperparameter I would be tuning in the following step. 

After fitting the pipeline with the relevant `X` and `y` data and producing an array of predicted `avg_rating` values, I calculated a RMSE value of 0.49. This value is halfway in between, which does not significantly indicate poor performance or good performance. Therefore, I would conclude that while the model does make some reasonable predictions, it can also be improved on. 

I also wanted to evaulate the model's ability to generalize to unseen data - to do this, I used the `train_test_split` function and fit the pipeline with the `x_train` and `y_train` data. Then, I used the pipeline to predict values for the `y_train` and `y_test` data and found the RMSE of each. The RMSE of the training data was 0.494 and for the test data it was 0.4975. Therefore, since the RMSE values of both are very similar, I concluded that the model is not overfitting to the data and it would be able to generalize well to unseen data. 

# Final Model

First, I focused on tuning the `max_depth` hyperparameter for the regressor. I used an iterative method, where I initially did a `train_test_split`, then looped through values ranging from 1 to 21 and set the `max_depth` parameter of the `DecisionTreeRegressor` equal to each of these values. I stored the training and testing error for each iteration in a dictionary and at the end looked at which value of the depth resulted in the lowest test error. This value was 20, which I would be using in the pipeline for the final model. 

To engineer the features, I looked at the range for each column - the difference between the maximum and minimum for each column. This range was reasonable for some values but quite high for some others, especially for 'calories'. I decided that it would be better to convert these columns into standard units, so the variables could be better compared among one another. I would engineer the features by applying the `StandardScaler()` function to each of the columns. 

My new pipeline was created using `StandardScaler()` and with the best hyperparameter for `DecisionTreeRegressor`. I fit this pipeline on the same X and y data, predicted new values, and calculated a final RMSE of 0.36. This was a significant improvement from the baseline model and indicates that the improvements did help the model perform better and make even more accurate predictions. It would now be safer to say that this model has "good" performance, compared to the baseline model. 

However, when conducting another `train_test_split` and evaluating the model's ability to generalize to unseen data, I obtained a training error of 0.36 and a test error of 0.47. Since the test error is larger, this could indicate the presence of overfitting and the model may not be able to generalize to unseen data as well. Therefore, while the model improved in quality and made better predictions than the baseline model, it was not able to generalize as well due to the potential presence of overfitting. 

# Fairness Analysis

The two groups I chose were recipes with cooking times of under 60 minutes and recipes with cooking times of over 60 minutes. My question was, does my model perform worse for recipes with cooking times of 60 minutes than it does for recipes with cooking times of over 100 minutes? The evaluation metric I chose was RMSE, to keep it consistent with the previous evaluations. 

Here are my null and alternative hypotheses: 

Null Hypothesis: My model is fair. Its RMSE for recipes with cooking times of under 60 minutes and over 60 minutes are roughly the same, and any differences are due to random chance.

Alternative Hypothesis: My model is unfair. Its RMSE for recipes with cooking times of under 60 minutes is unequal to its RMSE of cooking times of over 60 minutes.

I first used the `Binarizer()` function with a threshold of 60 minutes to transform the 'minutes' column of the dataframe.  I first used the already fit pipeline from the final model to predict the rating values for just the rows in the dataframe where the binarized value was 0, and then again to predict using the rows where the binarized value was 1. My test statistic was the absolute difference for the RMSE between these two groups. 

I ran my permutation test for 500 iterations and in each iteration I shuffled the binarized minutes column, used the pipeline to predict the ratings for each group separately, and calculated the absolute difference in RSME. I used a significance level of 0.05, the resulting p-value was 0.026. 

Therefore, since the p-value was under the significance value, I rejected the null hypothesis that the model is fair and its RMSE for cooking times in both groups is equal. 
