### [Return to Portfolio Directory](https://remypstewart.github.io/)

I am an affiliate of the [National Rent Project](https://national-rent.github.io/) via an ongoing research collaboration with the project team. Through this partnership I have access a web scraped dataset that includes 250,000 nationally distributed Craigslist rental listings collected during 2019. I am personally interested in exploring how characteristics of neighborhood may be used to develop accurate predictions towards a given listing’s asked rental price. To operationalize this predictive model, I decided to implement the popular Extreme Gradient Boosted- “XGBoost”- regression algorithm as constructed for my rental listings data. 

XGBoost is a well-suited model for my research interests due to its inherent nonlinearity and predictive accuracy within a reasonable computational processing time.  XGBoost employs decision trees to generate predictions by comparing a given data record’s features to learned feature relationships via produced “splits” in prediction values. It is an ensemble model in that it draws from the aggregate results of multiple decision trees which are boosted by having subsequent trees correct earlier prediction errors. A final defining characteristic of XGBoost is its use of the gradient descent optimization algorithm to guide its boosted ensemble to reduce a specified loss function. For a more comprehensive overview on XGBoost please refer to the original 2016 publication by authors [Chen and Guestrin](https://dl.acm.org/doi/10.1145/2939672.2939785).

To begin my model development, I load in my preprocessed Craigslist dataset that links listing’s posted geolocations with census tract-level characteristics sourced from the [American Community Survey]( https://www.census.gov/programs-surveys/acs). Census tracts serve as a rough proxy for regional neighborhoods. My variables include measures for the majority racial and ethnic group within a given census tract, college education levels, employment rates, the region of the country the tract is in, and beyond. 

``` python

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
from yellowbrick.regressor import residuals_plot
import pandas as pd
import xgboost
from numpy import absolute, sort, mean, ones
import seaborn as sns

listings = pd.read_csv("listings_portfolio_trim.csv")
listings = listings.drop(listings.columns[0], axis=1)
listings
```
![alt text](/images/listings.png]

I first consider descriptive characteristics of rental prices as my key outcome variable. The mean of $1,581 with a standard deviation of  $789 compared to the median of $1,395 indicates a right distributional skew in rent prices. Plotting a frequency histogram confirms this- the majority of listings cluster around the low $1,000s range while a select number of higher-priced listings inflates the average asked rent price. 

``` python
listings['clean_rent'].describe()
sns.histplot(data=listings, x='clean_rent')
```
![alt text](/images/histogram.png]

I convert all categorical variables into a numeric representations through one-hot encoding which produces a final feature set of 24 variables. I then prepare respective training and test sets for the independent variables and listing rental prices. My XGBoost model will use mean absolute error (MAE) as its guiding performance metric. I additionally compute a baseline model that attempts to predict the test set’s rental prices via the training set’s average rent price to serves as the ground metric for all of my subsequent models to improve on. 

``` python

mean_train = mean(rent_train)
baseline_predictions = ones(rent_test.shape) * mean_train
mae_baseline = mean_absolute_error(rent_test, baseline_predictions)
```

The mean absolute error results implies that on average the naïve model’s predicted rent value deviates by $604 from the true asked rent price. My goal is therefore to find the optimum model design to reduce MAE from this baseline by incorporating features as well as hyperparameter tuning- referring to different choices in model structure and behavior- within my XGBoost model. I instantiate my first instance of the model with an arbitrary hyperparameter specification and use 10-fold cross validation to fit my training data. 

``` python

xgb_reg = xgboost.XGBRegressor(max_depth=5, n_estimators=100,
                               random_state=200, eta=0.1)

crossval = RepeatedKFold(n_splits=10, n_repeats=1, random_state=300)

scores = cross_val_score(xgb_reg, feature_train, rent_train, scoring='neg_mean_absolute_error', cv=crossval, n_jobs=-1)
scores = absolute(scores)
print('MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
```

This initial model produces a MAE of 360 with a standard deviation of 2.8. This represents an encouraging decrease in prediction error over our baseline of 605. Before moving to further tuning the model’s parameters, I explore potential sources of model error through visually assessing the model’s residuals, referring to the difference between the predicted rents and known rent prices for each listing. 

``` python
residuals_plot(xgb_reg, feature_train, rent_train, feature_test, rent_test)
```
![alt text](/images/residuals.png]

The slight peaks at the bottom left and top right of the residual plots both allude to the impact of outlier listings via unexpectedly lower-priced and higher-priced listings respectively. This is a likely sizable source of model prediction error. The vertical histogram indicates that the error terms are approximately normally distributed despite these outliers. This supports our use of a regression model by not violating a key model assumption concerning error term heteroskedasticity. Heteroskedaskity indicates the presence of nonrandom variance for a variable’s impact on our outcome of interest regarding predicted rent prices and therefore violates assumptions of linear relationships between different feature values and expected outcomes. I therefore feel confident in the model’s overall predictive ability despite the presence of outliers. 

A strength of XGBoost models are their tunability regarding a range of hyperparameters to improve prediction performance. For my initial model development, I focus on tuning four of these hyperparameters:
•	Max depth dictates tree size, as in the number of decisions based on the provided feature space a given tree can make to generate its rental price predictions. 
•	Number of estimators delineates the model’s ensemble size by varying the number of decision trees used.
•	Gamma defines the minimum amount required in a decrease of the loss function to justify a tree node split on a given feature, with higher values leading to more conservative models.  
•	Eta refers to the learning rate which weights the impact of information added by each boosted tree to prevent model overfitting. 

By using Scikit-learn’s grid seach cross-validation function, I test all specified values of each of the hyperparameters across all potential model combinations. This allows me to identify which model specification leads to the greatest reduction in MAE.  

``` python

params_test={'max_depth': [4,6,8],
        'gamma': [0, 1, 2],
        'eta': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100],
    }

xgb_tune = xgboost.XGBRegressor(random_state = 200)
grid_results = GridSearchCV(estimator=xgb_tune,
                           param_grid=params_test,
                           scoring='neg_mean_absolute_error')
grid_results.fit(feature_train, rent_train)

print("Best parameters:", grid_results.best_params_)
print("Lowest MAE: ", (-grid_results.best_score_))
```
![alt text](/images/params.png]

The grid search process successfully identifies the optimum hyperparameter specification to follow for initializing a final XGBoost model which are the highest tested values for each paramater. This obtains a MAE of 317. I then instantiate my final model following the hyperparameter selections and generate rent price predictions on the held-out test data.

``` python
xgb_final = xgboost.XGBRegressor(max_depth=8, n_estimators=100, eta=0.2, gamma=2,
                                 random_state=400)
xgb_final.fit(feature_train, rent_train)
preds = xgb_final.predict(feature_test)
final_mae = mean_absolute_error(rent_test, preds)
print("RMSE: %f" % (final_mae))
```

Our final model’s MAE is 315 on the test set compared to our baseline MAE of 604 and our initial untuned model of 360. Let’s examine which neighborhood characteristic features are the most influential for defining splits within the decision trees. 

``` python
xgboost.plot_importance(xgb_final)
```
![alt text](/images/features.png]

The tract’s poverty level, share of college graduates, and share of single-family homes (compared to multi-family units such as apartments) stand out as the most predictive factors regarding rent price. These all make intuitive sense as proxies for the socioeconomic status. 

As a next step from this model development, I would like to scale up my data pipeline & generated model to include the approximately 1.7 million Craigslist records I have access to. I intend to do this through distributed computing via Google’s Cloud AI Platform. Stay tuned for another post delineating my cloud deployment steps soon!

References:
Chen, Tianqi and Carlos Guestrin. 2016. “XGBoost: A Scalable Tree Boosting System” Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining: 785-794. 
