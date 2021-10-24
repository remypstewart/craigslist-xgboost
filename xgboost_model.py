from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
from yellowbrick.regressor import residuals_plot, prediction_error
import pandas as pd
import xgboost
from numpy import absolute, sort, mean, ones
import matplotlib.pyplot as pyplot
import seaborn as sns

## initial view

listings = pd.read_csv("listings_portfolio_trim.csv")
listings = listings.drop(listings.columns[0], axis=1)
listings

listings['clean_rent'].describe()
sns.histplot(data=listings, x='clean_rent')

## train/test prep

cat_columns = ['prin_city', 'region']
cat_df = listings[cat_columns]
onehot = pd.get_dummies(cat_df, columns = cat_columns)
num_df = listings.drop(cat_columns, axis = 1)
del num_df['clean_rent']
features = pd.concat([onehot, num_df], axis = 1)
rent = listings['clean_rent']

feature_train, feature_test, rent_train, rent_test = train_test_split(features, rent, test_size=0.2, random_state=100)

## baseline mae

mean_train = mean(rent_train)
baseline_predictions = ones(rent_test.shape) * mean_train
mae_baseline = mean_absolute_error(rent_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

## first model 
xgb_reg = xgboost.XGBRegressor(max_depth=5, n_estimators=100,
                               random_state=200, eta=0.1)

crossval = RepeatedKFold(n_splits=10, n_repeats=1, random_state=300)

scores = cross_val_score(xgb_reg, feature_train, rent_train, scoring='neg_mean_absolute_error', cv=crossval, n_jobs=-1)
scores = absolute(scores)
print('MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

residuals_plot(xgb_reg, feature_train, rent_train, feature_test, rent_test)
