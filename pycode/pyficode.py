import pandas as pd
file_path1 = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240731.tsv'
df1 = pd.read_csv(file_path1, sep='\t')
df1.head()


file_path2 = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240801.tsv'
df2 = pd.read_csv(file_path2, sep='\t')
df2.head()


file_path3 = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240802.tsv'
df3 = pd.read_csv(file_path3, sep='\t')
df3.head()

file_path4 = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240803.tsv'
df4 = pd.read_csv(file_path4, sep='\t')
df4.head()


file_path5 = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240804.tsv'
df5 = pd.read_csv(file_path5, sep='\t')
df5.head()


file_path6 = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240805.tsv'
df6 = pd.read_csv(file_path6, sep='\t')
df6.head()


file_path7 = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240806.tsv'
df7 = pd.read_csv(file_path7, sep='\t')
df7.head()

combined_df = pd.concat([df1,df2,df3,df4,df5, df6, df7])
combined_df.to_csv('combined_file.csv', index=False)


# importing all required lib and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics
from sklearn.feature_selection import RFE #(recursive feature elimination) help use to select important feature for model building

from sqlalchemy import create_engine  #lib for to push the data to database like mysql
import joblib    # lib for to save data pipeline and model pipeline
import pickle

# loading the combined csv file
df = pd.read_csv('C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/combined_file.csv')
df.head()
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
df.to_sql('flight_ad_bid', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from flight_ad_bid;'
df = pd.read_sql_query(sql, engine)
df.sample()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()


# selecting important features and removing unimporatant features  
df = df[['Est. Clicks', 'Est. Spend (USD)', 'Third Rank Bid (USD)', 'Est. Impressions']]
corrmat = df.corr()

## extracting independent and dependent var
x = df.drop(columns = ['Est. Spend (USD)'])
y = df['Est. Spend (USD)']

# extracting numerical and categorical features
nf = x.select_dtypes(exclude = 'object').columns
cf = x.select_dtypes(include = 'object').columns

############################# univariate analysis
# 1. univariate analysis
df.info()
df.describe()

df.columns
df['Est. Spend (USD)'].unique()
df['Est. Spend (USD)'].value_counts()

# Histogram # Visualize the distribution and frequency of the target variable
plt.hist(df['Est. Spend (USD)'], bins=5, color='skyblue', edgecolor='red')
plt.title('Histogram of Estimate Spend in (USD)')
plt.xlabel('Est. Spend (USD)')
plt.ylabel('Frequency')
plt.show()


################## bivariate analysis
corrmat = df[nf].corr()
corrmat

# Correlation Matrix with Heatmap
import seaborn as sns
plt.figure(figsize=(20,20))
g=sns.heatmap(corrmat,annot=True,cmap="RdYlGn")

# Correlation with the target variable
corr_with_target = df[nf].corrwith(df['Est. Spend (USD)'])
plt.figure(figsize=(10, 6))
corr_with_target.plot(kind='bar')
plt.title('Correlation with Target Variable')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.show()


################### multivariate analysis
# Pairplot for visualizing relationships between features
sns.pairplot(df)
plt.title('Pairplot of All Features')
plt.show()



# playing with AutoEDA Lib to check data quality
# 1) SweetViz
import sweetviz as sv
s = sv.analyze(df)
s.show_html()

# 3) D-Tale
import dtale 
d = dtale.show(df)
d.open_browser()


x.isnull().sum()  # there is no null or missing value in the dataset
########################## creating the pipline for simpleImputer
# Define pipeline for missing data if any
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, nf)])
imputation = preprocessor.fit(x)
joblib.dump(preprocessor, 'meanimpute')

imputed_df = pd.DataFrame(preprocessor.transform(x), columns = nf)
imputed_df
imputed_df.isnull().sum()



###################### playing with outliers
# Defining a function to count outliers present in dataset
def count_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    return outliers
# Counting outliers before applying Winsorization tech
outliers_before = imputed_df.apply(count_outliers)
outliers_before      
outliers_before.sum()  # here 7532 total num of outlier/extreame values are present in dataset

# plotting boxplot for to check outliers
imputed_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()

############################## Define Winsorization pipeline
winsorizer_pipeline = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
X_winsorized = winsorizer_pipeline.fit_transform(imputed_df)
joblib.dump(winsorizer_pipeline, 'winsor')  

# Transform Winsorized data back to DataFrame
X_winsorized_df = pd.DataFrame(X_winsorized, columns=nf)

# Count outliers after Winsorization
outliers_after = X_winsorized_df.apply(count_outliers)
outliers_after

X_winsorized_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75)  
plt.show()

############################ creating pipline for standard scaler
scale_pipeline = Pipeline([('scale', StandardScaler())])
X_scaled = scale_pipeline.fit(X_winsorized_df)
joblib.dump(scale_pipeline, 'standscal')

X_scaled_df = pd.DataFrame(scale_pipeline.transform(X_winsorized_df), columns = X_winsorized_df.columns)
X_scaled_df


############################ Defining the encoding pipeline
encoding_pipeline = Pipeline([('onehot', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))])
# Applying the encoding pipeline to categorical features
prep_encoding_pipeline = ColumnTransformer([('cf', encoding_pipeline, cf)])
# Fit and transform the data together
X_encoded = prep_encoding_pipeline.fit(x)
# Save the fitted pipeline using joblib
joblib.dump(prep_encoding_pipeline, 'encoding_pipeline')

# Create a DataFrame from the transformed data
encode_data = pd.DataFrame(X_encoded.transform(x), columns=prep_encoding_pipeline.get_feature_names_out())
encode_data.info()

clean_data = pd.concat([X_scaled_df, encode_data], axis = 1)  
clean_data.info()




from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# import statsmodels.formula.api as smf
import statsmodels.api as sm
P = add_constant(clean_data)
basemodel = sm.OLS(y, P).fit()
basemodel.summary()       # p-values of coefficients found to be insignificant due to colinearity p values should below 0.05

vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif  # Identify the variale with highest colinearity using Variance Inflation factor (VIF)
# Variance Inflation Factor (VIF) Assumption: VIF > 10 = colinearity
# VIF on clean Data
# Tune the model by verifying for influential observations influence plot
sm.graphics.influence_plot(basemodel)

clean_data1_new = clean_data.drop(clean_data.index[[6070, 1852,144, 3029, 4226]])
y_new = y.drop(y.index[[6070, 1852, 144, 3029, 4226]])

# Build model on dataset
P = add_constant(clean_data1_new)
basemode1 = sm.OLS(y_new, P).fit()
basemode1.summary()


# Splitting data into training and testing data set
from sklearn.metrics import r2_score
X_train, X_test, Y_train, Y_test = train_test_split(P, y_new, test_size = 0.2, random_state = 0) 
model = sm.OLS(Y_train, X_train).fit()
model.summary()

 # ---- TRAINING DATA EVALUATION ---- #
# Predicting on the training data
ytrain_pred = model.predict(X_train)
r_squared_train = r2_score(Y_train, ytrain_pred)  # # Calculating R-squared for the training data
r_squared_train  #0.9012720450629462
# Calculating residuals for the training data
train_resid = Y_train - ytrain_pred
train_rmse = np.sqrt(np.mean(train_resid**2))    # # Calculating RMSE for the training data
train_rmse 


# ---- TEST DATA EVALUATION ---- #
# Predicting on the test data
ytest_pred = model.predict(X_test)
r_squared_test = r2_score(Y_test, ytest_pred)
r_squared_test  #0.9077712242333801

print('MAE:', metrics.mean_absolute_error(Y_test, ytest_pred))
print('MSE:', metrics.mean_squared_error(Y_test, ytest_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, ytest_pred)))

sns.distplot(Y_test-ytest_pred)
plt.scatter(Y_test,ytest_pred)

######################### hyperparameter tuning with with Cross Validation (cv)
from sklearn.linear_model import LinearRegression
# k-fold CV (using all variables)
lm = LinearRegression()
## Scores with KFold
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(lm, X_train, Y_train, scoring = 'r2', cv = folds)
scores.mean()



#################################### hyperparmeter with GridSeachCV & Model building with CV and RFE
# specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 9))}]

# perform grid search # 1. specify model  # 2. lm = LinearRegression()   # 3 call GridSearchCV()
lm.fit(X_train, Y_train)
# Recursive feature elimination
rfe = RFE(lm)
model_cv = GridSearchCV(estimator = rfe, param_grid = hyper_params, scoring = 'r2', cv = folds, verbose = 1,return_train_score = True)      
model_cv.fit(X_train, Y_train)     

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

model_cv.best_params_
cv_lm_grid = model_cv.best_estimator_
cv_lm_grid
## Saving the model into pickle file
pickle.dump(cv_lm_grid, open('finalmodel.pkl', 'wb'))


# ---- TRAINING DATA EVALUATION ---- #
# Predicting on the training data
ytrain_pred = cv_lm_grid.predict(X_train)

# Calculating R-squared for the training data
r_squared_train = r2_score(Y_train, ytrain_pred)
print("R-squared (Train):", r_squared_train)

# Calculating residuals for the training data
train_resid = Y_train - ytrain_pred

# Calculating RMSE for the training data
train_rmse = np.sqrt(np.mean(train_resid**2))
print("RMSE (Train):", train_rmse)


# ---- TEST DATA EVALUATION ---- #
# Predicting on the test data
ytest_pred = cv_lm_grid.predict(X_test)

# Calculating R-squared for the test data
r_squared_test = r2_score(Y_test, ytest_pred)
print("R-squared (Test):", r_squared_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, ytest_pred))
print('MSE:', metrics.mean_squared_error(Y_test, ytest_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, ytest_pred)))


# plotting cv results
plt.figure(figsize = (16, 6))
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc = 'upper left')

# train and test scores get stable after 3rd feature. 
# we can select number of optimal features more than 3



################################## Model testing on New sample dataset
file_path = 'C:/Users/SHINU RATHOD/Desktop/internship assignment/04_Quadrilite Technologies/dataset/Last Days Report/cmp2_panel_7003669_flight_20240803.tsv'
data = pd.read_csv(file_path, sep='\t')
data1 = df[['Est. Clicks', 'Third Rank Bid (USD)', 'Est. Impressions']]
data1 = data1.head(10)
data1.to_csv('fl_ad_test.csv', index=False)
data1.head()

# extracting numerical and categorical features
nf = data1.select_dtypes(exclude = 'object').columns
cf = data1.select_dtypes(include = 'object').columns


model1 = pickle.load(open('finalmodel.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
StandScal = joblib.load('standscal')
encoding = joblib.load('encoding_pipeline')

clean = pd.DataFrame(impute.transform(data1), columns = nf)
clean1 = pd.DataFrame(winsor.transform(clean),columns = data1.columns)
clean2 = pd.DataFrame(StandScal.transform(clean1),columns = data1.columns)
clean3 = pd.DataFrame(encoding.transform(data1), columns = encoding.get_feature_names_out(input_features = data1.columns))

clean_data = pd.concat([clean2, clean3], axis = 1)
clean_data.info()

# required_features = model1.get_feature_names_out()
# Reorder the columns in `clean_data` to match the order in `required_features`
# clean_data = clean_data[required_features]
# Add the missing 'const' column and its in first place of dataset order is important while fitting the model 'const' column was at first position 
clean_data.insert(0, 'const', 1)
prediction = pd.DataFrame(model1.predict(clean_data), columns=['pred_Est_Spend_(USD)'])
prediction
