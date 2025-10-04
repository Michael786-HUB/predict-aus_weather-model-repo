import pandas as pd 


# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('weatherAUS.csv')

# %%
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.shape)
print(df.info())

# %%
print(df.describe())

# %%
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()


print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# %%
print(df['Date'].head())

for col in categorical_cols:
    print(f"Unique values in {col}: {df[col].nunique()}")

# %%
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

year_counts = df['Year'].value_counts().sort_index()
total_rows = len(df)
for year, count in year_counts.items():
    percentage = (count / total_rows) * 100
    print(f"Year: {year}, Rows: {count}, Percentage: {percentage:.2f}%")

# %%
print(df.head())

# %%
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# %%
cols= ["#C2C4E2","#EED4E5"]
sns.countplot(x=df["RainTomorrow"], palette=cols)

# %%
plt.figure(figsize=(15, 6))

for col in df[numerical_cols]:
    plt.figure(figsize=(10, 4))
    sns.boxplot(df[col].dropna())
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# %%
plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 2)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


# %%
for col in categorical_cols:
    percent = df[col].isnull().mean() * 100
    print(f"{col}: {percent:.2f}% missing values")


# %%

df = df.dropna(subset=categorical_cols)
for col in categorical_cols:
    print(f"Value counts for {col}:\n{df[col].value_counts()}\n")

# %%
for col in numerical_cols:
    percent = df[col].isnull().mean() * 100
    if percent > 10:
        print(f"{col}: {percent:.2f}% missing values")
        print(df[col].describe())

# %% [markdown]
# Evaportation, sunshine, Cloud9am, Cloud3pm contain too a high percentage of missing values and need to be dealt with

# %%
# Removing outliers from the Evaporation column using IQR method

IQR = df['Evaporation'].quantile(0.75) - df['Evaporation'].quantile(0.25)
lower_bound = df['Evaporation'].quantile(0.25) - 1.5 * IQR
upper_bound = df['Evaporation'].quantile(0.75) + 1.5 * IQR

print(upper_bound, lower_bound)

outliers = df[(df['Evaporation'] > upper_bound) | (df['Evaporation'] < lower_bound)]
print(f"Number of outliers in Evaporation: {len(outliers)} out of {len(df['Evaporation'])}")


df = df[(df['Evaporation'] <= upper_bound) & (df['Evaporation'] >= lower_bound)]
print(f"Data shape after removing outliers: {df.shape}")

# %%
for col in numerical_cols:
    percent = df[col].isnull().mean() * 100
    if percent > 10:
        print(f"{col}: {percent:.2f}% missing values")
        print(df[col].describe())
 

# %%
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"Filled missing values in {col} with mean: {df[col].mean()}")
    

# %%
print(df.info())
print(df[categorical_cols].columns)
print(df[numerical_cols].columns)
 

# %%
df = pd.get_dummies(df, columns=['Location'], drop_first=True)
df = df['Location'].apply(lambda x: 1 if x == 'True' else 0)
print(df.head())

# %%
location_cols = [col for col in df.columns if col.startswith('Location_')]
for col in location_cols:
    df[col] = df[col].astype(int)
print(df[location_cols].head())

# %%
df = pd.get_dummies(df, columns=['RainToday'], drop_first=True)

# %%
rain_today_cols = [col for col in df.columns if col.startswith('RainToday_')]
for col in rain_today_cols:
    df[col] = df[col].astype(int)

# %%
y = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)
X = df.drop(columns=['RainTomorrow', 'Date', 'Year', 'Month', 'Day'])
print(X.head())
print(y.head())

# %%
print(X['WindDir9am'].unique())
print(X['WindDir3pm'].unique())

# %%
def encode_wind_directions(direction):
    direction_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

    degrees = direction_map.get(direction, 0)
    radians = np.radians(degrees)

    return np.sin(radians), np.cos(radians)

df[['WindDir9am_sin', 'WindDir9am_cos']] = df['WindDir9am'].apply(lambda x: pd.Series(encode_wind_directions(x)))
df[['WindDir3pm_sin', 'WindDir3pm_cos']] = df['WindDir3pm'].apply(lambda x: pd.Series(encode_wind_directions(x)))
df[['WindGusDir_sin', 'WindGustDir_cost']] = df['WindGustDir'].apply(lambda x: pd.Series(encode_wind_directions(x)))

# %%
y = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)
X = df.drop(columns=['RainTomorrow', 'Date', 'Year', 'Month', 'Day', 'WindDir9am', 'WindDir3pm', 'WindGustDir'])
print(X.head())
print(y.head())

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
print(X_train.shape)
print(X_test.shape)

# %%
print(X_train.describe())

# %%
def replace_outliers_with_mean(df):
    for col in df:
        if col in numerical_cols:
            IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
            lower_bound = df[col].quantile(0.25) - 1.5 * IQR
            upper_bound = df[col].quantile(0.75) + 1.5 * IQR
            mean_value = df[col].mean()
            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), mean_value, df[col])
        else:
            continue
    return df

X_train = replace_outliers_with_mean(X_train)
X_test = replace_outliers_with_mean(X_test)
print(X_train.describe())
print(X_test.describe())

# %%
# Check your target variable
print("Target variable (y) data type:", y.dtype)
print("Unique values in y:", y.unique())
print("Any non-numeric values in y:", y.apply(lambda x: not isinstance(x, (int, float))).sum())

# Debug what's in X before scaling
print("Shape of X:", X.shape)
print("\nData types in X:")
print(X.dtypes)
print("\nColumns in X:")
print(X.columns.tolist())

# Check for any remaining categorical/object columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns still in X: {categorical_cols}")

# If there are categorical columns, see what values they contain
for col in categorical_cols:
    print(f"\n{col} unique values: {X[col].unique()[:10]}")

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)
logreg100.fit(X_train, y_train)

# Get predictions
y_train_pred = logreg100.predict(X_train)
y_test_pred = logreg100.predict(X_test)

# Calculate the accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}") 
print(f"Accuracy Difference: {abs(train_accuracy - test_accuracy):.4f}")

cv_scores = cross_val_score(logreg100, X_train, y_train, cv=5)
print(f"Cross Validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# 6. Confusion Matrix
print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# %%
def diagnose_model_fit(train_acc, test_acc, cv_mean):
    """
    Diagnose if model is overfitting, underfitting, or well-fitted
    """
    accuracy_gap = train_acc - test_acc
    
    print("\n" + "="*40)
    print("MODEL DIAGNOSIS")
    print("="*40)
    
    if accuracy_gap > 0.05:  # 5% gap
        print("🚨 OVERFITTING DETECTED")
        print("- Training accuracy is significantly higher than test accuracy")
        print("- Model memorized training data rather than learning patterns")
        print("- Solutions: Increase regularization (lower C), get more data")
        
    elif test_acc < 0.6:  # Assuming 60% is reasonable baseline for your problem
        print("🚨 UNDERFITTING DETECTED") 
        print("- Both training and test accuracy are low")
        print("- Model is too simple to capture patterns")
        print("- Solutions: Decrease regularization (higher C), add features")
        
    elif abs(accuracy_gap) < 0.03:  # Less than 3% gap
        print("✅ GOOD FIT")
        print("- Training and test accuracy are close")
        print("- Model generalizes well to unseen data")
        
    else:
        print("⚠️ MODERATE OVERFITTING")
        print("- Some overfitting present but not severe")
        print("- Monitor performance, consider slight regularization increase")
    
    print(f"\nMetrics:")
    print(f"- Training Accuracy: {train_acc:.4f}")
    print(f"- Test Accuracy: {test_acc:.4f}")
    print(f"- CV Mean: {cv_mean:.4f}")
    print(f"- Accuracy Gap: {accuracy_gap:.4f}")

# Use the diagnosis function
diagnose_model_fit(train_accuracy, test_accuracy, cv_scores.mean())