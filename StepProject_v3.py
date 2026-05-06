import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from scipy import stats
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv"
response = requests.get(url)
if response.status_code == 200:
    print("Файл успішно завантажено")
else:
    print("Помилка завантаження")
    print("Status code:", response.status_code)
    exit()
df = pd.read_csv(StringIO(response.text))

if "Unnamed: 0" in df.columns:                                                                     
    df = df.drop("Unnamed: 0", axis=1)

df = df.drop_duplicates(subset=["item_id"])
df['designer'] = df['designer'].str.replace(r'\d+', '', regex=True)
df['designer'] = df['designer'].str.strip()
df.loc[df['designer'].str.len() > 40, 'designer'] = np.nan
df['designer'] = df['designer'].fillna("Unknown")                                                                                                                                      #перетворення на число
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df[['depth','height','width']] = df[['depth','height','width']].replace(0, np.nan)
df['category'] = df['category'].str.strip()


print("Описова статистика:")
print(df.describe().to_markdown())





category_counts = df["category"].value_counts()
plt.figure(figsize=(10, 5))
category_counts.plot(kind="bar")
plt.title("Кількість товарів по категоріях")
plt.xlabel("Категорія")
plt.ylabel("Кількість товарів")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()


median_price_by_category = df.groupby("category")["price"].median().sort_values()
plt.figure(figsize=(10, 5))
median_price_by_category.plot(kind="bar")
plt.title("Медіанна ціна товарів по категоріях")
plt.xlabel("Категорія")
plt.ylabel("Медіанна ціна")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=50)
plt.title("Розподіл ціни товарів")
plt.xlabel("Ціна")
plt.ylabel("Кількість")
plt.tight_layout()
plt.show()


price_compare = df[df["old_price"].notna()]
df['old_price'] = df['old_price'].replace(r'[^\d.]', '', regex=True)
df['old_price'] = df['old_price'].replace('', np.nan)
df['old_price'] = df['old_price'].astype(float)
price_compare = df.dropna(subset=['old_price', 'price'])
plt.figure(figsize=(7, 5))
sns.regplot(
    x=price_compare["old_price"],
    y=price_compare["price"],
    scatter_kws={"alpha": 0.3}
)
plt.title("Співвідношення old_price та price")
plt.xlabel("Стара ціна")
plt.ylabel("Нова ціна")
plt.tight_layout()
plt.show()


color_counts = df["other_colors"].value_counts()
plt.figure(figsize=(6, 4))
color_counts.plot(kind="bar")
plt.title("Кількість товарів з іншими кольорами")
plt.xlabel("Інші кольори")
plt.ylabel("Кількість")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


median_price_colors = df.groupby("other_colors")["price"].median()
plt.figure(figsize=(6, 4))
median_price_colors.plot(kind="bar")
plt.title("Медіанна ціна: з / без інших кольорів")
plt.xlabel("Інші кольори")
plt.ylabel("Медіанна ціна")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


top_designers = df["designer"].value_counts().head(15)
plt.figure(figsize=(10, 5))
top_designers.plot(kind="bar")
plt.title("Топ-15 дизайнерів за кількістю товарів")
plt.xlabel("Дизайнер")
plt.ylabel("Кількість товарів")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()


top_designers_names = top_designers.index
median_price_designers = (
    df[df["designer"].isin(top_designers_names)]
    .groupby("designer")["price"]
    .median()
    .sort_values()
)
plt.figure(figsize=(10, 5))
median_price_designers.plot(kind="bar")
plt.title("Медіанна ціна товарів по дизайнерах (топ-15)")
plt.xlabel("Дизайнер")
plt.ylabel("Медіанна ціна")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()


corr_df = df[['price', 'depth', 'height', 'width']].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title("Кореляційна матриця: Ціна та розміри")
plt.tight_layout()
plt.show()


online_counts = df['sellable_online'].value_counts()
plt.figure(figsize=(5,4))
online_counts.plot(kind='bar')
plt.title("Кількість товарів online / offline")
plt.xlabel("Продаається онлайн")
plt.ylabel("Кількість")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


median_online = df.groupby('sellable_online')['price'].median()
plt.figure(figsize=(5,4))
median_online.plot(kind='bar')
plt.title("Медіанна ціна online/offline")
plt.xlabel("Продається онлайн")
plt.ylabel("Медіанна ціна")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()








print("Гіпотеза 1: Чи відрізняється ціна товарів з іншими кольорами?")
price_with_colors = df[df["other_colors"] == "Yes"]["price"]
price_without_colors = df[df["other_colors"] == "No"]["price"]
print("Медіана з іншими кольорами:", price_with_colors.median())
print("Медіана без інших кольорів:", price_without_colors.median())

u_stat, p_value = stats.mannwhitneyu(price_with_colors, price_without_colors)
print("U-test")
print("p_value =", p_value)

log_with = np.log(price_with_colors)
log_without = np.log(price_without_colors)
t_stat, p_value_log = stats.ttest_ind(log_with, log_without)
print("T-test після логарифмування")
print("p_value =", p_value_log)

a = 0.05
if p_value < a:
    print("U-test: Є значна різниця.")
else:
    print("U-test: Різниці немає.")
if p_value_log < a:
    print("T-test: Є значна різниця.")
else:
    print("T-test: Різниці немає.")


print("Гіпотеза 2: Чи відрізняється ціна великих та малих товарів?")
df['volume'] = df['depth'] * df['height'] * df['width']
df['volume'] = df['volume'].fillna(df['volume'].median())
median_volume = df['volume'].median()

small_items = df[df['volume'] < median_volume]['price']
large_items = df[df['volume'] >= median_volume]['price']

print("Медіанна ціна малих товарів:", small_items.median())
print("Медіанна ціна великих товарів:", large_items.median())

u_stat, p_value = stats.mannwhitneyu(small_items, large_items)
print("U-test p-value:", p_value)

log_small = np.log1p(small_items)
log_large = np.log1p(large_items)

t_stat, p_value_t = stats.ttest_ind(log_small, log_large)
print("T-test після логарифмування p-value:", p_value_t)
alpha = 0.05
if p_value < alpha:
    print("U-test: Є значна різниця.")
else:
    print("U-test: Різниці немає.")
if p_value_t < alpha:
    print("T-test: Є значна різниця.")
else:
    print("T-test: Різниці немає.")







df_ml = df.copy()
df_ml = df_ml.drop(columns=['item_id', 'old_price', 'link',
                            'short_description', 'name', 'sellable_online'])
numeric_cols = ['depth', 'height', 'width']
for col in numeric_cols:
    df_ml[col] = df_ml[col].fillna(df_ml[col].median())

X = df_ml.drop('price', axis=1)
y = df_ml['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

categorical_cols = ['category', 'designer', 'other_colors']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), ['depth', 'height', 'width'])
    ],
    remainder='drop'
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(max_depth=10, random_state=42)
}
results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results.append([name, r2, rmse])

results_df = pd.DataFrame(results, columns=['Model', 'R2', 'RMSE'])
print(results_df)

plt.figure(figsize=(6,4))
plt.bar(results_df['Model'], results_df['R2'])
plt.title("Порівняння моделей (R2)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10]
}
grid = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)

print("Найкращі параметри:", grid.best_params_)
print("Найкращий R2:", grid.best_score_)

best_model = grid.best_estimator_

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')

print("Крос валідація R2:", cv_scores)
print("Середній R2:", cv_scores.mean())

plt.figure(figsize=(6,4))
plt.plot(range(1, len(cv_scores)+1), cv_scores, marker='o')
plt.title("Крос валідація R2 (Train only)")
plt.xlabel("")
plt.ylabel("R2")
plt.tight_layout()
plt.show()

best_model.fit(X_train, y_train)

y_test_pred = best_model.predict(X_test)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mse = mean_squared_error(y_test, y_test_pred)

print("Тест R2:", test_r2)
print("Тест RMSE:", test_rmse)
print("Тест MSE:", test_mse)






rf_model = best_model.named_steps['model']
importances = rf_model.feature_importances_

feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances
                              }).sort_values(by='Importance', ascending=False)

importance_df['Group'] = importance_df['Feature'].apply(
    lambda x: 'category' if 'category' in x
    else 'designer' if 'designer' in x
    else 'other_colors' if 'other_colors' in x
    else 'width' if 'width' in x
    else 'height' if 'height' in x
    else 'depth'
)

grouped_importance = importance_df.groupby('Group')['Importance'].sum()
plt.figure(figsize=(6,4))
grouped_importance.sort_values(ascending=False).plot(kind='bar')
plt.title("Важливість факторів")
plt.ylabel("Важливість")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(importance_df.head(10))

print("Кінець")
