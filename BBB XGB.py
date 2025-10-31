import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import PredictionErrorDisplay
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = ' Times New Roman, SimSun'
df = pd.read_csv('BBB.CSV')
X = df.drop(['degradation rate'], axis=1)
Y = df['degradation rate']

X_char = X.select_dtypes(include=['object'])


encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_char_encoded = encoder.fit_transform(X_char)
X_char_encoded_df = pd.DataFrame(X_char_encoded.toarray(), columns=encoder.get_feature_names_out(X_char.columns))
X_final = pd.concat([X_char_encoded_df, X.drop(X_char.columns, axis=1)], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_final, Y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBRegressor()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
X_final_scaled = scaler.transform(X_final)
Y_pred = model.predict(X_final_scaled)

fig, ax = plt.subplots()
ax.scatter(y_train, y_train_pred, color='#7895C1', label='Train')
ax.scatter(y_test, y_test_pred, color='#74070E', label='Test')
ax.plot([-0.01, 1], [-0.01, 1], "--k")
ax.set_ylabel("Target predicted(×100%)", fontweight='bold', fontsize=16)
ax.set_xlabel("True Target(×100%)", fontweight='bold', fontsize=16)
ax.set_title("XGB Performance", fontweight='bold')
ax.text(
    0.05, 0.8,
    r"$R^2$=%.2f, MAE=%.2f, RMSE=%.2f"
    % (r2_score(Y, Y_pred), mean_absolute_error(Y, Y_pred), np.sqrt(mean_squared_error(Y, Y_pred))),
    fontweight='bold'
)
ax.set_xlim([-0.01, 1])
ax.set_ylim([-0.01, 1])
font = FontProperties(weight='bold')
ax.legend(loc='best', prop=font)
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in')
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(axis='both', which='major', length=4, width=2)
ax.tick_params(axis='both', labelsize=17)
for tick in ax.get_xticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_weight('bold')
for tick in ax.get_yticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_weight('bold')

plt.show()


kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'))
print("R2 均值:", r2_scores.mean())
print("R2 标准差:", r2_scores.std())
print("RMSE 均值:", rmse_scores.mean())
print("RMSE 标准差:", rmse_scores.std())
test_rmse = root_mean_squared_error(y_test, y_test_pred)
print("测试集 RMSE:", test_rmse)
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals_train, color='#7895C1', alpha=0.5, label='Train')
plt.scatter(y_test_pred, residuals_test, color='#74070E', alpha=0.5, label='Test')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel("Predicted Values", fontweight='bold', fontsize=16)
plt.ylabel("Residuals", fontweight='bold', fontsize=16)
plt.title("Residual Plot (XGB)", fontweight='bold')
font = FontProperties(weight='bold')
plt.legend(loc='best', prop=font)
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tick_params(axis='both', which='major', length=4, width=2)
plt.tick_params(axis='both', labelsize=17)
for tick in plt.gca().get_xticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_weight('bold')
for tick in plt.gca().get_yticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_weight('bold')
plt.show()