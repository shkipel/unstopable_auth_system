import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 1. Загрузка легитимных данных
learn_df = pd.read_csv('learn2_ds.csv')
features = ['hold_time', 'latency']
X_legit = learn_df[features].values
y_legit = np.ones(len(X_legit))  # метки 1 (легитимные)

# 2. Загрузка всех нелегитимных датасетов
test_files = ['test_ds_sanya.csv', 'test_ds_ann.csv', 'test_ds_slow.csv']
X_fake_list = []

for file in test_files:
    fake_df = pd.read_csv(file)
    X_fake_list.append(fake_df[features].values)

X_fake = np.vstack(X_fake_list)
y_fake = np.zeros(len(X_fake))  # метки 0 (нелегитимные)

# 3. Объединяем обучающую выборку
X_train = np.vstack([X_legit, X_fake])
y_train = np.hstack([y_legit, y_fake])

# 4. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Обучение моделей
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# 6. Тестируем на leg_test.csv
test_df = pd.read_csv('test_ds_slow.csv')
X_test = test_df[features].values
X_test_scaled = scaler.transform(X_test)

knn_probs = knn.predict_proba(X_test_scaled)[:, 1]  # вероятность легитимности
logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]

knn_mean_prob = np.mean(knn_probs)
logreg_mean_prob = np.mean(logreg_probs)

print(f"=== LEG_TEST ===")
print(f"KNN: Средняя вероятность, что пользователь легитимен: {knn_mean_prob:.4f}")
print(f"Logistic Regression: Средняя вероятность, что пользователь легитимен: {logreg_mean_prob:.4f}")
