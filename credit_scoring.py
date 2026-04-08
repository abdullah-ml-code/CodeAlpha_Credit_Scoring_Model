import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. تحميل البيانات (سنستخدم رابط مباشر لبيانات UCI الشهيرة)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
columns = [f'Feature_{i}' for i in range(24)] + ['Target']
df = pd.read_csv(url, sep='\s+', names=columns)

# تحويل Target إلى 0 و 1 (1 = جيد، 0 = سيء)
df['Target'] = df['Target'].replace({1: 1, 2: 0})

print("✅ تم تحميل البيانات بنجاح!")

# 2. تقسيم البيانات (Training & Testing)
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. بناء النموذج (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. التنبؤ والتقييم
y_pred = model.predict(X_test)

print("\n📊 تقرير أداء النموذج:")
print(classification_report(y_test, y_pred))

# 5. رسم مصفوفة الارتباك (Confusion Matrix) لشكل احترافي
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Credit Scoring')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()