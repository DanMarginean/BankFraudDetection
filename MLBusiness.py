import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN

from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
data_path = "creditcard.csv"
data = pd.read_csv(data_path)

# Distribuția claselor
label_counts = data['Class'].value_counts()
print(label_counts)

# Proporția fiecărei clase
label_proportions = data['Class'].value_counts(normalize=True)
print(label_proportions)

print(data.head())

# Preprocessing
X = data.drop(columns=['Class'])
y = data['Class']


# Datele inițiale
plt.subplot(1, 2, 1)
data['Class'].value_counts().plot(kind='bar', color=['blue', 'orange'], alpha=0.7)
plt.title('Date Inițiale')
plt.xlabel('Clase')
plt.ylabel('Număr de Tranzacții')
plt.xticks([0, 1], ['Legitimă (0)', 'Fraudă (1)'], rotation=0)

# Aplicarea RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
X = data.drop(columns=['Class'])
y = data['Class']
# undersampler = RandomUnderSampler(random_state=42)
undersampler = RandomUnderSampler(sampling_strategy=0.33, random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Datele după subsampling
plt.subplot(1, 2, 2)
pd.Series(y_resampled).value_counts().plot(kind='bar', color=['blue', 'orange'], alpha=0.7)
plt.title('După Subsampling')
plt.xlabel('Clase')
plt.ylabel('Număr de Tranzacții')
plt.xticks([0, 1], ['Legitimă (0)', 'Fraudă (1)'], rotation=0)

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Logistic Regression with scaled data
log_reg = LogisticRegression(max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
y_pred_proba_lr = log_reg.predict_proba(X_test)[:, 1]

# Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = nn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)

y_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
y_pred_proba_nn = nn_model.predict(X_test_scaled).flatten()


# First Plot: Logistic Regression
plt.figure(figsize=(12, 6))  # Create a new figure

# ROC Curve
plt.subplot(1, 4, 1)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'ROC curve (area = {roc_auc_lr:.2f})')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # Set square aspect ratio

# Confusion Matrix
plt.subplot(1, 4, 2)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=plt.gca(), colorbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.gca().set_aspect('equal', adjustable='box')  # Set square aspect ratio

# Metrics
plt.subplot(1, 4, 3)  # Metrics subplot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [
    accuracy_score(y_test, y_pred_lr),
    precision_score(y_test, y_pred_lr),
    recall_score(y_test, y_pred_lr),
    f1_score(y_test, y_pred_lr)
]

# Create the bar plot
plt.bar(metrics, values, color='skyblue')
plt.ylim(0, 1)
plt.title('Metrics - Logistic Regression')

# Annotate the bars with values
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=10)

# Adjust the subplot aspect ratio to make it visually balanced
ax = plt.gca()
ax.set_box_aspect(0.5)  # Set height to half the width for balance

# Precision-Recall Curve
plt.subplot(1, 4, 4)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
PrecisionRecallDisplay(precision=precision_lr, recall=recall_lr).plot(ax=plt.gca())
plt.title('Precision-Recall Curve - Logistic Regression')
plt.gca().set_aspect('equal', adjustable='box')  # Set square aspect ratio

plt.tight_layout()
plt.show()

# Second Plot: Neural Network
plt.figure(figsize=(12, 6))  # Create another new figure

# ROC Curve
plt.subplot(1, 4, 1)
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_proba_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)
plt.plot(fpr_nn, tpr_nn, label=f'ROC curve (area = {roc_auc_nn:.2f})')
plt.title('ROC Curve - Neural Network')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # Set square aspect ratio

# Confusion Matrix
plt.subplot(1, 4, 2)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_nn, ax=plt.gca(), colorbar=False)
plt.title('Confusion Matrix - Neural Network')
plt.gca().set_aspect('equal', adjustable='box')  # Set square aspect ratio

# Metrics
plt.subplot(1, 4, 3)  # Metrics subplot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [
    accuracy_score(y_test, y_pred_nn),
    precision_score(y_test, y_pred_nn),
    recall_score(y_test, y_pred_nn),
    f1_score(y_test, y_pred_nn)
]

# Create the bar plot
plt.bar(metrics, values, color='skyblue')
plt.ylim(0, 1)
plt.title('Metrics - Neural Network')

# Annotate the bars with values
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=10)

# Adjust the subplot aspect ratio to make it visually balanced
ax = plt.gca()
ax.set_box_aspect(0.5)  # Set height to half the width for balance

# Precision-Recall Curve
plt.subplot(1, 4, 4)
precision_nn, recall_nn, _ = precision_recall_curve(y_test, y_pred_proba_nn)
PrecisionRecallDisplay(precision=precision_nn, recall=recall_nn).plot(ax=plt.gca())
plt.title('Precision-Recall Curve - Neural Network')
plt.gca().set_aspect('equal', adjustable='box')  # Set square aspect ratio

plt.tight_layout()
plt.show()

# Pierderea
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Evoluția Pierderii')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Acuratețea
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Evoluția Acurateței')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()