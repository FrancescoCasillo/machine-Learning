import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Caricamento del dataset
df = pd.read_csv("car_price_dataset.csv")

# Creazione della variabile target
median_price = df["Price"].median()
df["Price_Category"] = df["Price"].apply(lambda x: "Economica" if x < median_price else "Costosa")

# Rimozione della colonna originale "Price"
df = df.drop(columns=["Price"])

# Encoding delle variabili categoriche
label_encoders = {}
for col in ["Brand", "Model", "Fuel_Type", "Transmission", "Price_Category"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separazione feature e target
X = df.drop(columns=["Price_Category"])
y = df["Price_Category"]

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione delle feature numeriche
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Addestramento del modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predizione e valutazione
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Matrice di confusione
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["Economica", "Costosa"], yticklabels=["Economica", "Costosa"])
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione")
plt.legend(["Economica = Prezzo sotto la mediana", "Costosa = Prezzo sopra la mediana"])
plt.show()

# Istogramma della distribuzione delle categorie di prezzo
plt.figure(figsize=(8,5))
sns.countplot(x=df["Price_Category"], palette="Blues")
plt.xlabel("Categoria di Prezzo")
plt.ylabel("Numero di Auto")
plt.title("Distribuzione delle Categorie di Prezzo")
plt.legend(["Economica = Prezzo sotto la mediana", "Costosa = Prezzo sopra la mediana"])
plt.show()

# Boxplot del chilometraggio per categoria di prezzo
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Price_Category"], y=df["Mileage"], palette="coolwarm")
plt.xlabel("Categoria di Prezzo")
plt.ylabel("Chilometraggio")
plt.title("Distribuzione del Chilometraggio per Categoria di Prezzo")
plt.legend(["Economica", "Costosa "])
plt.show()

# Grafico a barre per tipo di trasmissione e categoria di prezzo
plt.figure(figsize=(8,5))
sns.countplot(x=df["Transmission"], hue=df["Price_Category"], palette="Set3")
plt.xlabel("Tipo di Trasmissione")
plt.ylabel("Numero di Auto")
plt.title("Distribuzione del Tipo di Trasmissione per Categoria di Prezzo")
plt.legend(title="Categoria di Prezzo e Categoria di Carburante", labels=["Economica", "Costosa"])
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x=df["Fuel_Type"], hue=df["Price_Category"], palette="Set3")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Numero di Auto")
plt.title("Distribuzione del Tipo di Carburante per Categoria di Prezzo")
plt.legend(title="Categoria di Prezzo", labels=["Economica", "Costosa"])
plt.xticks(rotation=0)
plt.show()

