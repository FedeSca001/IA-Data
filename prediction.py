import pandas as pd
import numpy as np
import json
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# ⚙️ Listas base
pilotos = [
    "Marc Márquez", "Álex Márquez", "Francesco Bagnaia", "Franco Morbidelli",
    "Fabio Di Giannantonio", "Johann Zarco", "Marco Bezzecchi", "Fabio Quartararo",
    "Ai Ogura", "Luca Marini", "Pedro Acosta", "Brad Binder", "Enea Bastianini",
    "Fermín Aldeguer", "Jack Miller", "Alex Rins", "Joan Mir", "Maverick Viñales",
    "Raúl Fernández", "Miguel Oliveira", "Lorenzo Savadori", "Somkiat Chantra"
]
equipos = ["Honda", "Ducati", "Yamaha", "KTM", "Suzuki", "Aprilia"]

# 🧩 Función para procesar JSON y devolver DataFrame
def procesar_json(ruta_json):
    with open(ruta_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    filas = []
    for carrera in raw:
        sesiones = carrera.get('competiciones', {}).get('motoGp', [])
        for sesion in sesiones:
            resultados = sesion.get('resultado', [])
            for i, piloto in enumerate(resultados):
                try:
                    nombre = piloto['nombre'].strip().title()
                    equipo = piloto['equipo'].strip().title()
                    tiempo = float(piloto['tiempo'])
                    posicion = i + 1
                    top3 = 1 if i < 3 else 0

                    if nombre not in pilotos or equipo not in equipos:
                        continue

                    filas.append({
                        "piloto": nombre,
                        "equipo": equipo,
                        "tiempo": tiempo,
                        "posicion": posicion,
                        "top3": top3
                    })
                except:
                    continue
    return pd.DataFrame(filas)

# 💾 Función para guardar y actualizar CSV
def actualizar_csv(df_nuevo, ruta_csv="datos.csv"):
    if os.path.exists(ruta_csv):
        df_existente = pd.read_csv(ruta_csv)
        df = pd.concat([df_existente, df_nuevo]).drop_duplicates().reset_index(drop=True)
    else:
        df = df_nuevo
    df.to_csv(ruta_csv, index=False)
    return df

# 🧠 Entrenamiento con red neuronal
def entrenar_red_neuronal(df):
    # 🧹 Limpiar datos incompletos o desconocidos
    df = df.dropna(subset=["piloto", "equipo", "tiempo", "posicion", "top3"])
    df = df[df["piloto"].isin(pilotos)]
    df = df[df["equipo"].isin(equipos)]

    # 🧠 Codificación
    df["piloto_num"] = df["piloto"].apply(lambda x: pilotos.index(x))
    df["equipo_num"] = df["equipo"].apply(lambda x: equipos.index(x))

    X = df[["piloto_num", "equipo_num", "tiempo", "posicion"]].astype(float).values
    y = df["top3"].astype(float).values

    # Normalización
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Red neuronal
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenamiento con ajuste de peso por clase (balance de clases)
    class_weights = {0: 1, 1: 3}  # Aumentar el peso de la clase 1 (top3 = 1) si los datos están desbalanceados
    model.fit(X, y, epochs=500, verbose=1, class_weight=class_weights)

    return model, scaler

# 🔮 Predicción individual
def predecir_piloto(modelo, scaler, df, nombre):
    piloto_data = df[df["piloto"] == nombre].iloc[-1]  # Último registro
    entrada = np.array([[pilotos.index(nombre), equipos.index(piloto_data["equipo"]),
                         piloto_data["tiempo"], piloto_data["posicion"]]])
    entrada = scaler.transform(entrada)
    prob = float(modelo.predict(entrada)[0][0])
    print(f"Probabilidad de que {nombre} esté en el podio: {prob:.2%}")
    return prob

# 🔁 Main
if __name__ == "__main__":
    df_nuevo = procesar_json("datosFijos.json")
    df_total = actualizar_csv(df_nuevo)
    modelo, scaler = entrenar_red_neuronal(df_total)

    piloto_objetivo = "Álex Márquez"
    if piloto_objetivo in df_total["piloto"].values:
        predecir_piloto(modelo, scaler, df_total, piloto_objetivo)
    else:
        print(f"No hay datos previos para {piloto_objetivo}")
