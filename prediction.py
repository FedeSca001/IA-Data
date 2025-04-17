import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json

# Lista de pilotos de MotoGP (puedes modificarla según tu necesidad)
pilotos = [
    "Marc Márquez", "Álex Márquez", "Francesco Bagnaia", "Franco Morbidelli",
    "Fabio Di Giannantonio", "Johann Zarco", "Marco Bezzecchi", "Fabio Quartararo",
    "Ai Ogura", "Luca Marini", "Pedro Acosta", "Brad Binder", "Enea Bastianini",
    "Fermín Aldeguer", "Jack Miller", "Alex Rins", "Joan Mir", "Maverick Viñales",
    "Raúl Fernández", "Miguel Oliveira", "Lorenzo Savadori", "Somkiat Chantra"
]

# Lista de equipos (deberás incluir los equipos reales de la lista)
equipos = [
    "Honda", "Ducati", "Yamaha", "KTM", "Suzuki", "Aprilia",
]


# 🔹 Cargar y procesar los datos del archivo JSON
def cargar_datos_json(ruta='datosFijos.json'):
    with open(ruta, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    X = []  # features
    y = []  # etiquetas (1 = top 3, 0 = no top 3)
    pilotos_features = {}  # Mapeo nombre → features

    for carrera in raw:
        sesiones = carrera.get('competiciones', {}).get('motoGp', [])
        for sesion in sesiones:
            resultados = sesion.get('resultado', [])

            try:
                tiempos_ordenados = sorted([float(p['tiempo']) for p in resultados])
                if len(tiempos_ordenados) < 3:
                    continue  # Saltamos si hay menos de 3 pilotos

                for i, piloto in enumerate(resultados):
                    try:
                        nombre = piloto['nombre']
                        tiempo = float(piloto['tiempo'])
                        equipo = piloto['equipo']
                        posicion = i + 1

                        # 🔹 Asignar el número del piloto y equipo
                        piloto_numero = pilotos_numeros.get(nombre, -1)  # Número del piloto
                        equipo_numero = equipos_numeros.get(equipo, -1)  # Número del equipo

                        # Si el piloto o equipo no tiene un número asignado, saltamos este piloto
                        if piloto_numero == -1 or equipo_numero == -1:
                            continue

                        # 🔹 Vector de características: Excluimos el nombre, usamos los números
                        features = [piloto_numero, tiempo, equipo_numero, posicion]

                        # Aquí asignamos las características de cada piloto a pilotos_features
                        pilotos_features[nombre] = features  # Agregamos al diccionario

                        X.append(features)
                        y.append([1 if i < 3 else 0])  # Podio: 1 si está en top 3
                        print(pilotos_features, '-*-*-*-*-*-*-*-*-*-*-*')  # Muestra de los pilotos y sus características
                    except Exception as e:
                        print(f"Error procesando piloto: {piloto} -> {e}")
            except Exception as e:
                print(f"Error procesando tiempos: {e}")

    print(f"Total muestras: {len(X)}")
    return np.array(X, dtype=float), np.array(y, dtype=float), pilotos_features

# 🔹 Crear y entrenar modelo
def entrenar_modelo(X, y, epochs=500):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Salida binaria
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, verbose=1)
    return model

# 🔹 Predicción individual
def predecir(modelo, entrada, nombre):
    resultado = modelo.predict(np.array([entrada], dtype=float))
    prob = float(resultado[0][0])
    if nombre:
        print(f"Probabilidad de que {nombre} esté en el podio: {prob:.2%}")
    else:
        print("Probabilidad de estar en podio:", prob)
    return prob

# 🔹 Ranking de probabilidades
def ranking_probabilidades(modelo, pilotos_features):
    resultados = []
    for nombre, features in pilotos_features.items():
        prob = predecir(modelo, features)
        resultados.append((nombre, prob))
    resultados.sort(key=lambda x: x[1], reverse=True)

    print("\n🏁 Ranking de probabilidad de podio:")
    for i, (nombre, prob) in enumerate(resultados[:10], 1):
        print(f"{i}. {nombre}: {prob:.2%}")

# 🔹 Main
if __name__ == "__main__":
    X, y, pilotos_features = cargar_datos_json()        # Procesamos el JSON
    modelo = entrenar_modelo(X, y)                       # Entrenamos el modelo
    #ranking_probabilidades(modelo, pilotos_features)     # Mostramos el ranking top 10

    # 🔹 Predicción para piloto específico
    piloto_objetivo = "Fabio Quartararo"
    
    if piloto_objetivo in pilotos_features:
        predecir(modelo, pilotos_features[piloto_objetivo], nombre=piloto_objetivo)
        print(pilotos_features[piloto_objetivo])  # Asegúrate de que sea una lista de números
    else: #[nombre, tiempo, equipo, posicion]
        print(f"No se encontró información de {piloto_objetivo}")
