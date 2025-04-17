import requests
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import time

# Obtener todo el histórico de datos de todas las sesiones
def obtener_datos_historicos(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        gran_premio = data[0]
        sesiones = gran_premio["competiciones"]["motoGp"]

        historico = []
        ultima_sesion_valida = None

        for sesion in sesiones:
            if "resultado" in sesion and sesion["resultado"]:
                for idx, piloto in enumerate(sesion["resultado"]):
                    historico.append({
                        "Posición": idx + 1,
                        "Piloto": piloto["nombre"],
                        "Equipo": piloto["equipo"],
                        "Tiempo": piloto["tiempo"],  # Este es el tiempo en segundos como decimal
                        "Sesion": sesion["descripcion"],
                        "Dia": sesion["dia"]
                    })
                ultima_sesion_valida = sesion

        if not historico:
            print("❌ No se encontró ninguna sesión con resultados.")
            return None, None

        df = pd.DataFrame(historico)
        return df, ultima_sesion_valida

    except requests.exceptions.RequestException as e:
        print(f"❌ Error al conectarse al API: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error procesando los datos: {e}")
        return None, None

def guardar_modelo(model, le_piloto, le_equipo, filename="modelo_entrenado.h5"):
    model.save(filename)
    joblib.dump((le_piloto, le_equipo), "encoders.pkl")
    print(f"✅ Modelo guardado en {filename}")

def cargar_modelo(filename="modelo_entrenado.h5"):
    try:
        tf.keras.utils.get_custom_objects().update({
            'mse': tf.keras.losses.MeanSquaredError()
        })
        model = tf.keras.models.load_model(filename)
        le_piloto, le_equipo = joblib.load("encoders.pkl")
        print(f"✅ Modelo cargado desde {filename}")
        return model, le_piloto, le_equipo
    except FileNotFoundError:
        print("❌ No se encontró el modelo guardado. Se entrenará desde cero.")
        return None, None, None

def entrenar_y_predecir_top5(df, model=None, le_piloto=None, le_equipo=None):
    if le_piloto is None:
        le_piloto = LabelEncoder()
        df["Piloto_encoded"] = le_piloto.fit_transform(df["Piloto"])
    else:
        df["Piloto_encoded"] = le_piloto.transform(df["Piloto"])

    if le_equipo is None:
        le_equipo = LabelEncoder()
        df["Equipo_encoded"] = le_equipo.fit_transform(df["Equipo"])
    else:
        df["Equipo_encoded"] = le_equipo.transform(df["Equipo"])

    # El tiempo ya está en formato decimal de segundos, así que lo usamos directamente
    df["Tiempo_segundos"] = pd.to_numeric(df["Tiempo"], errors="coerce")
    df = df.dropna(subset=["Tiempo_segundos"])

    X = df[["Piloto_encoded", "Equipo_encoded", "Tiempo_segundos"]]
    y = df["Posición"]

    if model is None:
        model = Sequential([
            Dense(128, input_dim=X.shape[1], activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(), loss='mse')

    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
        print(f"📊 Evaluación del modelo (MSE): {model.evaluate(X_test, y_test):.2f}")
    else:
        model.fit(X, y, epochs=50, batch_size=10, verbose=1)

    return model, le_piloto, le_equipo

def predecir_top5(model, le_piloto, le_equipo, sesion):
    df_pred = pd.DataFrame([{
        "Piloto": piloto["nombre"],
        "Equipo": piloto["equipo"],
        "Tiempo": piloto["tiempo"]
    } for piloto in sesion["resultado"]])

    df_pred["Piloto_encoded"] = df_pred["Piloto"].apply(
        lambda x: le_piloto.transform([x])[0] if x in le_piloto.classes_ else 0
    )
    df_pred["Equipo_encoded"] = df_pred["Equipo"].apply(
        lambda x: le_equipo.transform([x])[0] if x in le_equipo.classes_ else 0
    )

    # El tiempo ya viene en decimal de segundos, lo usamos directamente
    df_pred["Tiempo_segundos"] = pd.to_numeric(df_pred["Tiempo"], errors="coerce")
    df_pred = df_pred.dropna(subset=["Tiempo_segundos"])

    X = df_pred[["Piloto_encoded", "Equipo_encoded", "Tiempo_segundos"]]
    df_pred["PosiciónPredicha"] = model.predict(X)

    top5 = df_pred.sort_values("PosiciónPredicha").head(5)
    print("\n🏁 Top 5 Predicho:")
    print(top5[["Piloto", "Equipo", "PosiciónPredicha"]])

# Proceso automático
def ejecutar_entrenamiento_continuo(api_url, intervalo_segundos=3600):
    model, le_piloto, le_equipo = cargar_modelo()

    while True:
        print("\n📡 Obteniendo histórico de sesiones...")
        df_hist, sesion_pred = obtener_datos_historicos(api_url)

        if df_hist is not None and sesion_pred is not None:
            print("✅ Datos cargados correctamente.")
            model, le_piloto, le_equipo = entrenar_y_predecir_top5(df_hist, model, le_piloto, le_equipo)
            guardar_modelo(model, le_piloto, le_equipo)
            predecir_top5(model, le_piloto, le_equipo, sesion_pred)
        else:
            print("⚠️ No se pudo entrenar ni predecir.")

        print(f"💤 Esperando {intervalo_segundos // 60} minutos...")
        time.sleep(intervalo_segundos)

# Ejecutar
if __name__ == "__main__":
    url = "http://localhost:5050/dataTrain"
    ejecutar_entrenamiento_continuo(url, intervalo_segundos=75)
