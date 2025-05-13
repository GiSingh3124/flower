import flwr as fl
import tensorflow as tf
import numpy as np

# Patch per il dataset Iris che non esiste direttamente in tf.keras.datasets
def iris_dataset():
    # Carica il dataset Iris dalla libreria scikit-learn
    from sklearn.datasets import load_iris
    data = load_iris()
    
    # Suddividi in train e test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    return (x_train, y_train), (x_test, y_test)

# Aggiungi il metodo al modulo tf.keras.datasets
tf.keras.datasets.iris_dataset = iris_dataset

# Carica un dataset di esempio (Iris) per il test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.iris_dataset()

# Normalizza i dati
x_train = x_train / np.max(x_train)
x_test = x_test / np.max(x_test)

# Converti le etichette in formato one-hot
y_train = tf.keras.utils.to_categorical(y_train, 3)
y_test = tf.keras.utils.to_categorical(y_test, 3)

# Definisci il modello di ML (un semplice classificatore)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(4,), activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Classe client Flower
class IrisClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = create_model()
        
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        # Imposta i parametri ricevuti dal server
        self.model.set_weights(parameters)
        
        # Addestra il modello sul dataset locale
        history = self.model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Restituisci i pesi aggiornati e altre metriche
        return self.model.get_weights(), len(x_train), {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1]
        }
    
    def evaluate(self, parameters, config):
        # Imposta i parametri ricevuti dal server
        self.model.set_weights(parameters)
        
        # Valuta il modello sul dataset di test locale
        loss, accuracy = self.model.evaluate(x_test, y_test)
        
        # Restituisci le metriche di valutazione
        return loss, len(x_test), {"accuracy": accuracy}

# Avvia il client e connettilo al server
fl.client.start_numpy_client(
    server_address="localhost:8080", 
    client=IrisClient()
)