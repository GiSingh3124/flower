import flwr as fl
import tensorflow as tf
from typing import Dict, Optional, Tuple
import numpy as np

# Definizione della strategia di aggregazione
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        # Aggregazione standard (media ponderata dei pesi)
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights is not None:
            # Salva il modello dopo ogni round
            print(f"Saving round {server_round} aggregated_weights...")
            
            # Costruisci un modello semplice per testare i pesi aggregati
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, input_shape=(4,), activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax")
            ])
            
            # Imposta i pesi aggregati nel modello
            model.set_weights(aggregated_weights)
            
            # Salva il modello
            model.save(f"model_round_{server_round}")
        
        return aggregated_weights

# Definisci la strategia con parametri personalizzati
strategy = SaveModelStrategy(
    min_fit_clients=1,  # Numero minimo di client per il training in ogni round
    min_available_clients=1,  # Numero minimo di client che devono essere disponibili
    # min_eval_clients è stato rimosso perché non supportato
)

# Avvia il server
fl.server.start_server(
    server_address="localhost:8080",  # Ascolta su tutte le interfacce di rete, porta 8080
    config=fl.server.ServerConfig(num_rounds=3),  # Esegui 3 round di training
    strategy=strategy
)