import flwr as fl
import sys

# Define a simple Flower strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # Train on all available clients
    min_fit_clients=1,          # Set to 1 for testing, increase for real use
    min_available_clients=1,    
)

if __name__ == "__main__":
    print("Starting DeepScan Federated Server...")
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10), # Total "global" training steps
        strategy=strategy,
    )
