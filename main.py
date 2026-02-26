import ssl
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from model import MLP
from trainer import Trainer
from plot import plot_comparison

# Fix SSL for successful dataset import
ssl._create_default_https_context = ssl._create_unverified_context

def load_data(name):
    print(f"Downloading {name} from OpenML...")
    data_id = 'mnist_784' if name == 'MNIST' else 'Fashion-MNIST'
    dataset = fetch_openml(data_id, version=1, as_frame=False, parser='auto')
    X = dataset.data
    y = dataset.target.astype(int)
    
    # Scale
    X = X / 255.0       # greyscale has pixel values between 0 and 255
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
    return x_train, y_train, x_test, y_test

def main():
    # Configuration
    input_dim = 784
    hidden_dims = [128, 64] # 2 Hidden Layers
    output_dim = 10
    
    datasets = ['MNIST', 'FashionMNIST']
    histories = {}

    for name in datasets:
        # 1. Load Data
        x_train, y_train, x_test, y_test = load_data(name)
        
        # 2. Initialize Model
        print(f"Initializing MLP for {name}...")
        model = MLP(input_dim, hidden_dims, output_dim)
        
        # 3. Train
        trainer = Trainer(model, learning_rate=0.1, batch_size=64, epochs=15)
        history = trainer.train(x_train, y_train, x_test, y_test, dataset_name=name)
        
        histories[name] = history

    # 4. Compare
    print("\nGenerating comparison plots...")
    plot_comparison(histories)

if __name__ == "__main__":
    main()
