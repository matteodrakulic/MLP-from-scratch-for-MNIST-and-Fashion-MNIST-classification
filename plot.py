import matplotlib.pyplot as plt
import os

def plot_comparison(histories, save_path='plot.png'):
    
    plt.figure(figsize=(12, 5))

    # Plot Test Loss Comparison
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        epochs = range(1, len(history['test_loss']) + 1)
        plt.plot(epochs, history['test_loss'], label=f'{name} Test Loss')
    
    plt.title('Test Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Test Accuracy Comparison
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        epochs = range(1, len(history['test_acc']) + 1)
        plt.plot(epochs, history['test_acc'], label=f'{name} Test Acc')
        
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    try:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Error saving plot: {e}")
