import numpy as np
import time

class Trainer:
    
    def __init__(self, model, learning_rate, batch_size, epochs):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

    def train(self, x_train, y_train, x_test, y_test, dataset_name="Dataset"):
        
        print(f"\nTraining on {dataset_name}...")
        
        # Training loop
        num_train = x_train.shape[0]
        num_batches = num_train // self.batch_size   # integer division

        for epoch in range(self.epochs):     # dataset loop
            start_time = time.time()

            # shuffle data each epoch
            indices = np.random.permutation(num_train)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0.0
            epoch_acc = 0.0

            for i in range(num_batches):        # single epoch loop
                start = i * self.batch_size
                end = start + self.batch_size

                x_batch = x_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                loss, acc = self.model.forward(x_batch, y_batch)

                self.model.backward()
                self.model.step(self.learning_rate)
                epoch_loss += loss
                epoch_acc += acc

            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches

            # test set eval
            test_loss, test_acc = self.model.forward(x_test, y_test)

            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)

            print(f"[{dataset_name}] Epoch {epoch+1}/{self.epochs} - "
                  f"Time: {time.time()-start_time:.2f}s - "
                  f"Train Loss: {avg_loss:.4f} - Test Acc: {test_acc:.4f}")
                  
        return self.history
