import torch as t
from torch import nn
import matplotlib.pyplot as plt
from model import LinearRegressionModelV2
from pprint import pprint
import time

# Setup Cuda
device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def main():

    ## 1. Pre-Processing
    # Initialize parameters
    weights = 0.7
    bias = 0.3

    # Create features and labels
    X = t.arange(0,1,0.02).unsqueeze(dim=1) # 50 x 1
    y = weights * X + bias
    #print(X)

    # Train test split
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    ## 2. Model
    # random seed and model Instance
    t.manual_seed(42)
    model_1 = LinearRegressionModelV2()
    #print(model_1, model_1.state_dict())
    model_1.to(device) # Put model to cuda if available
    #print(next(model_1.parameters()).device) 

    # Create loss function and optimizer
    loss_fn = nn.L1Loss()
    optimizer = t.optim.Adam(params= model_1.parameters(), lr = 5e-2)


    ## 3. Training and Testing 
    # Put data in cuda if available
    t.manual_seed(42)
    epochs = 1000
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    tic = time.time()
    for epoch in range(epochs):
        
        # Training
        model_1.train() #train mode
        y_pred = model_1(X_train) # forward pass
        loss = loss_fn(y_pred, y_train) # loss 
        optimizer.zero_grad() # initialize gradient with zero
        
        loss.backward() # backward pass
        optimizer.step() # update parameters

        # Testing
        model_1.eval() # Evaluation mode
        with t.inference_mode():
            test_pred = model_1(X_test)
            test_loss = loss_fn(test_pred, y_test)
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

    toc=time.time()
    print(f"Time elapsed to train: {toc - tic}")
    print("The model learned the following values for weights and bias:")
    pprint(model_1.state_dict())
    print("\nAnd the original values for weights and bias are:")
    print(f"weights: {weights}, bias: {bias}")







    ## 4.Prediction and display
    model_1.eval() # Evaluation mode
    with t.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    plt.figure(figsize=(10,7))
    plt.scatter(X_train.cpu(), y_train.cpu(), c='b', s=4, label='Training Data')
    plt.scatter(X_test.cpu(), y_test.cpu(), c='g', s=4, label='Test Data')
    plt.scatter(X_test.cpu(), test_pred.cpu(), c='r', s=4, label='Prediction')
    plt.legend(prop={"size": 14})
    plt.show()

if __name__ == "__main__":
    main()


