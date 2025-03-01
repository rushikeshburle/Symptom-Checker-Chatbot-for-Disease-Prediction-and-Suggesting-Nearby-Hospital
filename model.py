import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the final time step's output
        return out





#  **# Model.py Flow**  

# The RNNModel class defines a neural network using Long Short-Term Memory (LSTM), which is a type of Recurrent Neural Network (RNN). This model processes sequential data, such as time-series data or text, and outputs predictions based on the patterns learned from the input.

# 1. Class Initialization (__init__):
   
# Parameters:
# input_size: Number of features in each time step of the input data.
# hidden_size: Number of units in the LSTM's hidden state (controls the capacity of the model).
# output_size: The size of the final output (e.g., number of classes for classification).
# num_layers: Number of LSTM layers stacked together.

# Components:
# self.lstm:
# This is an LSTM layer that processes sequential input.
# It takes input with dimensions [batch_size, sequence_length, input_size] and outputs a sequence of hidden states for each time step.
# batch_first=True ensures the input and output tensors have the batch size as the first dimension.
# self.fc:
# A fully connected (linear) layer that maps the LSTM's hidden state output to the desired output size.


# 2. Forward Pass (forward):
# The forward method defines how the model processes the input to produce an output:

# The LSTM processes the input x and returns:
# out: A tensor containing the hidden states for all time steps with shape [batch_size, sequence_length, hidden_size].
# hn: The final hidden state of the LSTM for all layers (not used here).
# cn: The final cell state of the LSTM for all layers (not used here).

# Select the Last Time Step's Output:
# out[:, -1, :]: Selects the hidden state of the last time step in the sequence for each batch. This represents the sequence's overall context.
# self.fc: Passes this selected output through the fully connected layer to produce the final output with shape [batch_size, output_size].

# The RNNModel performs the following:
# Processes sequential data through an LSTM layer to learn patterns in the sequence.
# Uses the final time step’s output to represent the sequence's information.
# Maps this representation to the desired output size using a fully connected layer.

