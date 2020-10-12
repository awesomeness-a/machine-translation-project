from preprocessing import num_encoder_tokens, num_decoder_tokens, decoder_target_data, encoder_input_data, decoder_input_data, decoder_target_data

from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# UNCOMMENT THE TWO LINES BELOW IF YOU ARE GETTING ERRORS ON A MAC
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


latent_dim = 305

# Choose a batch size
# and a number of epochs:
batch_size = 50
epochs = 100

# Encoder training setup
# Create an input layer which defines a matrix we feed to the model
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Create an LSTM layer, with some output dimensionality
encoder_lstm = LSTM(latent_dim, return_state=True)
# Link LSTM layer with the input layer
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
# Save the states in a list
encoder_states = [state_hidden, state_cell]

# Decoder training setup:
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# Build a final Dense activation layer, using teh Softmax function
# which gives the probability distribution—where all probabilities
# sum to one—for each token
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# Run the decoder_outputs through the Dense layer
decoder_outputs = decoder_dense(decoder_outputs)

# Building the training model:
# feed it the encoder and decoder inputs
# and the decoder output
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Model summary:\n")
training_model.summary()
print("\n\n")

# Compile the model:
# optimizer helps minimize the error rate
# a loss function determines the error rate
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training the model:\n")
# Train the model:
# feed in the encoder and decoder input data and decoder target data
# validation_split is what percentage of the data should be set aside for validating
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

training_model.save('training_model.h5')