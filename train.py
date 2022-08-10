from cProfile import label
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt


#Config needed because otherwise the GPU has no memory for training
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#Needed to set precision to float64
tf.keras.backend.set_floatx('float64')

def usage():
  print("USAGE: python3 train.py <train.csv> <test.csv> OPTIONAL: <nnet.npz>")

def main(train_csv, test_csv, nnet = None):
  train = pd.read_csv(train_csv)
  test = pd.read_csv(test_csv)
  x_train = train[["roll", "u_x","u_y", "yaw_der", "steering", "throttle"]].to_numpy()
  y_train = train[[ "u_x_der", "u_y_der", "yaw_der_der","roll_der"]].to_numpy()
  x_test =  test[["roll", "u_x","u_y", "yaw_der", "steering", "throttle"]].to_numpy()
  y_test =  test[["u_x_der", "u_y_der", "yaw_der_der","roll_der"]].to_numpy()

  logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


  # THESE ARE THE DESIGN PARAMETERS OF THE NEURAL NETWORK
  # input_shape = 6        # number of independent input parameters
  # input_timesteps = 1     # number of previous vehicle states to inlcude into neural network input
  # output_shape = 4        # number of vehicle states (output of NN); DO NOT CHANGE
  learning_rate = 0.001  # should not exceed 0.0005
  initializer = "he"
  l2regularization = 0.001
  l1regularization = 0.0

  #CREATE NEURAL NETWORK


  if initializer == "he":
      kernel_init = tf.keras.initializers.he_uniform(seed=True)

  elif initializer == "glorot":
      kernel_init = tf.keras.initializers.GlorotUniform(seed=True)

  reg_dense = tf.keras.regularizers.l1_l2(l1regularization,
                                        l2regularization)

  #input_shape = input_timesteps * output_shape

  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape=(6,)))
  layer_a = tf.keras.layers.Dense(32,
                          use_bias=True,
                          bias_initializer='zeros',
                          activation='tanh',
                          # kernel_initializer=kernel_init,
                          kernel_regularizer=reg_dense)
  layer_b = tf.keras.layers.Dense(32,
                          bias_initializer='zeros',
                          use_bias=True,
                          activation='tanh',
                          # kernel_initializer=kernel_init,
                          kernel_regularizer=reg_dense)
  layer_c = tf.keras.layers.Dense(4)

  model.add(layer_a)
  model.add(layer_b)
  model.add(layer_c)

  #After adding the layers to the model the weight matrix has values now
  if nnet is not None:
    old_nnet = np.load("autorally.npz")
    layer_a.set_weights(list([np.transpose(old_nnet['dynamics_W1']),np.transpose(old_nnet['dynamics_b1'])]))
    layer_b.set_weights(list([np.transpose(old_nnet['dynamics_W2']),np.transpose(old_nnet['dynamics_b2'])]))
    layer_c.set_weights(list([np.transpose(old_nnet['dynamics_W3']),np.transpose(old_nnet['dynamics_b3'])]))

  optimizer = 'Nesterov-ADAM'
  clipnorm = 1.0
  loss_function = 'mean_squared_error'


  if optimizer == 'ADAM':
      optimizer = tf.keras.optimizers.Adam(lr=learning_rate,
                                  clipnorm=clipnorm)  # , beta_1=0.9, beta_2=0.999)

  if optimizer == 'SGD':
      optimizer = tf.keras.optimizers.SGD(lr=learning_rate,
                                  nesterov=True,
                                  clipnorm=clipnorm)

  if optimizer == 'RMSPROP':
      optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate,
                                      momentum=0.9,
                                      clipnorm=clipnorm)

  if optimizer == 'ADADELTA':
      optimizer = tf.keras.optimizers.Adadelta(lr=learning_rate)  # , rho=0.95)

  if optimizer == 'Nesterov-ADAM':
      optimizer = tf.keras.optimizers.Nadam(lr=learning_rate,
                                    clipnorm=clipnorm)  # , beta_1=0.91, beta_2=0.997)


  model.compile(optimizer=optimizer,
                        loss=loss_function,
                        metrics=['mae', 'mse'])

  model.summary()

  #Training the model
  epochs = 10  
  error_epochs = np.empty([epochs, 4], dtype='float64')
  for epoch in range(epochs):

    print("Epoch %d/%d" % (epoch+1, epochs))
    #Run one epoch of the training
    model.fit(x_train, y_train, epochs=1,
                callbacks=[tensorboard_callback])
    # if(epoch%10 == 0):
    #   model.save('./tmp/model')

    #Calculate losses/progress per variable
    predic = model.predict(x_train)
    
    diff_epoch = abs(predic - y_train)
    error_epochs[epoch,0] = np.average(diff_epoch[:,0])
    error_epochs[epoch,1] = np.average(diff_epoch[:,1])
    error_epochs[epoch,2] = np.average(diff_epoch[:,2])
    error_epochs[epoch,3] = np.average(diff_epoch[:,3])
    
  #Plot the data
  x_axis = list(range(epochs))
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.set_title('Average of errors per epoch')
  ax.plot(x_axis, error_epochs[:,0], color='tab:blue',    label = "roll_der"    )
  ax.plot(x_axis, error_epochs[:,1], color='tab:orange',  label = "u_x_der"     )
  ax.plot(x_axis, error_epochs[:,2], color='tab:red',     label = "u_y_der"     )
  ax.plot(x_axis, error_epochs[:,3], color='tab:gray',   label = "yaw_der_der"  )
  ax.legend()
  plt.savefig('Errors.png')

  #Evaluating its overall performance
  model.evaluate(x_test, y_test)

  #Save the weights to a npz file
  weight_name = "dynamics_W"
  bias_name = "dynamics_b"
  it = 1

  files = {}
  # iterate over each set of weights and biases
  for layer in model.layers:
    files[bias_name + str(it)] = layer.bias
    files[weight_name + str(it)] = tf.Variable(np.transpose(layer.weights[0]))
    it +=1
      
  np.savez_compressed('custom.npz', dynamics_b1=files['dynamics_b1'],dynamics_b2=files['dynamics_b2'],dynamics_b3=files['dynamics_b3'], \
  dynamics_W3=files['dynamics_W3'],dynamics_W2=files['dynamics_W2'],dynamics_W1=files['dynamics_W1'])

if __name__ == '__main__':
  if len(sys.argv) == 3:
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    main(train_csv, test_csv)
  elif len(sys.argv) == 4:
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    nnet = sys.argv[3]
    main(train_csv, test_csv, nnet)
  else:
    usage()
