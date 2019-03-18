import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##Throughout this script .values will be used to return an array, this is because tensorflow deals with data in the form of arrays


# Load training data set from CSV file
training_data_df = pd.read_csv("C:\\Users\\reil\\Desktop\\Linkedin-LEarning\\Building & Deploying Deep Nueral Nets\\Ex_Files_TensorFlow\\Exercise Files\\03\\sales_data_training.csv", dtype = float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop('total_earnings', axis = 1).values
Y_training = training_data_df[['total_earnings']].values

# Load testing data set from CSV file
test_data_df =pd.read_csv("C:\\Users\\reil\\Desktop\\Linkedin-LEarning\\Building & Deploying Deep Nueral Nets\\Ex_Files_TensorFlow\\Exercise Files\\03\\sales_data_test.csv", dtype = float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings', axis = 1).values
Y_testing = training_data_df[['total_earnings']].values

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. This puts each attribute on the same scale, these lines only set the feature range which data should be scaled to
X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

# This is where we actually scale our data
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.fit_transform(X_testing)
Y_scaled_testing = Y_scaler.fit_transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))

# Define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

#-------------------------------------------Section One: Define the layers of the neural network itself----------------------------###

# Input Layer, we use a variable scope to define our input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape = (None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    # the shape of the weight has to include each of the inputs and each node
    weights = tf.get_variable(name = 'weights1', shape = [number_of_inputs, layer_1_nodes], initializer = tf.contrib.layers.xavier_initializer())
    # we want the biases to default to zero, so use tf zero initializer, also using tf.get_variable not tf.placeholder because
    # we want tensorflow to remember the value of that node.  
    biases = tf.get_variable(name = 'biases1', shape=[layer_1_nodes], initializer = tf.zeros_initializer)
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    # in this section we pretty much just copy and paste the code from layer 1 while changing a few key components, 
    # the shape of the weights is different, we neet the output from layer 1, rather than the original inputs
    weights = tf.get_variable(name = 'weights2', shape = [layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = 'biases2', shape=[layer_2_nodes], initializer = tf.zeros_initializer)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name = 'weights3', shape = [layer_2_nodes, layer_3_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = 'biases3', shape=[layer_3_nodes], initializer = tf.zeros_initializer)
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name = 'weights4', shape = [layer_3_nodes, number_of_outputs], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name = 'biases4', shape=[number_of_outputs], initializer = tf.zeros_initializer)
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

#---------------Section Two: Define the cost function of the neural network that will measure prediction accuracy during training---------###

#here we define the cost function, this tells us how wrong we are when creating a prediction 
with tf.variable_scope('cost'):
    # Y is a variable representing a respective node that we'll feed in during training of the model.  This will be kept
    # using a placeholder node because we're feeding in a new value each time
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))


#-------------Section Three: Define the optimizer function that will be run to optimize the neural network--------------------------------###

with tf.variable_scope('train'):
    # Adam optimizer common and powerful
    # We need a learning rate (already defined above)
    # Finally, we need a variable that we want to minimize (our cost function, which is already defined above)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.variable_scope('logging'):
    #We're using scalar here because a scalar is just a single number, and we're only logging a single number here
    tf.summary.scalar('current cost', cost)
    #This command allows us to log all the summary nodes in our graph without us having to list hem
    summary = tf.summary.merge_all()

saver = tf.train.Saver()

#---------------------------------Creating the Training loop for our data-----------------------------------------------------------------###


# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network,
    # This is the first step in initializing our training loop, global_variables_initializer intilializes all of our varibales
    session.run(tf.global_variables_initializer())

    #Creating the log files that will write our graphs
    training_writer = tf.summary.FileWriter("./logs/training", session.graph)
    testing_writer = tf.summary.FileWriter("./logs/testing", session.graph)

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set, we'll do this the funn 100 times.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        # here we're running over one pass training our model
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # we print out the accuracy every 5 steps so we get a better idea of if we're increaing our accuracy during training. for some reason, there 

        #if epoch % 5 == 0:
            #training_cost, training_summary = session.run([cost,summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            #testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})

            #print(epoch, training_cost, testing_cost)

        # Print the current training status to the screen
        print("Training pass: {}".format(epoch))

    # Training is now complete!
    print("Training is complete!")

    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    save_path = saver.save(session, "C:\\Users\\reil\\Desktop\\Linkedin-LEarning\\Building & Deploying Deep Nueral Nets\\Ex_Files_TensorFlow\\final_trained_model.ckpt")
