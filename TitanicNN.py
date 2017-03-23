import numpy as np
import pandas as pd        # For loading and processing the dataset
import tensorflow as tf    # Of course, we need TensorFlow.
from sklearn.model_selection import train_test_split



# Parameters
learning_rate = 0.005
training_epochs = 15
batch_size = 100
display_step = 1
dropout = 1

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 9
n_classes = 2

def load_data(train_data):
    df_train = pd.read_csv(train_data)

    # We can't do anything with the Name, Ticket number, and Cabin, so we drop them.
    df_train = df_train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

    # To make 'Sex' numeric, we replace 'female' by 0 and 'male' by 1
    df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1}).astype(int)

    # We replace 'Embarked' by three dummy variables 'Embarked_S', 'Embarked_C', and 'Embarked Q',
    # which are 1 if the person embarked there, and 0 otherwise.
    df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix='Embarked')], axis=1)
    df_train = df_train.drop('Embarked', axis=1)

    # We normalize the age and the fare by subtracting their mean and dividing by the standard deviation
    global age_mean
    age_mean = df_train['Age'].mean()
    global age_std
    age_std = df_train['Age'].std()
    df_train['Age'] = (df_train['Age'] - age_mean) / age_std

    global fare_mean
    fare_mean = df_train['Fare'].mean()
    global fare_std
    fare_std = df_train['Fare'].std()
    df_train['Fare'] = (df_train['Fare'] - fare_mean) / fare_std

    # In many cases, the 'Age' is missing - which can cause problems. Let's look how bad it is:
    print("Number of missing 'Age' values: {:d}".format(df_train['Age'].isnull().sum()))

    # A simple method to handle these missing values is to replace them by the mean age.
    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

    # Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and test set
    X_train = df_train.drop('Survived', axis=1).as_matrix()
    y_train = df_train['Survived'].as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    # We'll build a classifier with two classes: "survived" and "didn't survive",
    # so we create the according labels
    # This is taken from https://www.kaggle.com/klepacz/titanic/tensor-flow
    return X_train, X_test, y_train, y_test

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, dropout)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, dropout)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('train.csv')
    labels_train = (np.arange(2) == y_train[:, None]).astype(np.float32)
    labels_test = (np.arange(2) == y_test[:, None]).astype(np.float32)
    inputs = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='inputs')
    label = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

    # Cost function and optimizer
    lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')

    y_output = multilayer_perceptron(inputs, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_output, labels=label))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    # Prediction
    pred = tf.nn.softmax(y_output)
    pred_label = tf.argmax(pred, 1)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create operation which will initialize all variables
    init = tf.global_variables_initializer()

    # Configure GPU not to use all memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start a new tensorflow session and initialize variables
    sess = tf.InteractiveSession(config=config)
    sess.run(init)

    # This is the main training loop: we train for 50 epochs with a learning rate of 0.05 and another
    # 50 epochs with a smaller learning rate of 0.01
    for learning_rate in [0.01, 0.007, 0.005]:
        for epoch in range(50):
            avg_cost = 0.0

            # For each epoch, we go through all the samples we have.
            for i in range(X_train.shape[0]):
                # Finally, this is where the magic happens: run our optimizer, feed the current example into X and the current target into Y
                _, c = sess.run([optimizer, cost], feed_dict={lr: learning_rate,
                                                              inputs: X_train[i, None],
                                                              label: labels_train[i, None]})
                avg_cost += c
            avg_cost /= X_train.shape[0]

            # Print the cost in this epcho to the console.
            if epoch % 10 == 0:
                print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))

    acc_train = accuracy.eval(feed_dict={inputs: X_train, label: labels_train})
    print("Train accuracy: {:3.2f}%".format(acc_train * 100.0))

    acc_test = accuracy.eval(feed_dict={inputs: X_test, label: labels_test})
    print("Test accuracy:  {:3.2f}%".format(acc_test * 100.0))

    df_test = pd.read_csv('test.csv')

    # Do all pre-processing steps as above
    df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)
    df_test = df_test.drop('Embarked', axis=1)
    df_test['Age'] = (df_test['Age'] - age_mean) / age_std
    df_test['Fare'] = (df_test['Fare'] - fare_mean) / fare_std
    df_test.head()
    X_test = df_test.drop('PassengerId', axis=1).as_matrix()

    # Predict
    for i in range(X_test.shape[0]):
        df_test.loc[i, 'Survived'] = sess.run(pred_label, feed_dict={inputs: X_test[i, None]}).squeeze()

    # Important: close the TensorFlow session, now that we're finished.
    sess.close()

    output = pd.DataFrame()
    output['PassengerId'] = df_test['PassengerId']
    output['Survived'] = df_test['Survived'].astype(int)
    output.to_csv('./prediction.csv', index=False)

