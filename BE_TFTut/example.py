import tensorflow as tf


# Define constants, set them to the values
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

# Create a variable for the loss
loss = tf.Variable((y - y_hat)**2, name='loss')

# The ession
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print(session.run(loss))