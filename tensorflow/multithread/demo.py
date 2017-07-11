import time
import tensorflow as tf

# We simulate some raw input data 
# (think about it as fetching some data from the file system)
# let's say: batches of 128 samples, each containing 1024 data points
x_input_data = tf.random_normal([128, 1024], mean=0, stddev=1)

# We build our small model: a basic two layers neural net with ReLU
with tf.variable_scope("queue"):
    q = tf.FIFOQueue(capacity=5, dtypes=tf.float32) # enqueue 5 batches
    # We use the "enqueue" operation so 1 element of the queue is the full batch
    enqueue_op = q.enqueue(x_input_data)
    numberOfThreads = 3
    qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
    tf.train.add_queue_runner(qr)
    input = q.dequeue() # It replaces our input placeholder
    # We can also compute y_true right into the graph now
    y_true = tf.cast(tf.reduce_sum(input) > 0, tf.int32)

with tf.variable_scope('FullyConnected'):
    w1 = tf.get_variable('w1', shape=[1024, 1024], initializer=tf.random_normal_initializer(stddev=1e-1))
    b1 = tf.get_variable('b1', shape=[1024], initializer=tf.constant_initializer(0.1))
    z1 = tf.matmul(input, w1) + b1
    y1 = tf.nn.relu(z1)

    w2 = tf.get_variable('w2', shape=[1024, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
    b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(y1, w2) + b2

with tf.variable_scope('Loss'):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(z, tf.cast(y_true, tf.float32))
    loss_op = tf.reduce_mean(losses)

with tf.variable_scope('Accuracy'):
    y_pred = tf.cast(z > 0, tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

# We add the training op ...
adam = tf.train.AdamOptimizer(1e-2)
train_op = adam.minimize(loss_op, name="train_op")

startTime = time.time()
with tf.Session() as sess:
    # ... init our variables, ...
    sess.run(tf.initialize_all_variables())

    # ... add the coordinator, ...
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # ... check the accuracy before training (without feed_dict!), ...
    sess.run(accuracy)

    # ... train ...
    for i in range(5000):
        #  ... without sampling from Python and without a feed_dict !
        _, loss = sess.run([train_op, loss_op])

        # We regularly check the loss
        if i % 500 == 0:
            print('iter:%d - loss:%f' % (i, loss))

    # Finally, we check our final accuracy
    sess.run(accuracy)

    coord.request_stop()
    coord.join(threads)

print("Time taken: %f" % (time.time() - startTime))
