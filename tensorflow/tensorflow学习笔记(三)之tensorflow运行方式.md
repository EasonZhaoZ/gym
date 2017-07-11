# tensorflow学习笔记(三)之tensorflow运行方式

## 0.API

[https://www.tensorflow.org/versions/r0.12/api_docs/index.html](https://www.tensorflow.org/versions/r0.12/api_docs/index.html)

## 1.执行命令

```
python fully_connected_feed.py --train_dir ~/github/gym/tensorflow/mnist/
```

## 2.基本Class

### 2.1 tf.Variable 

```
def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + conv2_biases)
```

> 这里的变量类似于其他语言中的局部变量，当my_image_filter每次被调用的时候都会重新创建variables

如果希望使用共享变量的时候，可以利用TensorFlow提供的变量作用域机制，使用variable\_scope和get\_variable配合使用。

[wiki](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html)

### 2.2 tf.train.Saver

写checkpoint

```
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print "Model saved in file: ", save_path
```

恢复checkpoint

```
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print "Model restored."
  # Do some work with the model
  ...
```

## 3. 训练数据

* feed_dict方法

> 最好的做法应该是使用placeholder op节点。设计placeholder节点的唯一的意图就是为了提供数据供给(feeding)的方法。placeholder节点被声明的时候是未初始化的， 也不包含数据， 如果没有为它供给数据， 则TensorFlow运算的时候会产生错误， 所以千万不要忘了为placeholder提供数据。

```
with tf.Session():
  input = tf.placeholder(tf.float32)
  classifier = ...
  print classifier.eval(feed_dict={input: my_python_preprocessing_fn()})
```
* 从文件读取 

> 一般是从文件读取和feed方法相结合。 

```
http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/reading_data.html
```


## 4.TensorBoard

写summary

```
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
total_step = 0
while training:
  total_step += 1
  session.run(training_op)
  if total_step % 100 == 0:
    summary_str = session.run(merged_summary_op)
    summary_writer.add_summary(summary_str, total_step)
```

查看方法，pip安装tensorboard之后

```
tensorboard --logdir=/path/to/log-directory
```

## 5.多线程

所幸TensorFlow提供了两个类来帮助多线程的实现：tf.Coordinator和 tf.QueueRunner。从设计上这两个类必须被一起使用。Coordinator类可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常。QueueRunner类用来协调多个工作线程同时将多个张量推入同一个队列中。

```
Coordinator
Coordinator类用来帮助多个线程协同工作，多个线程同步终止。 其主要方法有：

should_stop():如果线程应该停止则返回True。
request_stop(<exception>): 请求该线程停止。
join(<list of threads>):等待被指定的线程终止。
```

在Python的训练程序中，创建一个QueueRunner来运行几个线程， 这几个线程处理样本，并且将样本推入队列。创建一个Coordinator，让queue runner使用Coordinator来启动这些线程，创建一个训练的循环， 并且使用Coordinator来控制QueueRunner的线程们的终止。

```
example = ...ops to create one example...
# Create a queue, and an op that enqueues examples one at a time in the queue.
queue = tf.RandomShuffleQueue(...)
enqueue_op = queue.enqueue(example)
# Create a training graph that starts by dequeuing a batch of examples.
inputs = queue.dequeue_many(batch_size)
train_op = ...use 'inputs' to build the training part of the graph...

# Create a queue runner that will run 4 threads in parallel to enqueue
# examples.
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

# Launch the graph.
sess = tf.Session()
# Create a coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
# Run the training loop, controlling termination with the coordinator.
for step in xrange(1000000):
    if coord.should_stop():
        break
    sess.run(train_op)
# When done, ask the threads to stop.
coord.request_stop()
# And wait for them to actually do it.
coord.join(threads)
```

demo: [https://github.com/EasonZhaoZ/gym/blob/master/tensorflow/multithread/demo.py](https://github.com/EasonZhaoZ/gym/blob/master/tensorflow/multithread/demo.py)