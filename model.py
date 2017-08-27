import tensorflow as tf
import mydata
import random

#print(mydata.r[0])

#data
data_size = len(mydata.r[0])
print ("Data size = %d" % data_size)
test_size = 178
print ("Test size = %d" % test_size)
train_size =  data_size - test_size

#embedding settings
vocab_size = len(mydata.dict)
embedding_size = 100
max_sentence_length = 10

#Network settings
n_shrt_term = 100
n_medi_term = 100
n_long_term = 100

n_hidden_1 = 512
n_hidden_2 = 512
n_classes = 2


# Parameters
learning_rate = 0.005
training_epochs = 200
batch_size = 100
display_step = 1

#Network inputs
x_shrt_term = tf.placeholder(tf.int32, [None,  7, max_sentence_length])
x_medi_term = tf.placeholder(tf.int32, [None,  7, max_sentence_length])
x_long_term = tf.placeholder(tf.int32, [None, 30, max_sentence_length])

y = tf.placeholder(tf.float32, [None, n_classes])
#Variables
embedding_matrix = tf.Variable(tf.random_normal([vocab_size, embedding_size]),name="W")

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([embedding_size * max_sentence_length, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#model
def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	
    return out_layer
	


lookup_shrt = tf.nn.embedding_lookup(embedding_matrix, x_shrt_term)
lookup_medi = tf.nn.embedding_lookup(embedding_matrix, x_medi_term)
lookup_long = tf.nn.embedding_lookup(embedding_matrix, x_long_term)

avg = tf.reduce_mean(lookup_shrt, 1) 
ff_input = tf.reshape(avg, [-1 , embedding_size * max_sentence_length])

# Construct model
pred = multilayer_perceptron(ff_input, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # sample_data = [ [[1,2,3,4,5],[2,7,10,15,17]], [[1,2,3,4,5],[2,7,10,15,17]]]
    # res = sess.run(lookup_shrt, feed_dict={x_shrt_term: sample_data})
    # print (res)
    # print ("---------")
    # res = sess.run(avg, feed_dict={x_shrt_term: sample_data})
    # print (res)
    # print ("---------")
    # res = sess.run(ff_input, feed_dict={x_shrt_term: sample_data})
    # print (res)
    
    
    train_x = mydata.r[0][:train_size]
    train_y = mydata.r[1][:train_size]
    
    test_x = mydata.r[0][-test_size:]
    test_y = mydata.r[1][-test_size:]
    
    
    for epoch in range(training_epochs):
        total_batch = train_size//batch_size
        avg_cost = 0.
        seq = [ k for k in range (total_batch)]
        random.shuffle(seq)
        for i in range(total_batch):
            batch_x = train_x[seq[i]:seq[i]+batch_size]
            batch_y = train_y[seq[i]:seq[i]+batch_size]
            #print (batch_y)
            _, c = sess.run([optimizer, cost], feed_dict={x_shrt_term: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print (avg_cost)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x_shrt_term: test_x, y: test_y}))
            #break
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
     