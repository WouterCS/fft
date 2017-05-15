import tensorflow as tf
import numpy as np

# first train a linear model on random vectors of length 5 and store the trained parameters.
# Then load those parameters and try to apply them to a new vector.
def run():
    train_model()
    apply_model()
    
def train_model():    
    # create random training data: 100 vectors of length 5 for both input and output.
    train_data  = np.random.random((100,5))
    train_labels = np.random.random((100,5))

    train_data_node = tf.placeholder(tf.float32, shape=(5), name="train_data_node")
    train_labels_node = tf.placeholder(tf.float32, shape=(5), name="train_labels_node")
        
    weights = defineWeights()
    
    prediction = model(train_data_node, weights)
    loss = tf.norm(prediction - train_labels_node)
    train_op = tf.train.AdagradOptimizer(learning_rate=1).minimize(loss)
    
    saver = tf.train.Saver(weights)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # train for 50 epochs on all 100 training examples, with a batchsize of 1.
    for _ in range(50):
        for i in range(100):
            batch_data = train_data[i,:]
            batch_labels = train_labels[i,:]

            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            sess.run([train_op, loss, weights], feed_dict=feed_dict)
        saver.save(sess, '/results/weights')
    
def apply_model():
    sess = tf.Session()

    weights = defineWeights()
    
    new_saver = tf.train.import_meta_graph('/results/weights.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('/results'))
    
    print(model(np.random.random(5).astype(np.float32), weights).eval(session=sess))
            
            
def model(data, weights):
    # multiply the matrix weights['a'] with the vector data
    l1 = tf.matmul(tf.expand_dims(data,0), weights['a'])
    l1 = l1 + weights['b']
    return l1

def defineWeights():
    weights = {
        'a': tf.Variable(tf.random_normal([5, 5],
                                                        stddev=0.01, 
                                                        dtype =  tf.float32),
                                                        name = 'a'),
        'b': tf.Variable(tf.random_normal([5]), name = 'b'),
        }
    return weights