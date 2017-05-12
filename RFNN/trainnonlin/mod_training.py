import tensorflow as tf

def do_training():
    # Create placeholders
    train_data_node = tf.placeholder(tf.complex64,
                                     shape=(25, 10),
                                     name="train_data_node")

    train_labels_node = tf.placeholder(tf.complex64,
                                       shape=(25, 10),
                                       name="train_labels_node")
     
    weights = {

        # Fully connected weights, layer 1
        'fc_w1': tf.Variable(tf.complex( tf.random_normal([10, 10],
                                                        stddev=0.01
                                                        , dtype =  tf.float32),
                                                    tf.random_normal([10, 10],
                                                        stddev=0.01
                                                        , dtype =  tf.float32))),
        'fc_b1': tf.Variable(tf.complex(tf.random_normal([10]), tf.random_normal([10]))),
        }
    
    logits = model(train_data_node, weights)
    loss = tf.real(tf.norm(logits - train_labels_node))

    
    train_op = tf.train.AdamOptimizer(learning_rate=1.0).minimize(loss)
            
            
def model(data, weights):
    
    l1 = tf.matmul(data, weights['fc_w1'])                                                        # FC
    l1 = l1 + weights['fc_b1']
    return l1



