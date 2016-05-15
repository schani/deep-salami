from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

def normalize_dataset(dataset):
    for i, d in enumerate(dataset):
        dataset[i] = d - np.mean(d)
    return dataset

class Data:
    def reformat_labels(self, labels):
        # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
        return (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)

    def reformat_1d(self, dataset):
        return dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)

    # Reformat into a TensorFlow-friendly shape:
    # - convolutions need the image data formatted as a cube (width by height by #channels)
    def reformat_2d(self, dataset):
        return dataset.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)

    def __init__(self):
        self.image_size = 28
        self.num_labels = 10
        self.num_channels = 1 # grayscale

        pickle_file = 'notMNIST-schani.pickle'

        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)

        self.train_dataset = normalize_dataset(save['train_dataset'])
        self.train_labels = self.reformat_labels(save['train_labels'])
        self.valid_dataset = normalize_dataset(save['valid_dataset'])
        self.valid_labels = self.reformat_labels(save['valid_labels'])
        self.test_dataset = normalize_dataset(save['test_dataset'])
        self.test_labels = self.reformat_labels(save['test_labels'])
        print('Training set', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set', self.test_dataset.shape, self.test_labels.shape)

        self.fc_train_dataset = None
        self.fc_valid_dataset = None
        self.fc_test_dataset = None

        self.conv_train_dataset = None
        self.conv_valid_dataset = None
        self.conv_test_dataset = None

    def datasets_1d(self):
        if self.fc_train_dataset is None:
            self.fc_train_dataset = self.reformat_1d(self.train_dataset)
            self.fc_valid_dataset = self.reformat_1d(self.valid_dataset)
            self.fc_test_dataset = self.reformat_1d(self.test_dataset)
        return (self.fc_train_dataset, self.fc_valid_dataset, self.fc_test_dataset)

    def datasets_2d(self):
        if self.conv_train_dataset is None:
            self.conv_train_dataset = self.reformat_2d(self.train_dataset)
            self.conv_valid_dataset = self.reformat_2d(self.valid_dataset)
            self.conv_test_dataset = self.reformat_2d(self.test_dataset)
        return (self.conv_train_dataset, self.conv_valid_dataset, self.conv_test_dataset)

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def wrong_predictions(predictions, labels):
    return np.nonzero(np.argmax(predictions, 1) != np.argmax(labels, 1))[0]

class Network:
    def __init__(self, batch_size, is_2d, data):
        self.batch_size = batch_size
        self.data = data
        self.is_2d = is_2d

        if is_2d:
            train_dataset, _, test_dataset = data.datasets_2d()
        else:
            train_dataset, _, test_dataset = data.datasets_1d()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.graph = tf.Graph()

    def session(self):
        return tf.Session(graph = self.graph)

    def train(self, session, num_steps):
        start_time = time.time()
        session.run(self.init_op)
        tf.initialize_all_variables().run()
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * self.batch_size) % (self.data.train_labels.shape[0] - self.batch_size)
            if offset + self.batch_size >= self.data.train_labels.shape[0]:
                offset = 0
            # Generate a minibatch.
            #(batch_data, batch_labels) = random_subset(self.train_dataset, self.data.train_labels, self.batch_size)
            batch_data = self.train_dataset[offset:(offset + self.batch_size), :]
            batch_labels = self.data.train_labels[offset:(offset + self.batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
            session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
            #if (step % 500 == 0):
            #    print("Minibatch loss at step %d: %f" % (step, l))
            #    if math.isnan(l):
            #        break
            #    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            #    validation_accuracy = accuracy(self.valid_prediction.eval(), valid_labels)
            #    print("Validation accuracy: %.1f%%" % validation_accuracy)
        end_time = time.time()
        total_time = end_time - start_time
        return total_time

    def test(self, session, save_threshold, save_filename):
        test_predictions = self.test_prediction.eval()
        test_accuracy = accuracy(test_predictions, self.data.test_labels)
        wrong_test_predictions = wrong_predictions(test_predictions, self.data.test_labels)
        if test_accuracy > save_threshold:
            print("accuracy of %f - saving to %s" % (test_accuracy, save_filename))
            self.saver.save(session, save_filename)
        return (test_accuracy, test_predictions, wrong_test_predictions)

    def fake_letter(self, session, wanted_class, minimum_prediction, model_filename):
        start_time = time.time()
        self.saver.restore(session, model_filename)

        fake_data = self.test_dataset[0:self.batch_size, :]
        fake_labels = self.data.test_labels[0:self.batch_size, :]
        #fake_data = np.random.rand(self.batch_size, self.data.image_size*self.data.image_size) - 0.5
        #fake_labels = (np.arange(num_labels) == np.full(self.batch_size, wanted_class)[:,None]).astype(np.float32)

        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {self.tf_train_dataset : fake_data, self.tf_train_labels : fake_labels}
        [predictions] = session.run([self.train_prediction], feed_dict=feed_dict)
        class_predictions = predictions[:, wanted_class]
        choice = np.argmin(class_predictions)
        #plt.imshow(np.reshape(fake_data[choice], (self.data.image_size, self.data.image_size)), cmap="gray")
        print("Prediction for worst: %f" % class_predictions[choice])
        for step in range(10000):
            #print("fake shape is ", fake_data.shape)
            # re-normalize choice
            choice_image = fake_data[choice]
            choice_image = choice_image - np.mean(choice_image)
            image_min = np.min(choice_image)
            image_max = np.max(choice_image)

            #print("image shape is ", choice_image.shape)

            if self.is_2d:
                fake_data = np.tile(choice_image, (self.batch_size, 1, 1, 1))
            else:
                fake_data = np.tile(choice_image, (self.batch_size, 1))

            #print("after tiling fake shape is ", fake_data.shape)

            for i in range(self.batch_size):
                new_pixel = np.random.uniform(image_min, image_max)
                if self.is_2d:
                    fake_data[i, np.random.randint(self.data.image_size), np.random.randint(self.data.image_size)] = new_pixel
                else:
                    fake_data[i, np.random.randint(self.data.image_size*self.data.image_size)] = new_pixel
            feed_dict = {self.tf_train_dataset : fake_data, self.tf_train_labels : fake_labels}
            [predictions] = session.run([self.train_prediction], feed_dict=feed_dict)
            class_predictions = predictions[:, wanted_class]
            choice = np.argmax(class_predictions)
            if class_predictions[choice] > minimum_prediction:
                break
        print("Prediction for best after %d steps: %f" % (step, class_predictions[choice]))
        image = np.reshape(fake_data[choice], (self.data.image_size, self.data.image_size))
        end_time = time.time()
        total_time = end_time - start_time
        print("Time: %.1fs" % total_time)
        return image

class FCNetwork(Network):
    def __init__(self, batch_size, data, relu_sizes, with_dropout, initial_weights, initial_lr, lr_decay_steps):
        super().__init__(batch_size, False, data)
        _, valid_dataset, test_dataset = data.datasets_1d()

        with self.graph.as_default():
            input_size = self.data.image_size * self.data.image_size

            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, data.num_labels))
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            weights_and_biases = []
            activation_size = input_size
            for layer_size in relu_sizes + [data.num_labels]:
                weights = tf.Variable(tf.truncated_normal([activation_size, layer_size], stddev=initial_weights))
                biases = tf.Variable(tf.constant(0.1, shape=[layer_size]))
                activation_size = layer_size
                weights_and_biases.append((weights, biases))

            def construct_logits(dataset, with_dropout):
                activation = dataset

                #if dropout_prob:
                #    activation = tf.nn.dropout(activation, 1 - dropout_prob)

                i = 0
                for (weights, biases) in weights_and_biases[0:-1]:
                    if with_dropout:
                        activation = tf.nn.dropout(activation, with_dropout)
                    #tf.verify_tensor_all_finite(activation, "activation %d not finite" % i)
                    i += 1
                    relu_input = tf.matmul(activation, weights) + biases
                    relu_output = tf.nn.sigmoid(relu_input)
                    #if dropout_prob:
                    #    relu_output = tf.nn.dropout(relu_output, 1 - dropout_prob)
                    activation = relu_output

                #tf.verify_tensor_all_finite(activation, "activation %d not finite" % i)
                (weights, biases) = weights_and_biases[-1]
                return tf.matmul(activation, weights) + biases

            logits = construct_logits(self.tf_train_dataset, with_dropout)
            #tf.verify_tensor_all_finite(logits, "logits not finite")

            #relu_l2_loss = tf.nn.l2_loss(relu_weights) / (self.data.image_size*self.data.image_size*relu_size) * l2_loss_weight
            #output_loss = tf.nn.l2_loss(weights) / (relu_size*data.num_labels) * l2_loss_weight
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels)) #+ (relu_l2_loss + output_loss)/2
            #tf.verify_tensor_all_finite(loss, "loss not finite")

            # Optimizer.
            #self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(initial_lr, global_step, lr_decay_steps, 0.95)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.optimizer = self.optimizer.minimize(self.loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            self.valid_prediction = tf.nn.softmax(construct_logits(tf_valid_dataset, False))
            self.test_prediction = tf.nn.softmax(construct_logits(tf_test_dataset, False))

            # Add an op to initialize the variables.
            self.init_op = tf.initialize_all_variables()

            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

class ConvNetwork(Network):
    def __init__(self, batch_size, data, patch_size, stride1, stride2, with_dropout, initial_weights, initial_lr, lr_decay_steps, depth1, depth2, num_hidden1, num_hidden2, num_hidden3):
        super().__init__(batch_size, True, data)
        _, valid_dataset, test_dataset = data.datasets_2d()

        with self.graph.as_default():
            # Input data.
            self.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(batch_size, data.image_size, data.image_size, data.num_channels))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, data.num_labels))
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            # Variables.
            layer1_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, data.num_channels, depth1], stddev=initial_weights))
            layer1_biases = tf.Variable(tf.zeros([depth1]))
            layer2_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, depth1, depth2], stddev=initial_weights))
            layer2_biases = tf.Variable(tf.constant(0.1, shape=[depth2]))
            layer3_weights = tf.Variable(tf.truncated_normal(
                [data.image_size // (stride1*stride2) * data.image_size // (stride1*stride2) * depth2, num_hidden1], stddev=initial_weights))
            layer3_biases = tf.Variable(tf.constant(0.1, shape=[num_hidden1]))
            layer4_weights = tf.Variable(tf.truncated_normal(
                [num_hidden1, num_hidden2], stddev=initial_weights))
            layer4_biases = tf.Variable(tf.constant(0.1, shape=[num_hidden2]))
            layer5_weights = tf.Variable(tf.truncated_normal(
                [num_hidden2, num_hidden3], stddev=initial_weights))
            layer5_biases = tf.Variable(tf.constant(0.1, shape=[num_hidden3]))
            layer6_weights = tf.Variable(tf.truncated_normal(
                [num_hidden3, data.num_labels], stddev=initial_weights))
            layer6_biases = tf.Variable(tf.constant(0.1, shape=[data.num_labels]))

            # Model.
            def model(data, training):
                conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer1_biases)
                hidden = tf.nn.max_pool(hidden, ksize=[1, stride1, stride1, 1], strides=[1, stride1, stride1, 1], padding='SAME')
                if training and with_dropout:
                    hidden = tf.nn.dropout(hidden, with_dropout)

                conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer2_biases)
                hidden = tf.nn.max_pool(hidden, ksize=[1, stride2, stride2, 1], strides=[1, stride2, stride2, 1], padding='SAME')
                if training and with_dropout:
                    hidden = tf.nn.dropout(hidden, with_dropout)

                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                if training and with_dropout:
                    hidden = tf.nn.dropout(hidden, with_dropout)

                hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
                if training and with_dropout:
                    hidden = tf.nn.dropout(hidden, with_dropout)

                hidden = tf.nn.relu(tf.matmul(hidden, layer5_weights) + layer5_biases)
                if training and with_dropout:
                    hidden = tf.nn.dropout(hidden, with_dropout)

                return tf.matmul(hidden, layer6_weights) + layer6_biases

            # Training computation.
            logits = model(self.tf_train_dataset, True)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))

            # Optimizer.
            #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(initial_lr, global_step, lr_decay_steps, 0.95)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.optimizer = self.optimizer.minimize(self.loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            self.valid_prediction = tf.nn.softmax(model(tf_valid_dataset, False))
            self.test_prediction = tf.nn.softmax(model(tf_test_dataset, False))

            # Add an op to initialize the variables.
            self.init_op = tf.initialize_all_variables()

            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

def rand_float(low, high):
    return 10**np.random.uniform(np.log10(low), np.log10(high))

def rand_int(low, high):
    return int(rand_float(low, high))

def run_random_fc(data, best_so_far):
    num_steps = 10000

    batch_size = 128

    num_hidden1 = 2415 #rand_int(500, 3200)
    num_hidden2 = 1151 #rand_int(150, 2600)
    num_hidden3 = 619 #rand_int(30, 1800)
    with_dropout = rand_float(0.9, 1)

    initial_weights = rand_float(0.0005, 0.04)
    initial_lr = rand_float(0.0002, 0.003)
    lr_decay_steps = rand_float(100, 5000)

    network = FCNetwork(batch_size, data, [num_hidden1, num_hidden2, num_hidden3], with_dropout, initial_weights, initial_lr, lr_decay_steps)
    with network.session() as session:
        total_time = network.train(session, num_steps)
        test_accuracy, _, _ = network.test(session, best_so_far, "best-fc.ckpt")
    return (test_accuracy, with_dropout, initial_weights, initial_lr, lr_decay_steps, num_hidden1, num_hidden2, num_hidden3, total_time)

def best_fc_network(data):
    return FCNetwork(128, data, [2415, 1151, 619], 0.9635408412392467, 0.001, 0.001, 200)

def run_random_conv(data):
    num_steps = 10000

    batch_size = 128

    #(97.540000000000006, 0.9347917370089803, 0.006684308476125528, 0.0013110400348753447, 600.8572323214571, 16, 16, 4043, 553, 425, 234.04074001312256)
    best_so_far = 97.54

    patch_size = 5
    stride1 = 2
    stride2 = 2
    depth1 = 16
    depth2 = 16
    num_hidden1 = 4043
    num_hidden2 = 553
    num_hidden3 = 425
    with_dropout = rand_float(0.9, 1)
    initial_weights = rand_float(0.0005, 0.04)
    initial_lr = rand_float(0.0002, 0.003)
    lr_decay_steps = rand_float(100, 5000)

    network = ConvNetwork(batch_size, data, patch_size, stride1, stride2, with_dropout, initial_weights, initial_lr, lr_decay_steps, depth1, depth2, num_hidden1, num_hidden2, num_hidden3)

    with network.session() as session:
        total_time = network.train(session, num_steps)
        test_accuracy, _, _ = network.test(session, best_so_far, "best-convolution.ckpt")
    return (test_accuracy, with_dropout, initial_weights, initial_lr, lr_decay_steps, depth1, depth2, num_hidden1, num_hidden2, num_hidden3, total_time)

def best_conv_network(data):
    return ConvNetwork(128, data, 5, 2, 2, 0.9347917370089803, 0.006684308476125528, 0.0013110400348753447, 600.8572323214571, 16, 16, 4043, 553, 425)

def main():
    data = Data()

    #FC: (95.930000000000007, 0.9635408412392467, 729, 1336, 1430, 82.64951276779175)
    best_so_far = 95.93

    for _ in range(1000):
        results = run_random_fc(data, best_so_far)
        print(results)
        test_accuracy = results[0]
        if test_accuracy > best_so_far:
            best_so_far = test_accuracy

if __name__ == "__main__":
    main()
