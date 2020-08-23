import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class LinearRegression:
    def __init__(self, lr=0.01, iters=100, fit_intercept=True):
        self.lr = lr
        self.iters = iters
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.X = tf.convert_to_tensor(X, tf.float32)

        assert(len(X.shape) == 2)
        assert(len(X) == len(y))
        self.nparams = X.shape[1]

        if self.fit_intercept:
            ones = tf.constant(1, shape=(len(X),1), dtype=tf.float32)
            self.params = tf.Variable(TruncatedNormal()(shape=(self.nparams+1, 1)), dtype=tf.float32)
            self.X = tf.concat([self.X, ones], axis=-1)
        else:
            self.params = tf.Variable(TruncatedNormal()(shape=(self.nparams, 1)), dtype=tf.float32)
        self.y = y

        for i in range(self.iters):
            self.fit_single_step()

    def fit_single_step(self):

        with tf.GradientTape() as tape:
            y_hat = tf.matmul(self.X, self.params)
            loss = tf.reduce_mean(tf.square(y_hat - self.y))
            print('Loss is: ', loss)
            dloss_dx = tape.gradient(loss, self.params)
            #print(dloss_dx, self.lr)
            self.params = tf.Variable(self.params - dloss_dx * self.lr)

