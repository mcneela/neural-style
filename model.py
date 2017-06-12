import os
import tensorflow as tf
from util import *

def relu(layer):
    return tf.nn.relu(layer)

def conv2d(layer, filter_val):
    return tf.nn.conv2d(layer, filter_val, [1, 1, 1, 1], 'SAME')

def avg_pooling(layer):
    return tf.nn.avg_pool(layer, [1, 2, 2, 1],
                          [1, 2, 2, 1], 'SAME')

def _content_loss(p, x):
    loss = 0.5 * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

def _style_loss(a, x):
    M = a.shape[1] * a.shape[2]
    N = a.shape[3]

    A = gram(a, M, N)
    G = gram(x, M, N)

    loss = (1 / ((M ** 2) * (N ** 2))) \
            * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def L_s(sess, model):
    layers = [('conv1_1', .5), ('conv2_1', 1.0), 
              ('conv3_1', 1.5), ('conv4_1', 3.0),
              ('conv5_1', 4.0)]
    loss = 0
    for (name, weight) in layers:
        layer = model[name]
        loss += weight * _style_loss(sess.run(layer), layer)
    return loss

def L_c(sess, model):
    layer = 'conv4_2'
    return _content_loss(sess.run(model[layer]), model[layer])

def gram(f, M, N):
    f_t = tf.reshape(f, (M, N))
    return tf.matmul(tf.transpose(f_t), f_t)

def run_session():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    with tf.Session() as sess:
        cimg = generate_noise_image(load_img(CONTENT_IMG))
        simg = load_img(STYLE_IMG)

        model = load_model()

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(cimg))
        content_loss = L_c(sess, model)

        sess.run(model['input'].assign(simg))
        style_loss = L_s(sess, model)

        total_loss = alpha * style_loss + beta * content_loss

        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(total_loss)

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(cimg))

        for i in range(NUM_ITERS):
            sess.run(train_step)
            print(i)
            if (i % 100) == 0 and i != 0:
                img = sess.run(model['input'])
                print("Step: %d" % i)

                fname = (OUTPUT_DIR + "%d.png") % i
                save_img(img, fname)

def load_model():
    ll = load_vggNet(model_path)
    graph = {}
    graph['input'] = tf.Variable(np.zeros((1, IMG_HEIGHT,
                     IMG_WIDTH, NUM_CHANNELS)),
                     dtype = 'float32')
    prev_layer = graph['input'] 
    for layer in ll:
        name = layer['name']
        if layer['type'] == 'conv' and not layer['name'].startswith('fc'):
            W = tf.constant(layer['W'].astype('float32'))
            b = layer['b'].astype('float32')
            b = tf.constant(np.reshape(b, b.size))
            graph[name] = relu(conv2d(prev_layer, W) + b)
        elif layer['type'] == 'pool':
            graph[name] = avg_pooling(prev_layer)
        else:
            continue
        prev_layer = graph[name]
    return graph

def main():
    run_session()

if __name__ == '__main__':
    main()
