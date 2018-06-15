
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class model():
    def __init__(self, state_shape, n_actions, samples=100):

        tf.set_random_seed(42)
        np.random.seed(42)
        
        self.samples = samples
        self.written_imgs = 0

        def weight(shape, conv=False):
            #xavier init from tensorflow contrib layers implementation
            f = 6.0 if conv else 2.3
            in_plus_out = shape[-2] + shape[-1]
            std = math.sqrt(f / in_plus_out)
            w = tf.truncated_normal(shape, mean=0.0, stddev=std)
            return tf.Variable(w)

        def bias(shape, v=0.0):
            #bias init to 0.0 in xavier paper
            return tf.Variable(tf.constant(v, shape=shape))

        #def batch_normalize(x, activation='none'):
        #    norm = tf.contrib.layer.batch_norm(x, is_training=self.)
        #    if activation == 'relu':
        #        norm = tf.nn.relu(norm)
        #    elif activation == 'elu':
        #        norm = tf.nn.elu(norm)
        #    return norm

        def residual_layer(x, y, activation='none'):
            x = tf.contrib.layers.flatten(x)
            y = tf.contrib.layers.flatten(y)
            res = tf.add(x, y)
            if activation == 'relu':
                res = tf.nn.relu(res)
            elif activation == 'elu':
                res = tf.nn.elu(res)
            return res

        def lstm_layer(x, n=256):
            flat = tf.contrib.layers.flatten(x)
            seq = tf.expand_dims(flat, 0) #fake sequence
            cell = tf.nn.rnn_cell.BasicLSTMCell(n)
            out, state = tf.nn.dynamic_rnn(cell, seq, 
                    dtype=tf.float32)
            return tf.reshape(out, [-1, n])

        def conv_layer(x, filter_size=(3,3), out_channels=32, 
                activation='none', stride=(1,1)):
            if len(x.get_shape()) is 3: #for non img input
                x = tf.reshape(x, (-1,) + state_shape + (1,))
            #filter shape = [height, width, in_channels, out_channels]
            filter_shape = filter_size + (x.get_shape()[-1].value, 
                    out_channels)
            w_conv = weight(filter_shape, conv=True)
            b_conv = bias([out_channels])
            conv = tf.nn.conv2d(x, w_conv, 
                    strides=(1,) + stride + (1,), padding='SAME') + b_conv
            if activation == 'relu':
                conv = tf.nn.relu(conv)
            elif activation == 'elu':
                conv = tf.nn.elu(conv)
            return conv

        def dense_layer(x, n=512, activation='none', drop=False):
            flat = tf.contrib.layers.flatten(x)
            w = weight([flat.get_shape()[-1].value, n])
            b = bias([n])
            dense = tf.matmul(flat, w) + b
            if activation == 'relu':
                dense = tf.nn.relu(dense)
            elif activation == 'elu':
                dense = tf.nn.elu(dense)
            dense = tf.nn.dropout(dense, self.keep_prob) if drop else dense
            return dense

        def minimize(x, rate=1e-4):
            step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(rate)
            return optimizer.minimize(x, global_step=step), step

        #FIXME:
        def mem_layer(n=1024):
            w = weight([n, n])
            b = bias([n])

        with tf.name_scope('input'):
            self.state_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.action_in = tf.placeholder(tf.float32, [None, n_actions])
            self.reward_in = tf.placeholder(tf.float32, [None])
            self.advantage_in = tf.placeholder(tf.float32, [None])
            self.nextstate_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.keep_prob = tf.placeholder(tf.float32)

        #DEBUG
        x = self.state_in
        batch_size = tf.shape(self.state_in)[0]
        #x = tf.contrib.layers.flatten(x)

        #choosing an action based on:
            #can the following things be learned?
                #black box possible

            #ideas
                #exploration
                    #aka encourage exploration of new states
                        #especially high reward states
                    #lstm sequence of actions
                        #even need this?
                            #state prediction will be equivalent to evaling
                            #long action sequences
                        #predicts next action
                        #used for similar action sequences
                        #used for boredom
                    #lstm sequence of states
                        #state depends on reward also
                            #same state w diff reward is different
                        #predicts next state?
                        #if pred - curr close to 0 --> bored
                #reward
                    #if bored make own reward fn
                        #choose random env state
                            #get to that state
                            #repeat on success with new choice
                        #choose random state with known better reward
                            #how to handle sparse rewards
                    #sequence of states has local reward
                        #like jump-dash, jump-dash-attack
                #memory
                    #

            #memory
                #transfer learning
                #look ahead
                #predict weights for following items?
            #reward maximization
                #length of time
                    #maximize the time
                    #ignore when all episodes same length
                    #take (running avg, curr - prev) of episode lengths
                        #use when rewards are sparse?
                #sparse rewards
                    #may seem like nothing is happening
                    #
                #create own rewards (based on given rewards?, bored?)
                    #gain hight, left right
                    #gain points
                    #etc
            #exploration
                #aka encourage exploration of new states
                #explore more if bored or doing poorly
                    #env doesnt change much
                    #returning to same state
                    #bad rewards
                #explore less on good rewards, rapid change of states
                #not just random moves, random moves down unexplored path
                #similar action sequences should be ignored?


        #FIXME:
        #with tf.name_scope('recall'):
        #    #predict next state
        #    lstm_next = lstm_layer(x)
        #    logit_next = dense_layer(lstm_next, np.prod(state_shape), 
        #            'relu')
        #
        #    #predicted state, actual state
        #        #how to differentiate same states with different rewards
        #            #add rewards?
        #        #subtract, reduce sum, square or absolute val
        #        #this is metric for how familiar a state is
        #            #by extension how effective action sequence is?
        #    state_next = tf.reshape(self.nextstate_in[-1], 
        #            [-1, np.prod(state_shape)])
        #    similar = tf.reduce_sum(
        #            tf.square(tf.subtract(logit_next, state_next)))
        #    #convert to percentages, reverse them 0-->1, .3-->.7, .8-->.2
        #        #uncommenting this ruins training...
        #    #similar_prob = tf.abs(tf.subtract(
        #    #        tf.nn.softmax(similar), 1.0))
        #
        #    #loss
        #    similar_loss = similar
            
        
        with tf.name_scope('adversarial_generation'):
            #train weights to get to a target goal state
                #pick a state from prev episode
                    #keep trying until achieve that state
                #play game with reward input
                #resample goal state

                #take patch of current state
                    #move around target state
                    #reward is max (patch - target state_patch)
            #predict the next state pixel by pixel
                #0 to 255

            #r_c = current state reward
            #r_t = target state reward (max(r_t, 1)?)
                #negative rewards, zero rewards
                #aka on pick a "bad" state
            #s = similarity to target state + 1e-8
            #reward used is function of similarity
                #and function of current reward
                    #aka reward 
                #r = 1/s * r_t + r_

            #NEW IDEAS
                #PICK MINI GOALS ALONG THE WAY TO GOAL
                #feed reverse states into dynamic lstm
                    #this way we can predict the goal for the current
                    #state in relation to the target state on choose action

            #h = dense_layer(x, 1024, 'relu')
            #logit_pred = dense_layer(h, np.prod(state_shape))
            #FIXME

                #evals how good an img is
                    #do we even need this?
                #input is random distribution
                    #what if distribution is based on state in
                #input an img for each action
                    #shape [batchsize, n_actions, flat(state_shape)]
                #output value for each img
                    #multiply logits by action input
                    #eval the image given the action
                    #shape [batchsize]

            #FIXME:
                #predict a img for each action

            source = tf.contrib.layers.flatten(self.state_in)
            target = tf.contrib.layers.flatten(self.nextstate_in)
            #z_dist = tf.distributions.Uniform(low=-1.0, high=1.0, 
            #        allow_nan_stats=False)
            self.uniform_in = tf.placeholder(tf.float32, 
                    shape=[None, self.samples])

            h = 2048
            s = source.get_shape()[-1].value

            #discriminator weights
            d_w1 = weight([s, h])
            d_b1 = bias([h])
            d_w2 = weight([h, 1])
            d_b2 = bias([1])

            #generator weights
            g_w1 = weight([self.samples, h])
            g_b1 = bias([h])
            g_w2 = weight([h, s])
            g_b2 = bias([s])

            d_vars = [d_w1, d_b1, d_w2, d_b2]
            g_vars = [g_w1, g_b1, g_w2, g_b2]

            def generator(z):
                g_h1 = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
                g_log = tf.matmul(g_h1, g_w2) + g_b2
                return tf.nn.sigmoid(g_log)

            def discriminator(x):
                d_h1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
                d_logit = tf.matmul(d_h1, d_w2) + d_b2
                d_prob = tf.nn.sigmoid(d_logit)
                return d_prob, d_logit

            #generate a fake img, pass real, fake through disciminator
            real = tf.nn.sigmoid(source)
            #real = target
            self.fake = generator(self.uniform_in)
            real_prob, real_logit = discriminator(real)
            fake_prob, fake_logit = discriminator(self.fake)

            #vanilla loss?
                #use negative to minimize (paper maximizes, tf only mins
            #d_loss = - tf.reduce_mean(tf.log(real_prob) + tf.log(
            #        1.0 - fake_prob))
            #g_loss = - tf.reduce_mean(tf.log(fake_prob))
            
            #alternative vanilla loss?
            #d_loss_real = tf.reduce_mean(
            #        tf.nn.sigmoid_cross_entropy_with_logits(
            #        logits=real_logit, labels=tf.ones_like(real_logit)))
            #d_loss_fake = tf.reduce_mean(
            #        tf.nn.sigmoid_cross_entropy_with_logits(
            #        logits=fake_logit, labels=tf.zeros_like(fake_logit)))
            #d_loss = d_loss_real + d_loss_fake
            #g_loss = tf.reduce_mean(
            #        tf.nn.sigmoid_cross_entropy_with_logits(
            #        logits=fake_logit, labels=tf.ones_like(fake_logit)))

            #least squares loss
            d_loss = 0.5 * tf.reduce_mean(tf.square(real_logit - 1.0)
                    ) + tf.reduce_mean(tf.square(fake_logit))
            g_loss = 0.5 * tf.reduce_mean(tf.square(fake_logit - 1))

            #optimize
            self.d_op = tf.train.AdamOptimizer(1e-4).minimize(d_loss, 
                    var_list=d_vars)
            self.g_op = tf.train.AdamOptimizer(1e-4).minimize(g_loss, 
                    var_list=g_vars)





        #FIXME:
        with tf.name_scope('DEBUG_COMBINE'):
            #things to try
            #x = source - self.fake
            #x = source - target
            #x = source - target * source - self.fake
            #x = self.fake
            pass






        with tf.name_scope('model'):

            def universe_model(x):
                for _ in range(4):
                    x = conv_layer(x, (3,3), 32, 'elu', (2,2))
                return lstm_layer(x)

            def conv_residual_model(x):
                x = conv_layer(x, (3,3), 32)
                a = conv_layer(x, (1,1), 32)
                a = conv_layer(a, (3,3), 32)
                x = residual_layer(x, a, 'relu')
                return dense_layer(x, 512, 'relu', drop=True)

            def conv_lstm_model(x):
                x = conv_layer(x, (3,3), 32, 'relu')
                x = conv_layer(x, (3,3), 32, 'relu')
                x = lstm_layer(x)
                return dense_layer(x, 512, 'relu', drop=True)
            
            def small_conv(x):
                x = conv_layer(x, (3,3), 32)
                return dense_layer(x, 256, 'relu', drop=True)

            def linear_model(x, activation='relu'):
                return dense_layer(x, 256, activation, drop=True)

            #just using debug for now
            #x = conv_layer(x, (3,3), 16, 'elu')
            #x = dense_layer(x, 1024, 'elu', drop=True)

        with tf.name_scope('policy'):
            logits_class = dense_layer(x, n_actions)
            probs_class = tf.nn.softmax(logits_class) + 1e-8
            logprobs_class = tf.nn.log_softmax(logits_class) + 1e-8
            action_dist = tf.multinomial(logits_class - tf.reduce_max(
                    logits_class, [1], keepdims=True), 1)
            self.action = tf.squeeze(action_dist, [1])
            self.test_action = tf.argmax(probs_class, axis=1)

        with tf.name_scope('value'):
            logit_val = dense_layer(x, 1)
            self.value = tf.reduce_sum(logit_val, axis=1)

        with tf.name_scope('loss'):
            entropy = - tf.reduce_sum(probs_class * logprobs_class)
            probs_act = tf.reduce_sum(logprobs_class * self.action_in, [1])
            policy_loss = - tf.reduce_sum(probs_act * self.advantage_in)
            value_loss = 0.05 * tf.reduce_sum(tf.square(
                    self.value - self.reward_in))
            self.loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        with tf.name_scope('optimize'):
            self.optimize, self.step = minimize(self.loss)

        with tf.name_scope('summary'):
            tf.summary.scalar('1_total_loss', tf.divide(self.loss, 
                    tf.cast(batch_size, tf.float32)))
            tf.summary.scalar('2_value_loss', value_loss)
            tf.summary.scalar('3_policy_loss', policy_loss)
            tf.summary.scalar('4_entropy', entropy)
            tf.summary.scalar('5_discriminator_loss', d_loss)
            tf.summary.scalar('6_generator_loss', g_loss)
            self.summaries = tf.summary.merge_all()

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('./logs', self.loss.graph)
        #for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #    print(v.name, v.shape)
        self.sess.run(tf.global_variables_initializer())

    def act(self, state, explore=True):
        action_op = self.action if explore else self.test_action
        action, value = self.sess.run([action_op, self.value], 
                feed_dict={
                    self.state_in: [state],
                    self.uniform_in: np.random.uniform(-1.0, 1.0, 
                            size=[1, self.samples]),
                    self.keep_prob: 1.0,
                })
        return action[0], value[0]

    def learn(self, batch, sample=False):
        states, actions, rewards, advantage, dones, next_states = batch
        loss, _, step, summ, _, _, fakes = self.sess.run([self.loss, 
                self.optimize, self.step, self.summaries, self.g_op, 
                self.d_op, self.fake],
                feed_dict={
                    self.state_in: states,
                    self.action_in: actions,
                    self.reward_in: rewards,
                    self.advantage_in: advantage,
                    self.nextstate_in: next_states,
                    self.uniform_in: np.random.uniform(-1.0, 1.0, 
                            size=[states.shape[0], self.samples]),
                    self.keep_prob: 0.5,
                })

        #sample real and fake imgs, write to logs
        if sample:
            fakes = np.reshape(fakes, states.shape)
            #states = 1.0 / (1.0 + np.exp(-states))
            #plt.imshow(states[0], cmap='gray')
            #plt.show()

            #random choices from both
            n_imgs = 4
            n_imgs = min(n_imgs, states.shape[0]-1)
            if n_imgs:
                idx = np.random.choice(np.arange(states.shape[0]), n_imgs, 
                        replace=False)
                states = states[idx, :, :]
                fakes = fakes[idx, :, :]

            n = states.shape[0]
            f, a = plt.subplots(nrows=2, ncols=n, figsize=(3*n,3*3))
            f.suptitle('real(top) vs generated(bottom) - %s' % (
                    self.written_imgs,), fontsize=20)
            f.tight_layout()
            for i in range(n * 2):
                if len(a.shape) == 1: #subplots() returns squeezed arrays
                    a[0].imshow(states[0], cmap='gray')
                    a[0].set_axis_off()
                    a[1].imshow(fakes[0], cmap='gray')
                    a[1].set_axis_off()
                    break
                if i < n:
                    a[0, i].imshow(states[i], cmap='gray')
                    a[0, i].set_axis_off()
                else:
                    a[1, i - n].imshow(fakes[i - n], cmap='gray')
                    a[1, i - n].set_axis_off()

            #plt.show()
            f.savefig('logs/generator_%s.jpg' % self.written_imgs)
            plt.close(f)
            self.written_imgs += 1

        self.writer.add_summary(summ, step)
        return loss

