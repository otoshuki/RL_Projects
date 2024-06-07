import numpy as np
from collections import deque
import json
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import datetime as dt

def createActor(obs_N, act_N, act_min=0.001, act_max=1):
    """
    Create Actor Model
    inp->fc1->tanh->fc2->tanh->fc3->sigmoid->out
    """
    inp = keras.Input(shape=obs_N, name='act_input')
    x = L.Dense(64, kernel_initializer='he_uniform', activation='relu', name='fc1')(inp)
    x = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc2')(x)
    x = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc3')(x)
    x = L.Dense(act_N, kernel_initializer='he_uniform', activation='sigmoid', name='fc4')(x)
    out = tf.clip_by_value(x, act_min, act_max)
    model = keras.Model(inputs=inp, outputs=out, name='actor')
    return model

def createCritic(obs_N, act_N):
    """
    Create Critic Model
    inp->fc1->relu->fc2->relu->fc3->out
    """
    inp1 = keras.Input(shape=obs_N, name='obs_input')
    inp2 = keras.Input(shape=act_N, name='act_input')
    inp = tf.concat([inp1, inp2], -1, name='concat')
    x1 = L.Dense(64, kernel_initializer='he_uniform', activation='relu', name='fc11')(inp)
    x1 = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc12')(x1)
    x1 = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc13')(x1)
    out1 = L.Dense(1, kernel_initializer='he_uniform', name='fc14')(x1)
    x2 = L.Dense(64, kernel_initializer='he_uniform', activation='relu', name='fc21')(inp)
    x2 = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc22')(x2)
    x2 = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc23')(x2)
    out2 = L.Dense(1, kernel_initializer='he_uniform', name='fc24')(x2)

    model = keras.Model(inputs=[inp1, inp2], outputs=[out1, out2], name='critic')
    return model

class TD3Agent:
    def __init__(self, obs_N, act_N, current_time,
                act_min=0.001, act_max=1,
                actor_lr=1e-3, critic_lr=1e-3,
                gamma=0.8, polyak = 5e-3,
                exp_noise_std=0.2, exp_noise_decay=1e-3,
                pol_noise_std=0.2, pol_noise_clip=0.5,
                actor_freq=2, target_freq=2,
                load_model="", summaries=True):
        self.current_time = current_time
        self.summaries = summaries
        self.iteration = 0
        #Inp/Out Definitions
        self.obs_dim = obs_N
        self.act_dim = act_N
        self.act_min = act_min
        self.act_max = act_max
        #Networks
        self.actor = createActor(obs_N, act_N, act_min, act_max)
        self.critic = createCritic(obs_N, act_N)
        self.actor_target = createActor(obs_N, act_N, act_min, act_max)
        self.critic_target = createCritic(obs_N, act_N)

        #Transfer parameters
        for t, e in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
            t.assign(e)
        for t, e in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            t.assign(e)
        print("DBG: Created new models")
        #Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)
        #Parameters
        self.gamma = gamma
        self.polyak = polyak
        self.exp_noise_std = exp_noise_std
        self.exp_noise_decay = exp_noise_decay
        self.pol_noise_std = pol_noise_std
        self.pol_noise_clip = pol_noise_clip
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        #Implement Load Model
        #TBD
        #Tensorboard summaries
        if self.summaries:
            self.train_writer = tf.summary.create_file_writer('./logs2/' + self.current_time)

    def get_action(self, obs, noise=False):
        """
        Return action
        """
        action = self.actor(obs)
        if noise:
            noise = tf.random.normal(action.shape, mean=0, stddev=self.exp_noise_std)
            action = tf.clip_by_value(action+noise, self.act_min, self.act_max)
        return action.numpy()

    def train_step(self, buffer, batch_size=128):
        """
        Train using experiences, assuming normal expreplay
        """
        self.iteration += 1
        #Sample
        [s,a,r,next_s,done] = buffer.sample(batch_size)
        #Update Critic
        #Sections without autodiff
        next_action = self.actor_target(next_s)
        noise = tf.random.normal(next_action.shape, mean=0, stddev=self.pol_noise_std)
        noise = tf.clip_by_value(noise, -self.pol_noise_clip, self.pol_noise_clip)
        noisy_action = tf.clip_by_value(next_action + noise, self.act_min, self.act_max)
        #Q1,2(s',a')
        tq1, tq2 = self.critic_target([next_s, noisy_action])
        target_q = tf.math.minimum(tq1, tq2)
        #TD target
        td_target = tf.stop_gradient(r + (1-done)*self.gamma*target_q)
        #Sections with autodiff
        critic_learnables = self.critic.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(critic_learnables)
            mq1, mq2 = self.critic([s, a])
            critic_loss = tf.math.reduce_mean(tf.math.square(td_target-mq1) + tf.math.square(td_target-mq2))
            critic_grads = tape.gradient(critic_loss, critic_learnables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_learnables))
        self.add_critic_summary(critic_loss, td_target, mq1, mq2)
        #Check actor update
        if self.iteration%self.actor_freq == 0:
            actor_learnables = self.actor.trainable_variables
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actor_learnables)
                action = self.actor(s)
                q1, q2 = self.critic([s, action])
                actor_loss = -tf.math.reduce_mean(q1)
                actor_grads = tape.gradient(actor_loss, actor_learnables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, actor_learnables))
            self.add_actor_summary(actor_loss)
        #Check target update
        if self.iteration%self.target_freq == 0:
            for t, e in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
                t.assign(t*(1-self.polyak) + e*self.polyak)
            for t, e in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
                t.assign(t*(1-self.polyak) + e*self.polyak)
        #Reduce exploration
        self.exp_noise_std *= 1-self.exp_noise_decay

    def save_models(self, steps):
        """
        Save network weights
        """
        self.actor.save_weights("./models/{}/actor_{}".format(self.current_time, steps))
        self.actor_target.save_weights('./models/{}/actor_target_{}'.format(self.current_time, steps))

        self.critic.save_weights('./models/{}/critic_{}'.format(self.current_time, steps))
        self.critic_target.save_weights('./models/{}/critic_target_{}'.format(self.current_time, steps))

    def add_critic_summary(self, loss, td_target, m_q1, m_q2):
        if self.summaries:
            td_error1 = td_target - m_q1
            td_error2 = td_target - m_q2
            with self.train_writer.as_default():
                tf.summary.scalar('critic_loss', loss, step = self.iteration)

    def add_actor_summary(self, loss):
        if self.summaries:
            with self.train_writer.as_default():
                    tf.summary.scalar('actor_loss', loss, step = self.iteration)
