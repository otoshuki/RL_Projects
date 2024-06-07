import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L

def createActor(obs_N, act_N):
    """
    Create Actor Model: Pi(a|s)
    inp->fc1->relu->fc2->relu->fc3->softmax out
    """
    inp = keras.Input(shape=obs_N, name='obs_input')
    x = L.Dense(64, kernel_initializer='he_uniform', activation='relu', name='fc1')(inp)
    x = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc2')(x)
    out = L.Dense(act_N, kernel_initializer='he_uniform', activation='softmax', name='fc3')(x)
    model = keras.Model(inputs=inp, outputs=out, name='actor')
    return model

def createCritic(obs_N):
    """
    Create Critic Model: V(s)
    inp->fc1->relu->fc2->relu->fc3->linear out
    """
    inp = keras.Input(shape=obs_N, name='obs_input')
    x = L.Dense(64, kernel_initializer='he_uniform', activation='relu', name='fc1')(inp)
    x = L.Dense(32, kernel_initializer='he_uniform', activation='relu', name='fc2')(x)
    out = L.Dense(1, kernel_initializer='he_uniform', name='fc3')(x)
    model = keras.Model(inputs=inp, outputs=out, name='critic')
    return model

class PPOClipAgent:
    def __init__(self, obs_N, act_N, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9, clip=0.2, lmbda=0.95, summary=None):
        #Agent parameters
        self.obs_N = obs_N
        self.act_N = act_N
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        #Networks
        self.actor = createActor(obs_N, act_N)
        self.critic = createCritic(obs_N)
        #Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)
        #PPO specifics
        self.clip = clip
        self.lmbda = lmbda

    def get_action(self, obs):
        """
        Return a discrete action from actor's softmax distribution
        """
        action_probs = self.actor(np.atleast_2d(obs)).numpy().flatten()
        action = np.random.choice(np.arange(self.act_N), p=action_probs)
        return action, action_probs[action]

    def get_gae_and_rtg(self, s, r, next_s, done):
        """
        Use generalized advantage estimation of the current batch of samples
        Also calculate the rewards to go, which is nothing but sum of discounted rewards from current state
        """
        #Current critic estimate of V(s)
        v_vals_s = self.critic(np.atleast_2d(s)).numpy()
        #Current critic estimate of V(s')
        #Please note this is essentially shifted i.e. s'(t)=s(t+1), but using it here to make implementation easy
        v_vals_next_s = self.critic(np.atleast_2d(next_s)).numpy()
        #Buffer
        advantage = np.zeros(len(r)+1)
        rewards_to_go = np.zeros(len(r)+1)
        #td_T calculation
        td_err = r[-1] + self.gamma*v_vals_next_s[-1]*(1-done) - v_vals_s[-1]
        #r_T calculation
        rewards_to_go[-1] = r[-1]
        for i in reversed(range(len(r))):
            #A_t = td_t + gamma*lambda*A_t+1
            advantage[i] = td_err + self.gamma*self.lmbda*advantage[i+1]
            #td_t-1 calculation
            td_err = r[i] + self.gamma*v_vals_next_s[i] - v_vals_s[i]
            #Rewards to go
            rewards_to_go[i] = r[i] +  self.gamma*(rewards_to_go[i+1])
        return advantage[:-1], rewards_to_go[:-1]

    def train_step(self, s, a, r, next_s, done, logprobs):
        """
        Train using experiences
        """
        #Get advantage and rtg values
        adv, rtg = self.get_gae_and_rtg(s, r, next_s, done)
        #Actor Update
        actor_learnables = self.actor.trainable_variables
        with tf.GradientTape() as tape:
            curr_logprobs = tf.math.log(self.actor(np.atleast_2d(s)))
            red_logprobs = tf.reduce_sum(curr_logprobs * tf.one_hot(a, self.act_N), axis=1)
            ratio = tf.math.exp(red_logprobs-logprobs.flatten())
            surrogate1 = ratio*adv
            surrogate2 = tf.clip_by_value(ratio, 1-self.clip, 1+self.clip)*adv
            actor_loss = -tf.math.reduce_mean(tf.math.minimum(surrogate1, surrogate2))
            actor_grad = tape.gradient(actor_loss, actor_learnables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_learnables))
        #Critic Update
        critic_learnables = self.critic.trainable_variables
        with tf.GradientTape() as tape:
            v_vals_s = self.critic(np.atleast_2d(s))
            critic_loss = keras.losses.mean_squared_error(rtg, v_vals_s)
            critic_grad = tape.gradient(critic_loss, critic_learnables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic_learnables))
        return actor_loss.numpy(), critic_loss.numpy()
