import numpy as np
import tensorflow as tf


class PPOAgent:
    def __init__(self, env, actor_lr=0.0003, critic_lr=0.0005, gamma=0.99, clip_ratio=0.2, epochs=10, batch_size=64):
        """
        Inicializa o agente PPO.
        :param env: O ambiente personalizado (NS3Environment).
        :param actor_lr: Taxa de aprendizagem para o ator.
        :param critic_lr: Taxa de aprendizagem para o crítico.
        :param gamma: Fator de desconto (gamma).
        :param clip_ratio: Fator de clipe para limitar mudanças na política.
        :param epochs: Quantidade de épocas para treinar em cada lote.
        :param batch_size: Tamanho do batch para treinamento.
        """
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size

        # Dimensões do ambiente
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.n

        # Redes do ator e do crítico
        self.actor = self.build_actor(actor_lr)
        self.critic = self.build_critic(critic_lr)

    def build_actor(self, learning_rate):
        """
        Cria o modelo do ator.
        :param learning_rate: Taxa de aprendizagem para o ator.
        :return: Modelo ator compilado.
        """
        inputs = Input(shape=(self.state_shape,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_shape, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=self.ppo_loss)
        return model

    def build_critic(self, learning_rate):
        """
        Cria o modelo do crítico.
        :param learning_rate: Taxa de aprendizagem para o crítico.
        :return: Modelo crítico compilado.
        """
        inputs = Input(shape=(self.state_shape,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    def ppo_loss(self, y_true, y_pred):
        """
        Define a função de perda PPO com base no fator de clipe.
        :param y_true: Vantagens esperadas e probabilidades antigas.
        :param y_pred: Probabilidades preditas pelo ator.
        :return: Perda PPO.
        """
        advantages, old_prediction_probs = y_true[:, :-1], y_true[:, -1]
        new_policy_probs = tf.reduce_sum(y_pred * old_prediction_probs, axis=1)
        ratio = new_policy_probs / (old_prediction_probs + 1e-10)

        clip_loss = K.minimum(ratio * advantages, K.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages)
        loss = -K.mean(clip_loss)
        return loss

    def choose_action(self, state):
        """
        Escolhe uma ação baseada na política atual (ator).
        :param state: Estado atual.
        :return: Ação escolhida e suas probabilidades.
        """
        state = np.expand_dims(state, axis=0)
        probabilities = self.actor.predict(state, verbose=0)
        action = np.random.choice(self.action_shape, p=probabilities[0])
        return action, probabilities[0]

    def discount_rewards(self, rewards, dones):
        """
        Calcula os retornos descontados para cada episódio.
        """
        discounted_rewards = []
        cumulative = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                cumulative = 0
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)
        return np.array(discounted_rewards)

    def train(self, states, actions, advantages, discounted_rewards, old_probs):
        """
        Treina a política (ator) e o crítico com os lotes fornecidos.
        """
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self.action_shape)

        # Preparar o batch de entrada para o ator
        y_true = np.hstack([advantages, old_probs.reshape(-1, 1)])

        # Treinar o ator
        self.actor.fit(states, y_true, epochs=self.epochs, verbose=0, batch_size=self.batch_size)

        # Treinar o crítico
        self.critic.fit(states, discounted_rewards, epochs=self.epochs, verbose=0, batch_size=self.batch_size)


# Loop de treinamento
def train_ppo_agent(episodes=1000, max_steps=100):
    """
    Treina o agente PPO no ambiente.
    :param episodes: Número de episódios.
    :param max_steps: Máximo de etapas por episódio.
    """
    env = NS3Environment(n_positions=10, n_vants=2)
    agent = PPOAgent(env)

    for episode in range(episodes):
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Escolher ação e probabilidades
            action, action_probs = agent.choose_action(state)

            # Executar ação e observar transição
            next_state, reward, done, _ = env.step(action)

            # Armazenar experiência
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_probs.append(action_probs[action])

            state = next_state
            total_reward += reward

            if done:
                break

        # Calcular vantagens e retornos descontados
        states = np.array(states)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        discounted_rewards = agent.discount_rewards(rewards, dones)
        values = agent.critic.predict(states, verbose=0).flatten()
        advantages = discounted_rewards - values

        # Normalizar vantagens
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # Treinar o agente
        agent.train(states, actions, advantages, discounted_rewards, old_probs)

        # Monitorar progresso
        print(f"Episódio: {episode + 1}, Recompensa Total: {total_reward}")

    env.close()


if __name__ == "__main__":
    train_ppo_agent()