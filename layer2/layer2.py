import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

os.makedirs('outputs', exist_ok=True)

# ── 1. LOAD + PREPROCESS (same as layer1) ────────────────────────────────────
df = pd.read_csv('datasets/creditcard.csv')

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time']   = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]

# ── 2. BUILD LAYER 2 ENVIRONMENT ─────────────────────────────────────────────
fraud_df = df[df['Class'] == 1]
legit_df = df[df['Class'] == 0]

# Load Layer 1 to identify hard cases for evaluation later
layer1_model = joblib.load('outputs/layer1_model.pkl')
X_fraud = fraud_df.drop('Class', axis=1)
fraud_probs_all = layer1_model.predict_proba(X_fraud)[:, 1]

# Identify hard cases (Layer 1 uncertain) — used for evaluation only
hard_fraud = fraud_df[fraud_probs_all < 0.80]
print(f"Hard fraud (L1 uncertain):  {len(hard_fraud)}")
print(f"Easy fraud (L1 confident):  {(fraud_probs_all >= 0.80).sum()}")

# Train on ALL fraud — agent needs enough signal to learn
df_layer2 = pd.concat([
    fraud_df,                               # all fraud types
    legit_df.sample(5000, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Layer 2 environment size:   {len(df_layer2)}")
print(f"Fraud in Layer 2 env:       {df_layer2['Class'].sum()}")

# ── 3. FRAUD DETECTION ENVIRONMENT ───────────────────────────────────────────
class FraudEnv(gym.Env):
    """
    Custom Gymnasium environment for fraud detection.

    At each step the agent sees one transaction (30 features)
    and must decide:
        0 = Approve
        1 = Flag for review
        2 = Block
    """

    def __init__(self, dataframe):
        super(FraudEnv, self).__init__()

        self.df = dataframe.reset_index(drop=True)
        self.n_samples = len(self.df)
        self.current_index = 0

        # Observation: 30 transaction features
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(30,),
            dtype=np.float32
        )

        # Action: 0=Approve, 1=Flag, 2=Block
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        true_label = self.df.loc[self.current_index, 'Class']
        reward = self._get_reward(action, true_label)

        self.current_index += 1
        terminated = self.current_index >= self.n_samples
        truncated = False

        if terminated:
            obs = self._get_observation(last=True)
        else:
            obs = self._get_observation()

        return obs, reward, terminated, truncated, {}

    def _get_observation(self, last=False):
        idx = self.current_index if not last else self.current_index - 1
        row = self.df.loc[idx].drop('Class').values.astype(np.float32)
        return row

    def _get_reward(self, action, true_label):
        """
        Reward function — the heart of the RL agent.

        true_label 0 = legitimate transaction
        true_label 1 = fraud

        action 0 = Approve
        action 1 = Flag
        action 2 = Block
        """
        if action == 2 and true_label == 1:
            return 3.0      # Correct block — caught fraud

        if action == 0 and true_label == 0:
            return 1.0      # Correct approval — legitimate passed through

        if action == 1 and true_label == 1:
            return 1.5      # Flagged fraud — acceptable

        if action == 1 and true_label == 0:
            return -0.5     # Flagged legitimate — minor false alarm

        if action == 2 and true_label == 0:
            return -1.0     # False positive — blocked legitimate

        if action == 0 and true_label == 1:
            return -5.0     # Missed fraud — worst outcome

        return 0.0


# ── 4. TRAIN PPO AGENT ────────────────────────────────────────────────────────
env = FraudEnv(df_layer2)

model_rl = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    seed=42
)

print("\nTraining PPO agent...")
model_rl.learn(total_timesteps=100000)
model_rl.save("outputs/layer2_ppo_model")
print("RL model saved")

# ── 5. EVALUATE ON ALL TRAINING DATA ─────────────────────────────────────────
print("\nEvaluating on training environment...")

obs, _ = env.reset()
all_actions = []
all_labels  = []

for i in range(len(df_layer2)):
    action, _ = model_rl.predict(obs, deterministic=True)
    all_actions.append(int(action))
    all_labels.append(int(df_layer2.loc[i, 'Class']))
    obs, reward, terminated, truncated, _ = env.step(int(action))
    if terminated:
        break

all_actions = np.array(all_actions)
all_labels  = np.array(all_labels)
binary_preds = (all_actions == 2).astype(int)

print("\n── RL Agent Results (full env) ──")
print(classification_report(all_labels, binary_preds))

caught = ((binary_preds == 1) & (all_labels == 1)).sum()
total  = all_labels.sum()
print(f"Fraud cases:       {total}")
print(f"Caught by agent:   {caught}")
print(f"Catch rate:        {caught/total:.2%}")

# ── 6. EVALUATE ON HARD CASES (Layer 1 uncertain) ────────────────────────────
print("\nEvaluating on hard fraud cases (L1 uncertain)...")

hard_legit   = legit_df.sample(500, random_state=99)
hard_eval_df = pd.concat([hard_fraud, hard_legit]).sample(frac=1, random_state=99).reset_index(drop=True)

eval_env = FraudEnv(hard_eval_df)
obs, _   = eval_env.reset()
hard_actions, hard_labels = [], []

for i in range(len(hard_eval_df)):
    action, _ = model_rl.predict(obs, deterministic=True)
    hard_actions.append(int(action))
    hard_labels.append(int(hard_eval_df.loc[i, 'Class']))
    obs, _, terminated, _, _ = eval_env.step(int(action))
    if terminated:
        break

hard_actions = np.array(hard_actions)
hard_labels  = np.array(hard_labels)
hard_preds   = (hard_actions == 2).astype(int)

print("\n── RL Agent Results (hard cases only) ──")
print(classification_report(hard_labels, hard_preds))

hard_caught = ((hard_preds == 1) & (hard_labels == 1)).sum()
hard_total  = hard_labels.sum()
print(f"Hard fraud cases:  {hard_total}")
print(f"Caught by agent:   {hard_caught}")
print(f"Catch rate:        {hard_caught/hard_total:.2%}")