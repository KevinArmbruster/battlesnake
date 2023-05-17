from gymBattlesnake.gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from gymBattlesnake.gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines import PPO2



num_agents = 4 #num of snakes
# placeholder_env necessary for model to recognize,
# the observation and action space, and the vectorized environment
placeholder_env = BattlesnakeEnv(n_threads=4, n_envs=16)
models = [PPO2(CustomPolicy, placeholder_env, verbose=1, learning_rate=1e-3) for _ in range(num_agents)]
# Close environment to free allocated resources
placeholder_env.close()

for _ in range(10):
    for model in models:
        env = BattlesnakeEnv(n_threads=4, n_envs=16, opponents=[ m for m in models if m is not model])
        model.set_env(env)
        # model.learn(total_timesteps=100000)
        model.learn(total_timesteps=10)
        env.close()

model = models[0]
env = BattlesnakeEnv(n_threads=1, n_envs=1, opponents=[ m for m in models if m is not model])
obs = env.reset()


for _ in range(1):
    action,_ = model.predict(obs)
    print(action, model.predict(obs))
    obs,_,_,_ = env.step(action)
    for o in obs:
        for row in o:
            print(row)

    print(obs.shape)
print(type(model))
    # r = env.render('rgb_array')

model.save("snake_model")
loaded_model = PPO2.load("snake_model")

action, _ = loaded_model.predict(obs)
print("loaded model", action)
