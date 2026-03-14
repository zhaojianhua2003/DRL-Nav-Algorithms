from tqdm import tqdm
import numpy as np
import torch
import collections
import random


def evaluate(env,agent, iteration, max_steps=500,eval_episodes=10):
    avg_reward = 0.0
    col = 0
    success = 0
    print("..............................................")
    print("validating")
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < max_steps+1:
            action = agent.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
            if reward > 90:
                success += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes*100
    avg_succ = success / eval_episodes*100
    
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward:%f, avg_collision: %d%% ,avg_Success: %d%%"
        % (eval_episodes, iteration, avg_reward, avg_col,avg_succ)
    )
    return avg_reward

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


#iter_episodes: how many episodes in an iteration
#max_steps:max env.reset()
def train_on_policy_agent(env, agent, iter_episodes,max_steps,iterations):
    return_list = []
    evaluations=[]
    for i in range(iterations):
        with tqdm(total=iter_episodes, desc='Iteration %d' % (i+1)) as pbar:
            for i_episode in range(iter_episodes):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                time_step=0
                while not done and time_step<max_steps:
                    action = agent.get_action(np.array(state))
                    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                    a_in = [(action[0] + 1) / 2, action[1]]
                    next_state, reward, done, _ = env.step(a_in)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    time_step+=1
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    current_ep = iter_episodes * i + i_episode + 1
                    avg_return = np.mean(return_list[-10:])
                    pbar.write(f"Episode:{int(current_ep)-10} ~ {int(current_ep)} | Return: {avg_return:.3f}")
                pbar.update(1)

        evaluations.append(
                evaluate(env=env,agent=agent,max_steps=max_steps,iteration=i)
            )
        agent.save()
        np.save("./results/%s" % (agent.filename), evaluations)
        print("model saved!")
        print("..............................................")
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(discount, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = discount * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                