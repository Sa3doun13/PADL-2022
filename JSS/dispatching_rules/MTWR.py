import random
import wandb
import gym
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from JSS.dispatching_rules.JSSEnv import JssEnv
from config import default_config

kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=1000,
    random_state=42
)

def MTWR_worker(default_config):
    wandb.init(config=default_config, name="MTWR")
    config = wandb.config
    env = JssEnv({'instance_path': config['instance_path']})
    done = False
    state = env.reset()
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))
        #
        remaining_time = reshaped[:, 3] / env.jobs_length
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * -1e8
        remaining_time += mask
        MTWR_action = np.argmax(remaining_time)
        assert legal_actions[MTWR_action]
        #
        state, reward, done, _ = env.step(MTWR_action)
    #print(env.solution)
    assignment = env.solution
    env.reset()
    make_span = env.last_time_step
    print(f"MTWR make span {make_span}")
    wandb.log({"nb_episodes": 1, "make_span": make_span})
    return assignment

def starting_time(solution, features, jobs, operations):
    count = 0
    for i in range(jobs):
        for j in range(operations):
            features[count][0] = i + 1
            features[count][1] = j + 1
            features[count][2] = solution[i][j]
            count += 1
    return features

def sequence_position(features, jobs, operations):
    length = jobs * operations
    count = 1
    temp_array = np.zeros(jobs * operations)
    for i in range(length):
        temp_array[i] = features[i, 2]

    for i in range(length):
        index = np.argmin(temp_array)
        features[index][3] = count
        temp_array[index] = 100000
        count += 1
    return features

def data_extraction(features, jobs, operations, data):
    machine_load = np.zeros(operations)
    job_length = np.zeros(jobs)
    job_index = 0
    feature_index = 0
    for job in data:
        count = 1
        job_pro_time = 0
        job = job.split()
        # add the processing of operations belong to same job (processing time attribute)
        while count < operations * 2:
            job_pro_time += int(job[count])
            features[feature_index][4] = int(job[count])
            machine_load[int(job[count-1])] += int(job[count])
            count += 2
            feature_index += 1
        job_length[job_index] = job_pro_time
        feature_index -= operations
        # add the remaining time of operations belong to same job (remaining time attribute)
        for i in range(operations):
            features[feature_index][5] = job_pro_time - features[feature_index][4]
            job_pro_time -= features[feature_index][4]
            feature_index += 1
        job_index += 1
    return features, machine_load

def waiting_time(features, jobs, operations):
    # go through the operations and calculate the waiting time
    for i in range(operations * jobs):
        if features[i][1] != 1:
            features[i][6] = features[i][2] - features[i-1][2] - features[i-1][4]
    return features

def earliest_start_time(features, jobs, operations):
    for i in range(operations * jobs):
        if features[i][1] == 1:
            features[i][7] = 0
        else:
            features[i][7] = features[i-1][4] + features[i-1][7]
    return features

def job_time_length(features, jobs, operations):
    time_job = np.zeros(jobs)
    i = 0
    job_index = 0
    # calculate the length of each job and store the values in time_job array
    while(i < jobs * operations):
        time_job[job_index] = features[i][5] + features[i][4]
        i += operations
        job_index += 1
    first_quartile = np.percentile(time_job, 25)
    second_quartile = np.percentile(time_job, 50)
    third_quartile = np.percentile(time_job, 75)
    i = 0
    # classify the jobs into four categories based on the time length
    for job_index in range(jobs):
        count = 0
        if time_job[job_index] <= first_quartile:
            while(i < jobs * operations and count < operations):
                features[i][8] = 1
                i += 1
                count += 1
        elif time_job[job_index] <= second_quartile:
            while(i < jobs * operations and count < operations):
                features[i][8] = 2
                i += 1
                count += 1
        elif time_job[job_index] <= third_quartile:
            while(i < jobs * operations and count < operations):
                features[i][8] = 3
                i += 1
                count += 1
        elif time_job[job_index] > third_quartile:
            while (i < jobs * operations and count < operations):
                features[i][8] = 4
                i += 1
                count += 1
    return features

def machine_loading(features, operations, data, machine_load):
    feature_index = 0
    median = np.percentile(machine_load, 50)
    for job in data:
        count = 1
        job = job.split()
        while count < operations * 2:
            # access the index of the machine from tuple job and check whether the machine is heavy or not
            if machine_load[int(job[count-1])] <= median:
                features[feature_index][9] = 1
            elif machine_load[int(job[count-1])] > median:
                features[feature_index][9] = 2
            count += 2
            feature_index += 1
    return features

if __name__ == "__main__":
    instance_path = '../instances/ta60'
    solution = MTWR_worker(default_config)
    mat_size = solution.shape
    operations = mat_size[1]
    jobs = mat_size[0]
    features = np.zeros((jobs * operations, 10), dtype = int)
    # I have the actual starting time fo each operation
    features = starting_time(solution, features, jobs, operations)
    features = sequence_position(features, jobs, operations)
    with open(instance_path, 'r') as f:
        data = f.read()
        data = data.split("\n")
        data = data[1: -1]
        f.close()
    features, machine_load = data_extraction(features, jobs, operations, data)
    features = waiting_time(features, jobs, operations)
    features = earliest_start_time(features, jobs, operations)
    features = job_time_length(features, jobs, operations)
    features = machine_loading(features, operations, data, machine_load)
    #print(features[:5])
