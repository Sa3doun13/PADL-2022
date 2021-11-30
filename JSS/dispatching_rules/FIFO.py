import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import math
from scipy.spatial import distance
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import spearmanr
from JSS.dispatching_rules.JSSEnv import JssEnv
from JSS.dispatching_rules.MTWR import MTWR_worker
from config import default_config
from sklearn.decomposition import PCA



def FIFO_worker(default_config):
    wandb.init(config=default_config, name="FIFO")
    config = wandb.config
    env = JssEnv({'instance_path': config['instance_path']})
    done = False
    state = env.reset()
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))
        remaining_time = reshaped[:, 5]
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * -1e8
        remaining_time += mask
        FIFO_action = np.argmax(remaining_time)
        assert legal_actions[FIFO_action]
        time_before_action = env.current_time_step
        state, reward, done, _ = env.step(FIFO_action)
        time_after_action = env.current_time_step

    #print(sum(env.solution[:, 0] == 0))
    assignment = env.solution
    env.reset()
    make_span = env.last_time_step
    print(f"FIFO make span {make_span}")
    wandb.log({"nb_episodes": 1, "make_span": make_span})
    return assignment

def starting_time(solution_FIFO, solution_MTWR, solution_EST, features, jobs, operations):
    count = 0
    for i in range(jobs):
        for j in range(operations):
            features[count][0] = i + 1
            features[count][1] = j + 1
            features[count][2] = solution_FIFO[i][j]
            features[count][3] = solution_MTWR[i][j]
            features[count][4] = solution_EST[i][j]
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
            features[feature_index][8] = int(job[count])
            machine_load[int(job[count-1])] += int(job[count])
            count += 2
            feature_index += 1
        job_length[job_index] = job_pro_time
        feature_index -= operations
        # add the remaining time of operations belong to same job (remaining time attribute)
        for i in range(operations):
            features[feature_index][9] = job_pro_time - features[feature_index][8]
            job_pro_time -= features[feature_index][8]
            feature_index += 1
        job_index += 1
    return features, machine_load

def waiting_time(features, jobs, operations):
    # go through the operations and calculate the waiting time
    for i in range(operations * jobs):
        if features[i][1] != 1:
            features[i][5] = features[i][2] - features[i-1][2] - features[i-1][8]
            features[i][6] = features[i][3] - features[i-1][3] - features[i-1][8]
            features[i][7] = features[i][4] - features[i-1][4] - features[i-1][8]
    return features

def earliest_start_time(features, jobs, operations):
    for i in range(operations * jobs):
        if features[i][1] == 1:
            features[i][10] = 0
        else:
            features[i][10] = features[i-1][8] + features[i-1][10]
    return features

def job_time_length(features, jobs, operations):
    i = 0
    job_index = 0
    # calculate the length of each job and store the values in the feature matrix
    while(job_index < jobs * operations):
        count = 0
        while(count < operations):
            features[job_index][8] = features[i][5] + features[i][4]
            count += 1
            job_index += 1
        i += operations
    return features

def machine_loading(features, operations, jobs, data, machine_load):
    est_array = np.zeros(len(features[:,0]))
    counter = 0
    # storing the EST values of the operations in an array called est_array
    for feature_index in features:
        est_array[counter] = feature_index[10]
        counter += 1
    # this is to go through the whole data points
    for i in range(operations * jobs):
        # getting the index of the minimum EST
        min_index = np.argmin(est_array)
        est_array[min_index] = 100**100
        # catching the job and operation number using operation which has the minimum EST
        job = features[min_index][0]
        operation = features[min_index][1]
        # use data matrix to catch which machine is assigned to the operation that has minimum EST
        job_row = data[job - 1]
        job_row = job_row.split()
        assigned_machine = int(job_row[2*(operation-1)])
        # store the load of machine which is assigned to the current operation in the feature matrix
        features[min_index][11] = machine_load[assigned_machine]
        # update the load of machine, by subtracting the processing time of the current operation from the machine load
        machine_load[assigned_machine] -= features[min_index][8]

    return features

def pre_processing(features, counter):

    # selected features [ST_FIFO, ST_MTWR, ST_EST, WT_FIFO, WT_MTWR, WT_EST, Remaning_Pro_Time, EST, ML]
    main_list = [[2, 3, 4], [2, 3, 4, 9], [2, 3, 4, 10], [2, 3, 4, 11],
                 [2, 3, 4, 9, 10], [2, 3, 4, 9, 11], [2, 3, 4, 10, 11], [2, 3, 4, 9, 10, 11],
                 [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 9], [2, 3, 4, 5, 6, 7, 10], [2, 3, 4, 5, 6, 7, 11],
                 [2, 3, 4, 5, 6, 7, 9, 10], [2, 3, 4, 5, 6, 7, 9, 11], [2, 3, 4, 5, 6, 7, 10, 11], [2, 3, 4, 5, 6, 7, 9, 10, 11]]
    selected_features = features[:, main_list[counter]]
    if counter >= 7:
        selected_features[:, 3] *= (-1)
        selected_features[:, 4] *= (-1)
        selected_features[:, 5] *= (-1)
    scaler = StandardScaler()
    scaled_selected_features = scaler.fit_transform(selected_features)
    return scaled_selected_features, main_list

def generate_centroids(jobs, operations, scaled_selected_features, features, n_clus):
    # these two lists contain the job_numbers and operation_numbers selected as centroids
    #temp_features = features[:, 2:]
    random_list_jindex = []
    random_list_oindex = []
    assignment = np.zeros((jobs * operations, 3), dtype=int)
    centroids = []
    LB_Job = 1
    LB_Ope = 1
    job_point_fact = int(jobs / n_clus)
    ope_point_fact = int(operations / n_clus)
    # this loop is to generate the (job/operation)_numbers to be the centroids
    for i in range(n_clus):
        UB_Job = (i+1) * job_point_fact
        UB_Ope = (i+1) * ope_point_fact
        random_list_jindex.append(random.randint(LB_Job, UB_Job))
        random_list_oindex.append(random.randint(LB_Ope, UB_Ope))
        LB_Job = UB_Job
        LB_Ope = UB_Ope
    # this loop is to access the right position of the selected centroid and in which cluster each centroid is assigned
    for c in range(n_clus):
        target_index = (random_list_jindex[c] * operations) - operations + random_list_oindex[c] - 1
        centroids.append(scaled_selected_features[target_index])
        assignment[target_index][0] = features[target_index][0]
        assignment[target_index][1] = features[target_index][1]
        assignment[target_index][2] = c+1
    return centroids, assignment

def const_kMeans(jobs, operations, features, scaled_selected_features, n_clus, centroids, assignment):
    max_size = math.floor(jobs * operations / n_clus)
    max_clus = np.ones(n_clus)
    max_clus *= max_size
    count = np.zeros(n_clus)
    for index in range(jobs * operations):
        min_distance = 1000000000
        assigned_clust = 0
        # calculate the Euclidean distance between a point and each centroid
        for c in range(n_clus):
            if count[c] >= max_clus[c]-1:
                continue
            a = scaled_selected_features[index]
            b = centroids[c]
            dist = distance.euclidean(a, b)
            if dist < min_distance:
                min_distance = dist
                assigned_clust = c + 1
        # to avoid assigning a successor into an earlier cluster
        if (index > 0) and (features[index-1][0] == features[index][0]):
            if assignment[index - 1][2] > assigned_clust:
                assigned_clust = assignment[index - 1][2]
        assignment[index][0] = features[index][0]
        assignment[index][1] = features[index][1]
        assignment[index][2] = assigned_clust
        count[assigned_clust - 1] += 1
        # this loop is to update the centroids after assigning a new data into a cluster
        for f in range(len(scaled_selected_features[index])):
            centroids[assigned_clust - 1][f] = ((centroids[assigned_clust - 1][f]) + (scaled_selected_features[index][f])) / 2
    return assignment

def pars_sol(jobs, operations):
    solution_EST = np.zeros((jobs, operations), dtype=int)
    f = open('C:\\Users\\mohammed\\SeafIle\\Seafile\\My Library\\Research papers\\Benchmark problems\\JSP Encoding\\Multi-shot\\EST_TA60_solution.lp', 'r')
    main_text = f.read()
    f.close()
    main_text = main_text.split('\n')
    for text in main_text:
        if text != "":
            text = text.split("startTime((")
            text = text[1]
            text = text.split(")")
            operation = text[0]
            operation = operation.split(',')
            text = text[1]
            text = text.split(",")
            AST = text[1]
            job = int(operation[0]) - 1
            ope = int(operation[1]) - 1
            solution_EST[job][ope] = int(AST)
    return solution_EST

def write_in_file(jobs, operations, features, cluster_assignment, counter):
    clust = 'TA60_Clus({}).lp'.format(counter+1)
    path = 'C:\\Users\\mohammed\\SeafIle\\Seafile\\My Library\\Research papers\\Benchmark problems\\JSP Encoding\\Multi-shot\\' + clust
    f = open(path, 'a')
    for i in range(operations * jobs):
        f.writelines('assignToTimeWindow({}, {}, {}).'.format(features[i][0], features[i][1], cluster_assignment[i][2]))
        f.write('\n')
    f.close()

def pair_plotting(scaled_selected_features):
    feature_name = ['ST_FIFO', 'ST_MTWR', 'ST_EST', 'WT_FIFO', 'WT_MTWR', 'WT_EST', 'Remaning_Pro_Time', 'EST', 'ML']
    df = pd.DataFrame(scaled_selected_features, columns=['ST_FIFO', 'ST_MTWR', 'ST_EST', 'WT_FIFO', 'WT_MTWR', 'WT_EST', 'Remaning_Pro_Time', 'EST', 'ML'])
    for i in range(9):
        j = i + 1
        while(j < 9):
            x = scaled_selected_features[:, i]
            y = scaled_selected_features[:, j]
            corr, _ = spearmanr(x, y)
            #print('Completion Time for Window {} : {} '.format(x + 1, makespan_time_window[x]))
            print('The correlation between {}, {} is {}'.format(feature_name[i], feature_name[j], np.corrcoef(x, y)))
            print('The Spearman correlation between {}, {} is {}'.format(feature_name[i], feature_name[j], corr))
            print('*******************************************************************************')
            j += 1
    #sns.pairplot(df)
    #plt.show()

def pca_plotting(scaled_selected_features, assignment, main_list, counter):
    features = ['Job', 'Oper', 'ST_FIFO', 'ST_MTWR', 'ST_EST', 'WT_FIFO', 'WT_MTWR', 'WT_EST', 'Pro_Time', 'Remaning_Time', 'EST', 'ML']
    Title = ' '
    num_selected_feat = len(main_list[counter])
    for i in range(num_selected_feat):
        Title += features[main_list[counter][i]] + ' | '
    # Reducing the features to two
    pca_schedule = PCA(n_components=2)
    principalComponents_schedule = pca_schedule.fit_transform(scaled_selected_features)
    principal_schedule_Df = pd.DataFrame(data=principalComponents_schedule
                                       , columns=['principal component 1', 'principal component 2'])
    # This is to show how much information is lost
    print('Explained variation per principal component: {}'.format(pca_schedule.explained_variance_ratio_))
    # This is to add the column of assignment to show which data point assigned to which cluster
    principal_schedule_Df.insert(2, 'Cluster', assignment[:, 2], True)
    # This is to plot the data point, color is changed based on the cluster assignment
    sns.scatterplot(data=principal_schedule_Df, x='principal component 1', y='principal component 2', hue='Cluster').set(title=Title)
    plt.show()

if __name__ == "__main__":
    # features (Job, Oper, ST_FIFO, ST_MTWR, ST_EST, WT_FIFO, WT_MTWR, WT_EST, Pro_Time, Remaning_Time, EST, ML)
    n_clus = 3
    counter = 0
    instance_path = '../instances/ta60'
    solution_FIFO = FIFO_worker(default_config)
    solution_MTWR = MTWR_worker(default_config)

    mat_size_FIFO = solution_FIFO.shape
    operations = mat_size_FIFO[1]
    jobs = mat_size_FIFO[0]
    solution_EST = pars_sol(jobs, operations)
    features = np.zeros((jobs * operations, 12), dtype = int)
    # I have the actual starting time fo each operation
    features = starting_time(solution_FIFO, solution_MTWR, solution_EST, features, jobs, operations)
    #features = starting_time(solution_FIFO, solution_MTWR, features, jobs, operations)
    #features = sequence_position(features, jobs, operations)
    with open(instance_path, 'r') as f:
        data = f.read()
        data = data.split("\n")
        data = data[1: -1]
        f.close()
    features, machine_load = data_extraction(features, jobs, operations, data)
    features = waiting_time(features, jobs, operations)
    features = earliest_start_time(features, jobs, operations)
    #features = job_time_length(features, jobs, operations)
    features = machine_loading(features, operations, jobs, data, machine_load)
    #for i in range(jobs * operations):
    #    print(features[i])

    while(counter < 16):
        scaled_selected_features, main_list = pre_processing(features, counter)
        #pair_plotting(scaled_selected_features)
        centroids, assignment = generate_centroids(jobs, operations, scaled_selected_features, features, n_clus)
        cluster_assignment = const_kMeans(jobs, operations, features, scaled_selected_features, n_clus, centroids, assignment)
        write_in_file(jobs, operations, features, cluster_assignment, counter)
        #pca_plotting(scaled_selected_features, assignment, main_list, counter)
        counter += 1
        #break





