import os
import copy
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.stats as stats

def add_majority_vote_gait_score_column(df):
    majority_votes = []
    for _, row in df.iterrows():
        scores = (row['Score 1'], row['Score 2'], row['Score 3'])
        majority_map = {0: 0, 1: 0, 2: 0, 3: 0}
        majority = (0, 0)
        for score in scores:
            if score == 4: score = 3 # merging 3 and 4
            majority_map[score] += 1
            
            if majority_map[score] > majority[1]:
                majority = (score, majority_map[score])
        if scores == (3, 2, 1): # pre-defined tie-breaker
            majority_votes.append(3)
        else:
            majority_votes.append(majority[0])
    df['majority vote'] = majority_votes

def load_clinical_data(clinical_data_path, gait_scores_path):
    clinical_data = pd.read_csv(clinical_data_path)
    if 'TD/PIGD subtype' not in clinical_data.columns:
        # add TD and PIGD
        tremor_columns = ['Tremor', 'Post_Trem_R', 'Post_Trem_L', 'Kinet_Trem_R', 'Kinet_Trem_L', 'Rest_Trem_RUE', 'Rest_Trem_LUE', 'Rest_Trem_RLE', 'Rest_Trem_LLE', 'Rest_Trem_head', 'Rest_Trem_time']
        pigd_columns = ['Walking', 'Freezing', 'Gait', 'P_III_Freezing', 'Postur_Stab']
        clinical_data['TD'] = clinical_data[tremor_columns].mean(axis=1)
        clinical_data['PIGD'] = clinical_data[pigd_columns].mean(axis=1)
        clinical_data['TD/PIGD'] = clinical_data[tremor_columns].mean(axis=1) / clinical_data[pigd_columns].mean(axis=1)
        # binary subtype
        td_pigd_subtype = []
        for _, row in clinical_data.iterrows():
            if row['TD/PIGD'] <= 0.9:
                td_pigd_subtype.append('PIGD')
            elif row['TD/PIGD'] < 1.15:
                td_pigd_subtype.append('indeterminate')
            else:
                td_pigd_subtype.append('TD')

        clinical_data['TD/PIGD subtype'] = td_pigd_subtype
    if 'MoCA' in clinical_data.columns:
        # process MoCA 0s as NaN
        clinical_data['MoCA'] = clinical_data['MoCA'].replace(0, np.nan)

    gait_scores = pd.read_csv(gait_scores_path)
    add_majority_vote_gait_score_column(gait_scores)

    gait_scores = gait_scores.sort_values(by=['PID'])

    clinical_data = gait_scores.merge(clinical_data, on='PID')
    return clinical_data

def load_motion_encoder_data(motion_encoder_dir):
    motion_encoder_data = {}

    for pid in os.listdir(motion_encoder_dir):
        sample_outputs = []
        id_motion_encoder_dir = os.path.join(motion_encoder_dir, pid)
        for batch in os.listdir(id_motion_encoder_dir): # multiple smaller videos processed for each video
            clip_file_path = os.path.join(id_motion_encoder_dir, batch)
            clip_outputs = np.load(clip_file_path)
            clip_outputs = np.squeeze(clip_outputs)
            clip_outputs = np.mean(clip_outputs, axis=0) # take mean across frames
            sample_outputs.append(clip_outputs)
        motion_encoder_data[pid] = sample_outputs.copy()

    return motion_encoder_data

def load_fmri_data(fc_matrices_dir):
    fmri = {}
    for pid_path in os.listdir(fc_matrices_dir):
        path = os.path.join(fc_matrices_dir, pid_path)
        matrix = sio.loadmat(path)
        matrix_array = matrix['arr'][:, 0:165] # Last two columns should be spliced off
        matrix_array = np.nan_to_num(matrix_array)
        matrix_lower_flattened = matrix_array[np.tril_indices(165)]
        fmri[pid_path.split('.')[0]] = matrix_lower_flattened
    return fmri

def load_toy_fmri_data(fc_data_dir):
    fmri = {}
    for pid_path in os.listdir(fc_data_dir):
        path = os.path.join(fc_data_dir, pid_path)
        fmri_data = np.load(path)
        fmri[pid_path.split('.')[0]] = fmri_data
    return fmri

def reduce_fmri_data(fmri_data, clinical_data):
    fmri_data_reduced = {}

    # select indices with greatest correlation to gait impairment
    fmri_vector = []
    gait_scores = []
    significant_indices = []
    for pid, data in fmri_data.items():
        row = clinical_data.loc[clinical_data['PID'] == pid]
        if len(row) == 0:
            print(pid, 'not in clinical data.')
            continue
        assert len(row) == 1
        row = row.iloc[0]
        gait_score = row['majority vote']
        fmri_vector.append(data)
        gait_scores.append(gait_score)
    fmri_np = np.array(fmri_vector)
    for i in range(np.shape(fmri_np)[1]): # iterate through every adjacency index
        adjacency_across_scores = fmri_np[:,i]
        correlation, p_value = stats.pearsonr(adjacency_across_scores, gait_scores)
        if p_value < 0.001:
            significant_indices.append(i)

    for pid, data in fmri_data.items():
        fmri_data_reduced[pid] = data[significant_indices]
    return fmri_data_reduced

CEREBELUM_INDICES = list(range(107, 125)) + list(range(163, 165))
SENSORIMOTOR_INDICES = list(range(137, 140))
PREFRONTAL_INDICES = [49, 60, 61, 76, 77]
PALLIDAL_INDICES = [98, 99]
CEREBELLAR_PREFRONTAL_TOP5 = [(107, 61), (115, 61), (107, 60), (164, 61), (116, 61)]
def select_subnetwork(fmri_data, clinical_data, subnetwork):
    fmri_data_reduced = {}

    assert subnetwork in ['pallidal-sensorimotor', 'cerebellar-motor', 'cerebellar-prefrontal', 'top-cerebellar-prefrontal', 'top-whole-brain']
    if subnetwork == 'top-whole-brain':
        return reduce_fmri_data(fmri_data, clinical_data)
    elif subnetwork == 'pallidal-sensorimotor':
        x_indices, y_indices = PALLIDAL_INDICES, SENSORIMOTOR_INDICES
    elif subnetwork == 'cerebellar-motor':
        x_indices, y_indices = CEREBELUM_INDICES, SENSORIMOTOR_INDICES
    elif subnetwork == 'cerebellar-prefrontal':
        x_indices, y_indices = CEREBELUM_INDICES, PREFRONTAL_INDICES
    
    # get indices connecting two subnetworks
    network_indices = []
    roi_grid = np.empty((165, 165), dtype=object)
    for i in range(np.shape(roi_grid)[0]):
        for j in range(np.shape(roi_grid)[0]):
            roi_grid[i][j] = (i, j)
    roi_flattened = roi_grid[np.tril_indices(165)]

    if subnetwork != 'top-cerebellar-prefrontal':
        for i in x_indices:
            for j in y_indices:
                flattened_index = [index for index, (x, y) in enumerate(roi_flattened) if (x, y) == (max(i, j), min(i, j))][0]
                network_indices.append(flattened_index)
    else:
        network_indices = []
        for i, j in CEREBELLAR_PREFRONTAL_TOP5:
            flattened_index = [index for index, (x, y) in enumerate(roi_flattened) if (x, y) == (max(i, j), min(i, j))][0]
            network_indices.append(flattened_index)

    for pid, data in fmri_data.items():
        fmri_data_reduced[pid] = data[network_indices]
    return fmri_data_reduced

def load_data(clinical_data_path, gait_scores_path, motion_encoder_dir, fc_matrices_dir, toy=False, missing_pids=['U218_ST022_OFF']):
    clinical_data = load_clinical_data(clinical_data_path, gait_scores_path)
    motion_encoder_data = load_motion_encoder_data(motion_encoder_dir)

    if not toy:
        fmri_data = load_fmri_data(fc_matrices_dir)
        fmri_data = reduce_fmri_data(fmri_data, clinical_data)
    else:
        fmri_data = load_toy_fmri_data(fc_matrices_dir)

    clinical_data = clinical_data[~clinical_data['PID'].isin(missing_pids)].reset_index(drop=True) # no clinical data for this example

    motion_encoder_outputs = [] # multiple clips per person
    fmri_corresponding_outputs = [] # save corresponding fmri outputs
    person_samples_start_ends = [] # keep track of examples per person
    person_curr = 0

    for pid in clinical_data['PID']:
        example_motion_encoder_outputs = motion_encoder_data[str(pid)]
        for sample_index in range(np.shape(example_motion_encoder_outputs)[0]):
            motion_encoder_outputs.append(example_motion_encoder_outputs[sample_index])
            fmri_corresponding_outputs.append(fmri_data[str(pid)])

        person_samples_start_ends.append((person_curr, person_curr+ np.shape(example_motion_encoder_outputs)[0]))
        person_curr += np.shape(example_motion_encoder_outputs)[0]

    return clinical_data, fmri_data, motion_encoder_outputs, fmri_corresponding_outputs, person_samples_start_ends




