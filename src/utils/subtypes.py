import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.feature_selection import chi2
import scipy.stats as stats

def get_subtypes(clinical_data, fmri_data, motion_encoder_outputs, fmri_corresponding_outputs, person_samples_start_ends, include_clinical_subtypes=True):
    person_sample_nums = []
    for start, end in person_samples_start_ends:
        person_sample_nums.append(end-start)
    
    clinical_data_per_clip = clinical_data.loc[clinical_data.index.repeat(person_sample_nums)]

    # perform CCA
    X_mc = np.array(motion_encoder_outputs)
    Y_mc = np.array(fmri_corresponding_outputs)
    X_mc = (X_mc - X_mc.mean()) / (X_mc.std())
    Y_mc = (Y_mc - Y_mc.mean()) / (Y_mc.std())
    ca = CCA()
    ca.fit(X_mc, Y_mc)
    gait_c, fmri_c = ca.transform(X_mc, Y_mc)

    clinical_data_per_clip['CCX_1'] = gait_c[:, 0]
    clinical_data_per_clip['CCX_2'] = gait_c[:, 1]
    clinical_data_per_clip['CCY_1'] = fmri_c[:, 0]
    clinical_data_per_clip['CCY_2'] = fmri_c[:, 1]

    # Get subtypes from averaging embeddings of same person
    gait_c_reduced = []
    fmri_c_reduced = []

    for i, (start_index, end_index) in enumerate(person_samples_start_ends):
        person_gait_c = np.mean(gait_c[start_index:end_index], axis=0)
        person_fmri_c = np.mean(fmri_c[start_index:end_index], axis=0)
        
        gait_c_reduced.append(person_gait_c)
        fmri_c_reduced.append(person_fmri_c)
        
    gait_c_reduced = np.array(gait_c_reduced)
    fmri_c_reduced = np.array(fmri_c_reduced)

    cca_features = np.concatenate((gait_c_reduced, fmri_c_reduced), axis=1)
    kmeans = KMeans(n_clusters=3, random_state=2).fit(cca_features) # setting random state gives deterministic group assignment

    clinical_data_per_clip['Our subtype'] = np.repeat(kmeans.labels_, person_sample_nums)

    if include_clinical_subtypes:
        # Generate conventional, clinical variable-based subtypes
        # Disease duration, MDS-UPDRS Parts I, II, III, and IV, cognitive tests
        conventional_subtypes_data = clinical_data[['Disease duration', 'Part I_total', 'Cognitive_Impariment', 
                                                    'Hallucinations', 'Depression', 'Anxiety', 'Apathy', 
                                                    'Dopamine_Dysregulation', 'Insomnia', 'Daytime_sleepiness', 
                                                    'Pain', 'Urinary', 'Constipation', 'Lightheadedness', 'Fatigue',
                                                    'Part II_total', 'Speech', 'Saliva', 'Swallowing', 'Eating', 
                                                    'Dressing', 'Hygiene', 'Handwriting', 'Hobbies', 'Bed', 'Tremor', 
                                                    'Deep_Chair', 'Walking', 'Freezing',
                                                    'Part III_total', 'PIII_Speech', 'Expression', 'Rigid_neck', 
                                                    'Rigidity_RUE', 'Rigidity_LUE', 'Rigidity_RLE', 'Rigidity_LLE', 
                                                    'FingerTap_R', 'FingerTap_L', 'HandOpen_R', 'HandOpen_L', 'Pron_Sup_R', 
                                                    'Pron_Sup_L', 'Toe_Tap_R', 'Toe_Tap_L', 'Heal_Tap_R', 'Heal_Tap_L', 
                                                    'Standup', 'Gait.1', 'P_III_Freezing', 'Postur_Stab', 'Posture', 
                                                    'Body_Brady', 'Post_Trem_R', 'Post_Trem_L', 'Kinet_Trem_R', 
                                                    'Kinet_Trem_L', 'Rest_Trem_RUE', 'Rest_Trem_LUE', 'Rest_Trem_RLE', 
                                                    'Rest_Trem_LLE', 'Rest_Trem_head', 'Rest_Trem_time',
                                                    'Dyskinesiqs', 'Interfere_Rating', 'H_Y',
                                                    'Part IV_total', 'Time_Dyskin', 'Funct_Dyskin', 'Time_Off', 
                                                    'impact_Flux', 'Complex_Flux',
                                                    'Dystonia', 'MoCA', 'BVMTR_immediaterecall', 'BVMTR_delayedrecall', 
                                                    'CVLT_immediaterecall', 'CVLT_Shortt_delay_free', 'CVLT_Long_Delay_free', 
                                                    'CVLT_Long_Delay_cued', 'JLO', 'HVOT', 'SDMT_written', 'SDMT_oral', 
                                                    'FAS', 'Animals',
                                                    'Trails_A', 'Trails_B', 'Stroop_word', 'Stroop_Interference', 'BNT']].copy()
        conventional_subtypes_data.Dyskinesiqs.replace(('Yes', 'No'), (1, 0), inplace=True)
        conventional_subtypes_data.Interfere_Rating.replace(('Yes', 'No'), (1, 0), inplace=True)
        conventional_subtypes_array = conventional_subtypes_data.to_numpy()
        conventional_features_norm = stats.zscore(conventional_subtypes_array, nan_policy='omit')
        conventional_features = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(conventional_features_norm)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(conventional_features)

        clinical_data_per_clip['Clinical variable subtype'] = np.repeat(kmeans.labels_, person_sample_nums)

    return clinical_data_per_clip, cca_features

CATEGORICAL_SCORES = ['majority vote', 'Med status', 'isMale', 'Dyskinesiqs', 'Interfere_Rating',
                      'Gait', 'FingerTap_Right', 'FingerTap_Left', 
                      'Cognitive_Impariment', 'Hallucinations', 'Depression', 
                      'Anxiety', 'Apathy', 'Dopamine_Dysregulation', 'Insomnia', 
                      'Daytime_sleepiness', 'Pain', 'Urinary', 'Constipation', 'Lightheadedness', 
                      'Fatigue', 'Speech', 'Saliva', 'Swallowing', 
                      'Eating', 'Dressing', 'Hygiene', 'Handwriting', 'Hobbies', 'Bed', 
                      'Tremor', 'Deep_Chair', 'Walking', 'Freezing', 
                      'PIII_Speech', 'Expression', 'Rigid_neck', 'Rigidity_RUE', 'Rigidity_LUE', 
                      'Rigidity_RLE', 'Rigidity_LLE', 'FingerTap_R', 'FingerTap_L', 'HandOpen_R', 
                      'HandOpen_L', 'Pron_Sup_R', 'Pron_Sup_L', 'Toe_Tap_R', 'Toe_Tap_L', 
                      'Heal_Tap_R', 'Heal_Tap_L', 'Standup', 'Gait.1', 'P_III_Freezing', 
                      'Postur_Stab', 'Posture', 'Body_Brady', 'Post_Trem_R', 'Post_Trem_L', 
                      'Kinet_Trem_R', 'Kinet_Trem_L', 'Rest_Trem_RUE', 'Rest_Trem_LUE', 
                      'Rest_Trem_RLE', 'Rest_Trem_LLE', 'Rest_Trem_head', 'Rest_Trem_time',
                      'H_Y', 'Time_Dyskin', 'Funct_Dyskin', 'Time_Off', 'impact_Flux',
                      'Complex_Flux', 'Dystonia', 
                    ]

NON_NORMAL_CONTINUOUS_SCORES = ['Age', 'Onset age', 'Education', 'LEDD', 
                                'Disease duration',
                                'Part I_total', 'Part II_total', 'Part III_total', 
                                'Part IV_total', 'MoCA', 'BVMTR_immediaterecall',
                                'BVMTR_delayedrecall', 'CVLT_immediaterecall', 'CVLT_Shortt_delay_free', 
                                'CVLT_Long_Delay_free', 'CVLT_Long_Delay_cued', 'JLO', 'HVOT', 'SDMT_written', 
                                'SDMT_oral', 'FAS', 'Animals', 'Digit_forward', 'Digit_backwards',
                                'Trails_A', 'Trails_B', 'Stroop_word', 'Stroop_Interference', 'BNT', 
                                'TD', 'PIGD', 'TD/PIGD', 'Walking time', 'Walking speed', 'Avg torso angle'
                                ]

def process_value(value):
    if value == 'OFF' or value == 'No':
        value = 0
    elif value == 'ON' or value == 'Yes':
        value = 1
    return value

def find_subtype_clinical_characteristics(clinical_data, label='Our subtype'):
    labels = clinical_data[label]
    significant_clinical_variables = []
    significant_clinical_variables_p_values = []

    for (name, column) in clinical_data.iteritems():
        if not (name in CATEGORICAL_SCORES or name in NON_NORMAL_CONTINUOUS_SCORES):
            continue
        subtype_data = {}

        for i, subtype in enumerate(list(labels)):
            column_value = column[i]
            column_value = process_value(column_value)
            if subtype not in subtype_data:
                subtype_data[subtype] = []
            subtype_data[subtype].append(column_value)
            
        if name in NON_NORMAL_CONTINUOUS_SCORES:
            if all(x == column[0] for x in column):
                continue
            statistic, p_value = stats.kruskal(*subtype_data.values(), nan_policy='omit')
            if p_value < 0.05:
                significant_clinical_variables.append(name)
                significant_clinical_variables_p_values.append(p_value)
            
        if name in CATEGORICAL_SCORES:
            column_values = np.array([process_value(value) for value in list(column)])
            keep_indices = np.logical_not(np.isnan(column_values))
            column_values_filtered = column_values[keep_indices]
            labels_filtered = np.array(labels)[keep_indices]
            chi2_statistic, p_value = chi2(column_values_filtered.reshape(-1, 1), labels_filtered)
            if p_value < 0.05:
                significant_clinical_variables.append(name)
                significant_clinical_variables_p_values.append(p_value[0])
                
    return significant_clinical_variables, significant_clinical_variables_p_values