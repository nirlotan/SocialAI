import numpy as np
import sklearn.metrics as skm
import tensorflow as tf

def fix_list(row):
    if row['source'] == 'volkova':
        return "[" + row['list'] + "]"
    else:
        return row['list']

def get_num( local_df, ground_truth, pred ):
    return local_df[ (local_df['political_enum']==ground_truth)&(local_df['preds']==pred)].shape[0]

def recall_sub( subset_df , pos_label):
    return skm.recall_score( subset_df['political_enum'].values.tolist(),
                             subset_df['preds'].values.tolist(),
                             pos_label = pos_label)

def precision_sub( subset_df , pos_label):
    return skm.precision_score( subset_df['political_enum'].values.tolist(),
                             subset_df['preds'].values.tolist(),
                             pos_label = pos_label)

def prep_tf_inputs(df_subset, field):
    values = np.stack(df_subset[field].to_numpy())
    X = tf.constant(values)
    y = df_subset['class'].values
    return X,y
