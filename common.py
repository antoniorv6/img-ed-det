import pyter
import numpy as np

def get_name(head='fcn', n_samples='single', seq_len='single', dataset='imagenette', repetitions=True):
    return f"{head}_{dataset}_seq{'1' if seq_len=='single' else ('N' if seq_len=='multiple' else seq_len)}_np{'1' if n_samples=='single' else 'N'}{'_rep' if repetitions else ''}"

def load_model(**params):
    import tensorflow as tf
    return tf.keras.models.load_model(get_name(**params))

def iou(y_true, y_pred):
    intersection = np.intersect1d(y_true, y_pred)
    union = np.union1d(y_true, y_pred)
    return intersection.shape[-1] / union.shape[-1]

def ter(y_true, y_pred):
    end_token = y_true[-1]
    if end_token in y_pred:
        index = np.argwhere(y_pred == end_token)
        if len(index) != 0:
            i = index[0][0]+1
            y_pred = y_pred[:i]

        y_pred = np.append(y_pred, end_token)


    y_true = y_true[y_true!=end_token]
    y_true = np.append(y_true, end_token)

    return pyter.ter(y_pred, y_true)

def seq_log_prob(y_true, y_probs):
    print(y_probs[np.arange(y_probs.shape[0]), y_true])
    return np.sum(np.log(np.maximum(y_probs[np.arange(y_probs.shape[0]), y_true], 1e-8)))

def seq_prob(y_true, y_probs):
    return np.prod(y_probs[np.arange(y_probs.shape[0]), y_true])

def top_n(y_true, y_probs, n=1000):
    import tensorflow as tf
    end_token = y_true[-1]
    y_true = y_true[y_true!=end_token]
    y_true = np.concatenate([y_true, [end_token]])

    eps = np.zeros((y_probs.shape[1]+1))
    eps[-1] = 1
    inputs = y_probs
    inputs =  np.concatenate([inputs, np.zeros((inputs.shape[0], 1))], axis=1)
    inputs =  np.concatenate([inputs, np.tile(eps, [inputs.shape[0], 1])], axis=1).reshape((-1, inputs.shape[1]))
    inputs = np.log(inputs).reshape((inputs.shape[0], 1, inputs.shape[1]))
    # print(inputs)


    try:
        seq, lp = tf.nn.ctc_beam_search_decoder(
            np.transpose(inputs, [0, 1, 2]), [inputs.shape[0]], beam_width=n, top_paths=n,
        )
        sequences = np.stack([s.values for s in seq], axis=0)
        return int(np.any(np.all(sequences[:, :y_true.shape[0]] == y_true, axis=1)))
    except:
        k = n
        while k > 1:
            try:
                seq, lp = tf.nn.ctc_beam_search_decoder(
                    np.transpose(inputs, [0, 1, 2]), [inputs.shape[0]], beam_width=k, top_paths=k,
                )
                sequences = np.stack([s.values for s in seq], axis=0)
                return int(np.any(np.all(sequences[:, :y_true.shape[0]] == y_true, axis=1)))
            except:
                pass
            k = k // 2
    return 0
    

def ter(y_true, y_pred):
    end_token = y_true[-1]
    if end_token in y_pred:
        index = np.argwhere(y_pred == end_token)
        if len(index) != 0:
            i = index[0][0]+1
            y_pred = y_pred[:i]

        y_pred = np.append(y_pred, end_token)


    y_true = y_true[y_true!=end_token]
    y_true = np.append(y_true, end_token)

    return pyter.ter(y_pred, y_true)
    
    
#    [       nan 6.35932722 4.4017744  3.18333333 2.43333333 1.98531073
# 1.62767546 1.38027597 1.19358534 1.04836601 0.92824567 0.83278509
# 0.76186579 0.69662058 0.63501199]