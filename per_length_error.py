import pickle 
import numpy as np

if __name__ == '__main__':
    # path = 'results/fcn_imagenette_seqN_npN_rep.pk'
    # path = 'results/fcn_imagenette_seqN_np1_rep.pk'
    # path = 'results/fcn_gdl_seqN_npN_rep.pk'
    # path = 'results/fcn_gdl_seqN_np1_rep.pk'

    path = 'results/rnn_imagenette_seqN_npN_rep.pk'
    # path = 'results/rnn_imagenette_seqN_np1_rep.pk'
    # path = 'results/rnn_gdl_seqN_npN_rep.pk'
    # path = 'results/rnn_gdl_seqN_np1_rep.pk'



    # path = 'results/fcn_gdl_seqN_npN_rep.pk'
    # path = 'results/fcn_imagenette_seqN_npN.pk'
    # path = 'results/fcn_gdl_seqN_npN.pk'
    data = pickle.load(open(path, 'rb'))
    
    res = np.zeros(15)
    counts = np.zeros(15)

    for yt, yp, ter in zip(data['y_true'], data['y_pred'], data['ter']):
        id = np.sum(yt!=yt[-1])
        res[id] += ter
        counts[id] += 1
    print(res/counts)

