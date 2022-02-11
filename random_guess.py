import fire
from data import DataGenerator
import progressbar
import numpy as np
import common

def eval(seq_len='single', repetitions=True):
    datagen_params = DataGenerator.params_from_experiment(n_samples='single', seq_len=seq_len, dataset='imagenette', repetitions=repetitions)

    test_gen = DataGenerator(split='test', shuffle=True, **datagen_params)

    # for ds in ['imagenette', 'gdl']:
    #     for s in ['train', 'test', 'val']:
    #         params = datagen_params.copy()
    #         params['dataset'] = ds
    #         gen = DataGenerator(split=s, shuffle=True, **params)
    #         print(ds, s, gen.get_size())
    # quit()
    test_gen = DataGenerator(split='test', shuffle=True, **datagen_params)

    print(test_gen.get_random_edit_sequence())

    y_trues = []
    y_preds = []
    ious = []
    ters = []

    res = np.zeros(15)
    counts = np.zeros(15)
    for i in progressbar.progressbar(range(test_gen.get_size())):
        #if i > 10:
        #    break
        label = test_gen.get_random_edit_sequence()
        label = np.append(label, len(test_gen.distortions))
        # pred = test_gen.get_random_edit_sequence(include_end_token=False)
        pred = np.random.randint(0, len(test_gen.distortions), test_gen.max_n_distortions+1)
        clean_label = label[label!=label[-1]]
        if np.all(pred[:len(clean_label)] == clean_label):
            pred[len(clean_label):] = label[-1]

        y_trues.append(label)
        y_preds.append(pred)
        # print(label, pred)
        ious.append(common.iou(label, pred))
        ter = common.ter(label, pred)
        ters.append(ter)

        id = len(clean_label)
        res[id] += ter
        counts[id] += 1

    print(np.mean(ters), np.mean(ious))
    print(res / counts)

if __name__ == '__main__':
    fire.Fire(eval)

# 6.22509225 4.44521338 3.20166667 2.44984227 1.96909091 1.63760504 1.38264388 1.20842411 1.06091549 0.95228028 0.83712121 0.7617866  0.69887476 0.62709832
# 4.84532374 4.33806147 3.33628319 2.71538462 2.23404255 1.87096774 1.61739865 1.41319444 1.24867257 1.11589174 1.02696078 0.94909502 0.87762238 0.80952381
# 2.46641791 1.87969925 1.43237705 1.42380952 1.21279762 1.20535714 1.35253906 1.24781145 1.17394366 1.07404692 1.01277372 0.94699418 0.87657563 0.82257218