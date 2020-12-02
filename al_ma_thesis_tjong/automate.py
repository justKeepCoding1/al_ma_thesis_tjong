'''Automate test on entire MNIST and F_MNIST, both reduced and unreduced'''


from active_learning_prototypes.always_from_scratch import qbc_cnn_ce as ce_scratch, \
    qbc_cnn_ve as ve_scratch, qbc_cnn_random as random_scratch
from active_learning_prototypes.only_new_data import qbc_cnn_ce as ce_new, \
    qbc_cnn_ve as ve_new, qbc_cnn_random as random_new
from active_learning_prototypes.incremental_with_step import qbc_cnn_ce as ce_increment, \
    qbc_cnn_ve as ve_increment, qbc_cnn_random as random_increment

'''
Order always:
1. Random
2. Consensus
3. Vote
'''


def main():
    seed = 13
    for i in range(1): 
        
        random_new.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                       idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist', seed=seed)
        ce_new.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                   idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist', seed=seed)
        ve_new.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                   idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist', seed=seed)

        '''random_increment.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                             idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist')
        ce_increment.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                         idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist')
        ve_increment.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                         idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist')

        random_scratch.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                           idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist', seed=seed)
        ce_scratch.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                       idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist', seed=seed)
        ve_scratch.qbc(n_model=10, n_train=60000, qbc_batch_size=3000, batch_size=60,
                       idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist', seed=seed)'''

        a = 1


if __name__ == '__main__':
    main()