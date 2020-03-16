import numpy as np
from copy import deepcopy


def preprocess_Y(Yin, nscf):

    """
    :param Yin: numpy array, the Entity-to-feature
    :param nscf:  Dict. the dict-key is the index of categorical variable V_l in Y, and dict-value is the number of
                    sub-categorie b_v (|V_l|) in categorical feature V_l.
            Apply Z-scoring, normalization by range, and Prof. Mirkin's 3-stage normalization method,
            constant features elimination normalization.
    :return: Original entity-to-feature data matrix, Z-scored preprocessed matrix, 2-stages preprocessed matrix,
            3-stages preprocessed matrix and their corresponding relative contribution
    """

    Y_std = np.std(Yin, axis=0)
    cnst_features = np.where(Y_std == 0)[0].tolist()  # detecting constant features, I.e std = rng(MaxMin) = 0
    Yin_cnst_free = np.delete(Yin, obj=cnst_features, axis=1)  # New Yin s.t all constant features are removed

    TY = np.sum(np.multiply(Yin_cnst_free, Yin_cnst_free))  # data scatter, T stands for data scatter
    TY_v = np.sum(np.multiply(Yin_cnst_free, Yin_cnst_free), axis=0)  # feature scatter
    Y_rel_cntr = TY_v / TY  # relative contribution


    mean_Y = np.mean(Yin_cnst_free, axis=0)
    std_Y = np.std(Yin_cnst_free, axis=0)

    Yz = np.divide(np.subtract(Yin_cnst_free, mean_Y), std_Y)  # Z-score

    TYz = np.sum(np.multiply(Yz, Yz))
    TYz_v = np.sum(np.multiply(Yz, Yz), axis=0)
    Yz_rel_cntr = TYz_v / TYz

    scale_min_Y = np.min(Yin_cnst_free, axis=0)
    scale_max_Y = np.max(Yin_cnst_free, axis=0)
    rng_Y = scale_max_Y - scale_min_Y

    # 3 steps normalization (Range-without follow-up division)
    Yrng = np.divide(np.subtract(Yin_cnst_free, mean_Y), rng_Y)
    TYrng = np.sum(np.multiply(Yrng, Yrng))
    TYrng_v = np.sum(np.multiply(Yrng, Yrng), axis=0)
    Yrng_rel_cntr = TYrng_v / TYrng

    # For the combination of quantitative and categorical feature  with one hot vector encoding
    # this section should be modified such that the removed columns are taken into account.
    Yrng_rs = deepcopy(Yrng)  # 3 steps normalization (Range-with follow-up division)
    for k, v in nscf.items():
        Yrng_rs[:, int(k)] = Yrng_rs[:, int(k)] / np.sqrt(int(v))  # : int(k)+ int(v)

    #     Yrng_rs = (Y_rescale - Y_mean)/ rng_Y
    TYrng_rs = np.sum(np.multiply(Yrng_rs, Yrng_rs))
    TYrng_v_rs = np.sum(np.multiply(Yrng_rs, Yrng_rs), axis=0)
    Yrng_rel_cntr_rs = TYrng_v_rs / TYrng_rs

    return Yin_cnst_free, Y_rel_cntr, Yz, Yz_rel_cntr, Yrng, Yrng_rel_cntr, cnst_features  # Yrng_rs, Yrng_rel_cntr_rs

