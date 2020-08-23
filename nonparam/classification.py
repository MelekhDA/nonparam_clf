import pandas as pd
import math

from functools import reduce
from typing import Union
from copy import deepcopy as dcopy

from .indicator import RaisedIndicator
from .kernel import RaisedKernel

__all__ = [
    'NonParamGapClf'
]


class NonParamGapClf:
    def __init__(
            self,
            k: Union[dict, int],
            kernel: Union[str, RaisedKernel],
            con_features: list,
            cat_features: list,
            *args,
            **kwargs
    ):
        """
        :param k: the number of objects under kernel,
                  example: {'feat1': val1, 'feat2': val2, ...} or val
        :param core: type kernel or function
        :param con_features: list continuous features
        :param cat_features: list categorical features
        :param epsi_core: value for objects, which not under kernel
        :param epsi_indicator: value, if rankes no equal
        """

        self._kernel = kernel
        self._con_feats = con_features.copy()
        self._cat_feats = cat_features.copy()

        if isinstance(k, dict):
            self._k = k.copy()
        else:
            self._k = {i: k for i in con_features}

        self._init_kernel(kwargs.get('epsi_kernel'))
        self._init_indicator(kwargs.get('epsi_indicator'))

    def _init_kernel(self, epsi: float) -> None:
        if isinstance(self._kernel, str):
            if epsi is None:
                self._kernel = RaisedKernel(self._kernel)
            else:
                self._kernel = RaisedKernel(self._kernel, epsi)

    def _init_indicator(self, epsi: float) -> None:
        if epsi is None:
            self._indicator = RaisedIndicator()

        self._indicator = RaisedIndicator(epsi)

    def fit(self, x: pd.DataFrame, y: pd.Series, *args, **kwargs) -> None:
        """fake fit (similar to Scikit-learn)"""

        self._x_tr = x.copy()
        self._y_tr = y.copy()

    def predict(self, x: pd.DataFrame, *args, **kwargs) -> list:
        """
        :param x: input features in the form of a dataframe
        :return: list of predictions for objects from dataframe x
        """

        predicts = []

        # test object
        for xts_i in x.index:
            values_cls = {}

            c = {i: self._get_c(x[i][xts_i], self._k[i], list(self._x_tr[i])) for i in self._con_feats}

            # all possible labels/classes
            for cls in self._y_tr.unique():
                st, cls = self._proximity_measure(x, cls, xts_i, c)
                values_cls.update({cls: st})

            pred = sorted(values_cls.items(), key=lambda x: x[1], reverse=True)[0]
            predicts.append(pred[0])

        return predicts

    def predict_proba(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        :param x: input features in the form of a dataframe
        :return: dataframe with a measure of proximity to classes
        """

        df_predicts = pd.DataFrame(columns=self._y_tr.unique().sort())

        # test object
        for xts_i in x.index:
            values_cls, normalizer = {}, 0

            c = {i: self._get_c(x[i][xts_i], self._k[i], list(self._x_tr[i])) for i in self._con_feats}

            # all possible labels/classes
            for cls in self._y_tr.unique():
                st, cls = self._proximity_measure(x, cls, xts_i, c)
                values_cls.update({cls: st})
                normalizer += st

            values_normalize_cls = {cls: st / normalizer for cls, st in values_cls.items()}

            df_predicts = df_predicts.append(values_normalize_cls, ignore_index=True)

        return df_predicts

    def _proximity_measure(self, x: pd.DataFrame, cls: int, xts_i: int, c: dict) -> tuple:
        """
        resulting value by class <cls>

        :param x - test object
        :param cls - label/class
        :param xts_i - index of test object
        :param c - dict <continuous feature: window width>

        :return tuple (class, proximity measure)
        """

        st = 0

        # object from train with corresponding label
        for xtr_i in self._y_tr[self._y_tr == cls].index:
            if self._con_feats:
                con_l = [
                    self._kernel.kernel((self._x_tr[i][xtr_i] - x[i][xts_i]) / c[i])
                    for i in self._con_feats
                ]
                con = reduce(lambda x, y: x * y, con_l)
            else:
                con = 1

            if self._cat_feats:
                cat_l = [
                    self._indicator.indicator(self._x_tr[i][xtr_i], x[i][xts_i])
                    for i in self._cat_feats
                ]
                cat = reduce(lambda x, y: x * y, cat_l)
            else:
                cat = 1

            st += con * cat

        return st, cls

    def _get_c(self, value: float, n: int, tr_list: list) -> float:
        """window width"""

        tr_list_tmp = dcopy(tr_list)

        # added a test value to the train part
        tr_list_tmp.append(value)
        # unique values
        tr_list_tmp = list(set(tr_list_tmp))
        # test item index
        ix = tr_list_tmp.index(value)

        ix_left = [i for i in range(ix - 1, self._ceil_l(ix - n), -1)]
        ix_right = [i for i in range(ix + 1, self._ceil_r(ix + n, len(tr_list_tmp)))]

        # distance between test value and all (train)
        dist_s = [math.fabs(tr_list_tmp[i] - tr_list_tmp[ix]) for i in ix_left + ix_right]
        # sorted
        dist_s = sorted(dist_s)
        # maximum value
        c = dist_s[n - 2]

        return c

    def _ceil_l(self, x: int, limit: int = 0) -> int:
        if x < limit:
            return limit
        return x

    def _ceil_r(self, x: int, limit: int) -> int:
        if x > limit:
            return limit
        return x

    def get_max_k(self) -> dict:
        """
        To determine the maximum possible value of k

        :return: dict <feature: maximum value k>
        """

        ks = {}
        for i in self._con_feats:
            ks[i] = len(set(self._x_tr[i])) - 1

        return ks
