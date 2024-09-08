import random
import torch
from core.wacsp_module import WACSP
from core.utils import to_numpy
import math
import os
import sys
from tqdm import tqdm
import numpy as np
from core.feature_fusion import FeatureFusion

SEARCH_DEFAULT_ARGS = dict(
    # >> SEARCH PARAMETERS  << #

    # Multiple-window searching settings
    max_windows=20,  # Max number of windows to be searched.
    overlap_mode="max",
    overlap_gamma=2.,

    # Initial Windows Settings
    freq_band_points=(5, 10, 15, 20, 25, 30, 35, 40),  # Initial frequency windows that required to cover these points (Hz)
    freq_band_init_min_size=4.,  # Min Freq-Width (Hz)
    freq_band_init_max_size=12.,  # Max Freq-Width (Hz)
    fix_init_freq_bands=None,  # Use fixed freq bands instead of random selected

    time_slice_random_select=2,  # Number of initial time windows per frequnecy window
    time_slice_init_min_length=0.5,  # Min Time-Length (s)
    time_slice_init_max_length=2.5,  # Max Time-Length (s)
    fix_init_time_bands=None,  # Use fixed time periods instead of random selected

    # VS-CD Parameters
    step_size_list=(4,2,1),  # Step-size Multipliers
    max_iter=5,  # Repeat number per multiplier

    # HC Parameters
    max_step=10,  # n_step
    default_time_modes=["move", "scale"],  # Control the HC directions (Default: Move, L-Scale, R-Scale)
    default_freq_modes=["move", "scale"],

    # TF-Grid (Quantity) Settings
    grid_s_Hz=(0.1, 1.),  # Min grid size (Default 0.1s x 1.Hz)
    max_Hz=50.,  # Frequency Upper Bound
    min_tfg=(5, 4),  # Min window size (Default: 5T-Grid x 4F-Grid, i.e. 0.5s x 4Hz)
    max_tfg=(99999, 99999),  # Max window size (Default: INF)

    # >> CSP PARAMETERS << #
    select_csp_pair=1,  # Number of CSP Feature Pairs
    CCS=None,
    tkreg=0.,

    # >> Memory Settings << #
    device="cuda",
    batchsize=20,
    extract_batchsize=50,
    max_masks_in_one_batch=20,

    # >> CV-Based Metric Settings << #
    criterion="AFR",
    csp_cv_seed=1,
    csp_cv=5,

    # >> TF-Filter Settings << #
    filter_mode="fir",
    # Fir Settings
    fir_time=1.,  # Length: 1s
    fir_tap="auto",  # Use Fir_time as number of taps
    fir_window="hamming",
    # Time Window Settings
    crop_window="hamming",

    # >> Verbose << #
    verbose=True,
    view_acc_at_each_epoch=False,
)

class WACSP_Handle:
    def __init__(self, Fs, n_length, n_ch, **kwargs):
        args = SEARCH_DEFAULT_ARGS.copy()
        args.update(kwargs)
        self.args = args
        self.Fs = Fs
        self.n_length = n_length
        self.n_ch = n_ch
        if "grid_tf" in args:
            self.grid_tf = args["grid_tf"]
        else:
            self.grid_s_Hz = args["grid_s_Hz"]
            grid_t = int(math.ceil(self.grid_s_Hz[0] * self.Fs))
            grid_f = int(math.ceil(self.grid_s_Hz[1] / self.Fs * self.n_length + 1))
            self.grid_tf = (grid_t, grid_f)


        fir_tap = args["fir_tap"]
        if fir_tap == "auto":
            fir_tap = int((args["fir_time"] * self.Fs) // 2 * 2 + 1)
        self.fir_tap = fir_tap

        self.csp = WACSP(
            grid_tf=self.grid_tf,
            min_tfg=args["min_tfg"],
            Fs=self.Fs,
            n_length=self.n_length,
            n_ch=self.n_ch,
            select_csp_pair=args["select_csp_pair"],
            tkreg=args["tkreg"],
            max_Hz=args["max_Hz"],
            device=args["device"],
            n_cv=args["csp_cv"],
            CCS=args["CCS"],
        )
        self.csp.batchsize = args["batchsize"]
        self.csp.extract_batchsize = args["extract_batchsize"]
        self.csp.max_masks_in_one_batch = args["max_masks_in_one_batch"]
        self.csp.fir_window = args["fir_window"]
        self.csp.crop_window = args["crop_window"]
        self.csp.filter_mode = args["filter_mode"]
        self.csp.overlap_mode = args["overlap_mode"]
        self.csp.overlap_gamma = args["overlap_gamma"]
        self.csp.fir_tap = self.fir_tap
        self.csp.default_time_modes = self.default_time_modes = args["default_time_modes"]
        self.csp.default_freq_modes = self.default_freq_modes = args["default_freq_modes"]
        self.csp.csp_cv_seed = args["csp_cv_seed"]
        self.record = {
            "args": args,
            "mode_score": self.csp.mode_score,
            "enhances": self.csp.enhances,
            "enhance_cache": self.csp.enhance_cache,
            "CCS_cache": self.csp.CCS_selected_cache,
            "all_tracks": [],
            "modes": self.csp.windows,
            "op_cv_acc": [],
            "svm_op_cv_acc": [],
            "svm_op_cv_cm": [],
            "init_tfs": [],
            "split_state":{},
        }

    def initialize_dataset(self, TD):
        self.csp.init_train_dataset(TD)

    def generate_init_tfs(self, divider=1):
        n_fft = self.csp.n_fft
        # n_init_slice_length = self.time_slice_length * self.Fs
        # init_t0 = np.floor((np.linspace(0, self.n_length - n_init_slice_length, self.time_slice_num))).astype(int)
        # init_t1 = (init_t0 + n_init_slice_length).astype(int)
        init_tfs = []
        grid_t, grid_f = self.grid_tf

        for f_center in self.args["freq_band_points"] \
                if self.args["fix_init_freq_bands"] is None\
                else self.args["fix_init_freq_bands"]:
            # if self.time_slice_random_select == 0:
            #     for t0, t1 in zip(init_t0, init_t1):
            #         init_tfs.append([t0 // grid_t, t1 // grid_t, f0 // grid_f, f1 // grid_f])
            # else:
            for time_band in range(self.args["time_slice_random_select"])\
                    if self.args["fix_init_time_bands"] is None\
                    else self.args["fix_init_time_bands"]:
                if self.args["fix_init_time_bands"] is None:
                    time_length = random.random() * (self.args["time_slice_init_max_length"] - self.args["time_slice_init_min_length"]) \
                                  + self.args["time_slice_init_min_length"]
                    cur_grid_num = int(time_length * self.Fs / grid_t)
                    gt0 = random.randrange(0, self.csp.n_time-cur_grid_num)
                    gt1 = gt0 + cur_grid_num
                else:
                    t0, t1 = time_band
                    gt0 = int(t0*self.Fs//grid_t)
                    gt1 = int(t1*self.Fs//grid_t)
                if self.args["fix_init_freq_bands"] is None:
                    freq_length = random.random() * (self.args["freq_band_init_max_size"] - self.args["freq_band_init_min_size"]) \
                                 + self.args["freq_band_init_min_size"]
                    f0 = f_center - freq_length/2
                    f1 = f_center + freq_length/2
                else:
                    f0, f1 = f_center
                if f0<0.5:
                    f1 = f1 + (0.5 - f0)
                    f0 = 0.5
                if f1>self.args["max_Hz"]-0.5:
                    f0 = f0 - (f1 - (self.args["max_Hz"]-0.5))
                    f1 = self.args["max_Hz"]-0.5
                gf0 = int(f0/self.Fs*self.n_length/grid_f)
                gf1 = int(f1/self.Fs*self.n_length/grid_f)
                if gf1 >= self.csp.n_freq-1:
                    gf1 = self.csp.n_freq-1
                if self.args["fix_init_freq_bands"] is None:
                    gf0 = gf0 // divider * divider
                    gf1 = gf1 // divider * divider
                if self.args["fix_init_time_bands"] is None:
                    gt0 = gt0//divider*divider
                    gt1 = gt1//divider*divider
                if gt0==gt1:
                    gt1=gt0+divider
                if gf0==gf1:
                    gf1=gf0+divider
                if gt1 >= self.csp.n_time-1:
                    gt0, gt1 = gt0-divider, gt1-divider
                if gf1 >= self.csp.n_freq-1:
                    gf0, gf1 = gf0-divider, gf1-divider
                init_tfs.append([gt0, gt1, gf0, gf1])
        # print(init_tfs)
        return init_tfs

    def load_record(self, record):
        self.record = record
        self.csp.load_mode(self.record["modes"])
        self.csp.mode_score = self.record["mode_score"]
        self.csp.enhances = self.record["enhances"]
        if "CCS_cache" in self.record:
            self.csp.CCS_selected_cache = self.record["CCS_cache"]
        if "enhance_cache" in self.record:
            self.csp.enhance_cache = self.record["enhance_cache"]

    def get_record(self):
        return self.record

    def logging(self, words):
        if self.args["verbose"]:
            print(words)

    def search_new_window(self, show_bar=True):
        modes = self.record["modes"]
        n_comp = len(modes) + 1
        self.logging(f"Search component: {n_comp}")
        all_tracks = []
        all_best_mode = None
        all_best_score = -9999999999
        init_tfs = self.generate_init_tfs(divider=self.args["step_size_list"][0])
        self.record["init_tfs"].append(init_tfs)
        for tfs in tqdm(init_tfs, disable=not show_bar, desc=f"Comp. {n_comp:2}"):
            track = []
            t0, t1, f0, f1 = tfs
            backup_covmat = None
            for stepsize in self.args["step_size_list"]:

                best_mode, best_score, recs, backup_covmat = self.csp.tf_onlytime_descent((t0, t1, f0, f1),
                                                                                          stepsize,
                                                                                          self.args["max_step"],
                                                                                          self.args["max_iter"],
                                                                                          criterion=self.args["criterion"],
                                                                                          backup_covmat=backup_covmat,
                                                                                          )

                if best_score > all_best_score:
                    all_best_score = best_score
                    all_best_mode = best_mode
                # Add tracking
                track.append(recs)
                # if len(track) == 0:
                #     _tfs = recs[0]["ori_tfs"]
                #     _sco = recs[0]["all_scores"][0].item()
                #     track.append((_tfs, _sco))
                # for rec in recs[:-1]:
                #     _ind = rec['best_ind']
                #     _tfs = rec['all_tfgs'][_ind]
                #     _sco = rec['all_scores'][_ind].item()
                #     track.append((_tfs, _sco))
                t0, t1, f0, f1, mi = best_mode
            all_tracks.append(track)

        return all_best_mode, all_best_score, all_tracks

    def train_windows(self, TD, ED=None, show_bar=True):
        # ED is for view epoch accs
        # TD/ED: Dataset of (n_ch x n_length)
        self.initialize_dataset(TD)
        TX, _ = TD[0]
        Tshape = (len(TD), *TX.shape)

        bincnt = torch.bincount(self.csp.train_dataset["ys"]).cpu().tolist()
        self.logging(f"Train Shape: {Tshape};  Class Cnt: {bincnt}")
        # self.logging(f"ARGS: {self.args}")

        while len(self.csp.windows) < self.args["max_windows"]:
            mode, score, all_tracks = self.search_new_window(show_bar)
            self.logging(f"+ Mode: {mode} with score: {score:.4f}")
            if self.csp.do_spatial_enhance:
                enhance = self.csp.enhance_cache[mode[:4]][0]
                self.logging(f"+ Enhance: {enhance}")
                self.csp.enhances.append(enhance)
                self.record["enhances"] = self.csp.enhances
            self.csp.windows.append(mode)
            self.record["modes"] = self.csp.windows
            self.record["all_tracks"].append(all_tracks)
            # test single window score
            if ED is not None and self.args["view_acc_at_each_epoch"]:
                windows = self.csp.windows
                op_window = windows[-1]
                op_cv_acc = self.csp.calc_op_cv_score(op_window, "acc")
                op_cv_score = self.csp.mode_score[op_window[:4]][op_window[-1]]
                # svm_op_cv_score = self.csp.calc_op_cv_score(mode, "svm_acc")
                # self.logging(f"CV acc: {op_cv_acc * 100:.2f} Score: {op_cv_score} SVM: {svm_op_cv_score}", verbose=True)
                # self.record["op_cv_acc"].append(op_cv_acc)
                # self.record["svm_op_cv_acc"].append(svm_op_cv_score)
                # svm_op_cv_score = self.csp.calc_op_cv_score(mode, "svm_acc")
                side_score = self.side_score(TD=TD, ED=ED, check=("rank",))
                detail = {
                    "mode": windows[-1],
                    "op_cv_acc": op_cv_acc,
                    "op_cv_score": op_cv_score,
                    "rank_acc":side_score["rank_acc"],
                }
                detail.update(side_score)
                # self.logging(f"SVM acc: {len(self.csp.windows)}-p: {detail['svm_acc']*100:.2f} ;"
                #              f" 1-p: {detail['svm_op_acc']*100:.2f}")
                self.logging(f"Vote acc: {detail['rank_acc']*100:.2f}")


        self.logging("Finished.")

    def fit(self, TD=None, show_bar=True):
        self.train_windows(TD, show_bar=show_bar)
        state_dict = self.csp.fit(TD)
        train_rdis = state_dict["rdis_out"]
        train_ys = state_dict["ys"]
        # output = self.csp.transform(state_dict, ED)
        # test_rdis = output["rdis_out"]
        # test_out = output["mp_out"]
        # test_ys = output["ys"]
        tx = train_rdis.cpu().numpy()
        # ex = test_rdis.cpu().numpy()
        tx = tx.reshape(len(tx), -1)
        # ex = ex.reshape(len(ex), -1)
        ty = train_ys.cpu().numpy()
        # ey = test_ys.cpu().numpy()
        SF = FeatureFusion('rank')
        SF.fit(tx, ty)
        return {
            "record": self.get_record(),
            "csp_state_dict": state_dict,
            "SF": SF,
        }

    def predict(self, state_dict, X=None, ED=None, max_window=None, proba=False):
        self.load_record(state_dict["record"])
        csp_state_dict = state_dict["csp_state_dict"]
        SF = state_dict["SF"]
        result = self.csp.transform(csp_state_dict, ED, X, max_window=max_window)
        rdis = result["rdis_out"]
        back_clf = SF.clf
        SF.clf = SF.clf[:rdis.shape[1]]
        if proba:
            out = SF.predict_proba(rdis, target_cls_weight=0.5)
        else:
            out = SF.predict(rdis)
        SF.clf = back_clf
        return out

    def evaluate_cv_result(self, check=('rank',)):
        txs, tys, exs, eys = self.get_fit_selfCV_data()
        cv_results = {}
        for tx, ty, ex, ey in zip(txs, tys, exs, eys):
            side_score = self.side_score(all_p=True, check=check, ft_data=(tx, ty, ex, ey))
            for p in range(self.args["max_windows"]):
                cv_results.setdefault(p, {})
                for clf in check:
                    cv_results[p].setdefault(clf, [])
                    cv_results[p][clf].append(side_score[f"{clf}_accs"][p])
        return cv_results


    def get_fit_selfCV_data(self):
        clf_output = self.csp.fit_selfCV()
        cv_select = clf_output["cv_select"]
        rdis_out = clf_output['rdis_out']
        ys = clf_output["ys"]
        txs = []
        exs = []
        tys = []
        eys = []
        for ind, (train_ind, test_ind) in enumerate(cv_select):
            txs.append(rdis_out[ind, train_ind].cpu().numpy())
            exs.append(rdis_out[ind, test_ind].cpu().numpy())
            tys.append(ys[train_ind].cpu().numpy())
            eys.append(ys[test_ind].cpu().numpy())
        scdata = (txs,tys,exs,eys)
        return scdata

    def fit_transform(self, TD, ED):
        state_dict = self.csp.fit(TD)
        train_rdis = state_dict["rdis_out"]
        train_ys = state_dict["ys"]
        output = self.csp.transform(state_dict, ED)
        test_rdis = output["rdis_out"]
        # test_out = output["mp_out"]
        test_ys = output["ys"]
        tx = train_rdis.cpu().numpy()
        ex = test_rdis.cpu().numpy()
        tx = tx.reshape(len(tx), -1)
        ex = ex.reshape(len(ex), -1)
        ty = train_ys.cpu().numpy()
        ey = test_ys.cpu().numpy()
        return tx,ty,ex,ey

    def side_score(self, TD=None, ED=None, all_p=False,
                   check=('svm','RR','rank','wrank'), ft_data=None, select_p=None,**kwargs):
        if ft_data is not None:
            tx, ty, ex, ey = ft_data
        else:
            tx, ty, ex, ey = self.fit_transform(TD,ED)
        scores = {}
        all_p_list = [p[:4] for p in self.csp.windows]

        if 'svm' in check:
            SF = FeatureFusion("SVM")
            SF.fit(tx, ty)
            svm_acc = SF.score(ex, ey)
            scores['svm_acc'] = svm_acc
            SF.fit(tx[:, -1:], ty)
            svm_op_acc = SF.score(ex[:, -1:], ey)
            scores["svm_op_acc"] = svm_op_acc
        if 'bsvm' in check:
            SF = FeatureFusion("BSVM", **kwargs)
            SF.fit(tx, ty)
            bsvm_acc = SF.score(ex, ey)
            scores['bsvm_acc'] = bsvm_acc
        if "RR" in check:
            SF = FeatureFusion("RR",**kwargs)
            SF.fit(tx, ty)
            RR_acc = SF.score(ex, ey)
            scores['RR_acc'] = RR_acc
        if "HGB" in check:
            SF = FeatureFusion("HGB",**kwargs)
            SF.fit(tx, ty)
            RR_acc = SF.score(ex, ey)
            scores['HGB_acc'] = RR_acc
        if 'rank' in check:
            SF = FeatureFusion('rank')
            SF.fit(tx, ty)
            scores['rank_acc'] = SF.score(ex,ey)
        if 'wrank' in check:
            SF = FeatureFusion('wrank')
            all_mode_score = [self.csp.mode_score[p][0] for p in all_p_list]
            SF.fit(tx, ty, clf_weight=all_mode_score)
            scores['wrank_acc'] = SF.score(ex,ey)

        if all_p:
            svm_accs = []
            bsvm_accs = []
            rank_accs = []
            wrank_accs = []
            RR_accs = []
            HGB_accs = []
            if select_p is None:
                select_p = list(range(1,len(self.csp.windows)+1))
            for p in select_p:
                if 'svm' in check:
                    SF = FeatureFusion("SVM")
                    SF.fit(tx[:,:p], ty)
                    svm_accs.append(SF.score(ex[:,:p], ey))
                if 'bsvm' in check:
                    SF = FeatureFusion("BSVM", **kwargs)
                    SF.fit(tx[:,:p], ty)
                    bsvm_accs.append(SF.score(ex[:,:p], ey))
                if 'rank' in check:
                    SF = FeatureFusion('rank')
                    SF.fit(tx[:, :p], ty)
                    rank_accs.append(SF.score(ex[:, :p], ey))
                if 'wrank' in check:
                    SF = FeatureFusion('wrank')
                    SF.fit(tx[:,:p], ty, clf_weight=all_mode_score[:p])
                    wrank_accs.append(SF.score(ex[:, :p], ey))
                if 'RR' in check:
                    SF = FeatureFusion('RR', **kwargs)
                    SF.fit(tx[:, :p], ty)
                    RR_accs.append(SF.score(ex[:, :p], ey))
                if 'HGB' in check:
                    SF = FeatureFusion('HGB', **kwargs)
                    SF.fit(tx[:, :p], ty)
                    HGB_accs.append(SF.score(ex[:, :p], ey))

            if 'svm' in check:
                scores["svm_accs"] = svm_accs
            if 'bsvm' in check:
                scores["bsvm_accs"] = bsvm_accs
            if 'rank' in check:
                scores['rank_accs'] = rank_accs
            if 'wrank' in check:
                scores['wrank_accs'] = wrank_accs
            if 'RR' in check:
                scores['RR_accs'] = RR_accs
            if 'HGB' in check:
                scores['HGB_accs'] = RR_accs

        # self.side_fusion.fit(tx, ty)
        # train_acc = self.side_fusion.score(tx, ty)
        # test_acc = self.side_fusion.score(ex, ey)
        return scores
