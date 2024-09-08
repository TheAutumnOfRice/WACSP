from wacsp_handle import WACSP_Handle
from core.wacsp_module import solve_GRQ
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from core.utils import to_numpy
import torch
import mne
mne.set_log_level(99999)

# No preprocess needed for all the datasets in our work, except simple (x1e6) for uV scales.

Fs = 250
n_length = 1000
n_ch = 22

all_X = np.load(r"F:\dataset\BCICIV2AO\2traindata.npy")  # 288 x 22 x 1000
all_Y = np.load(r"F:\dataset\BCICIV2AO\2trainlabel.npy")  # 288

select_classes = [0,1]
select_ind = np.isin(all_Y, select_classes)
all_X = all_X[select_ind]
all_Y = all_Y[select_ind]

cv = StratifiedKFold(10, random_state=0, shuffle=True).split(all_X, all_Y)

all_X = torch.tensor(all_X).float().cuda()
all_Y = torch.tensor(all_Y).long().cuda()

scores = []
for all_ind, (train_ind, test_ind) in enumerate(cv):
    # if all_ind<9:
    #     continue
    TX = all_X[train_ind]
    TY = all_Y[train_ind]
    EX = all_X[test_ind]
    EY   = all_Y[test_ind]
    TD = [(x,y) for x,y in zip(TX,TY)]
    ED = [(x, y) for x, y in zip(EX, EY)]

    handle = WACSP_Handle(Fs, n_length, n_ch, max_windows=1, device="cuda")

    # Spatial Enhancement Settings
    handle.csp.do_spatial_enhance = True  # Manually enable spatial enhance

    # Automatically select K
    handle.csp.enhance_max_k = 4
    handle.csp.do_enhance_k = True

    # Automatically Channel Selection
    # handle.csp.enhance_CCS_list = [9,15,-1]
    # handle.csp.do_enhance_CCS = True

    state_dict = handle.fit(TD=TD, show_bar=True)
    pred = handle.predict(state_dict,EX)
    score = (to_numpy(EY.bool())==pred).mean()
    scores.append(score)

    # The First Selected TF Window
    t0, t1, f0, f1, _ = handle.csp.windows[0]
    t0 = int(handle.csp.time_indexes[t0])
    t1 = int(handle.csp.time_indexes[t1])
    t0r = t0 / Fs
    t1r = t1 / Fs

    freqs = np.fft.rfftfreq(n_length)
    f0r = freqs[int(handle.csp.freq_indexes[f0])] * Fs
    f1r = freqs[int(handle.csp.freq_indexes[f1])] * Fs

    print(f"First selected time-frequency window: [{t0r}-{t1r}]s x [{f0r}-{f1r}]Hz  ACC: {score}")
    use_k = handle.csp.enhances[0]['k']
    print(f"Enhancement: K={use_k}")

    # The next part of the code is to decompose the workflow of how WACSP makes final classifications.
    TX = all_X[train_ind]
    TY = all_Y[train_ind]
    EX = all_X[test_ind]
    EY = all_Y[test_ind]

    # Step 1:  Freq-Filter
    TZ = handle.csp.extract_filtered_covmats(f0, f1, TX, calc_cov=False)["Zs"]
    EZ = handle.csp.extract_filtered_covmats(f0, f1, EX, calc_cov=False)["Zs"]
    # TZ = torch.tensor(mne.filter.filter_data(TZ.cpu().numpy().astype(np.float64), Fs, f0r, f1r, n_jobs=1)).cuda()
    # EZ = torch.tensor(mne.filter.filter_data(EZ.cpu().numpy().astype(np.float64), Fs, f0r, f1r, n_jobs=1)).cuda()

    # Step 2:  Time-Filter
    TX = TZ[:, :, t0:t1]
    EX = EZ[:, :, t0:t1]

    # Step 3:  CSP Calculate
    TX0 = TX[TY == 0]
    TX1 = TX[TY == 1]
    C0 = TX0 @ TX0.transpose(-1, -2)
    C1 = TX1 @ TX1.transpose(-1, -2)
    C0 = C0.mean(axis=0)
    C1 = C1.mean(axis=0)
    w0 = solve_GRQ(C0, C1, k=use_k)
    w1 = solve_GRQ(C1, C0, k=use_k)
    ws = torch.cat([w0,w1], dim=0)

    # Step 4: CSP Apply
    TF = torch.mean((ws@TX)**2, dim=-1)
    TF = torch.log(TF/TF.sum(dim=-1, keepdims=True))
    EF = torch.mean((ws @ EX)**2, dim=-1)
    EF = torch.log(EF / EF.sum(dim=-1, keepdims=True))

    # Step 5: LDA Dimension Reduce
    TF = TF.cpu().numpy()
    EF = EF.cpu().numpy()
    TY = TY.cpu().numpy()
    EY = EY.cpu().numpy()
    lda = LinearDiscriminantAnalysis()
    lda.fit(TF, TY)
    TF = lda.transform(TF)
    EF = lda.transform(EF)

    # Step 6: SVM Classifier
    svm = LinearSVC(dual="auto")
    svm.fit(TF,TY)
    print(svm.score(EF,EY))



print("CV ACC: ", np.mean(scores))
