"""Blah"""
# %%
from curses.ascii import FF
import pickle
import gc
import copy

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

INFERENCE = True


def amex_metric(y_true, y_pred, return_components=False) -> float:
    """Amex metric for ndarrays"""

    def top_four_percent_captured(df) -> float:
        """Corresponds to the recall for a threshold of 4 %"""
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df["weight"].sum())
        df["weight_cumsum"] = df["weight"].cumsum()
        df_cutoff = df.loc[df["weight_cumsum"] <= four_pct_cutoff]
        return (df_cutoff["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(df) -> float:
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos_found"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(df) -> float:
        """Corresponds to 2 * AUC - 1"""
        df2 = pd.DataFrame({"target": df.target, "prediction": df.target})
        df2.sort_values("prediction", ascending=False, inplace=True)
        return weighted_gini(df) / weighted_gini(df2)

    df = pd.DataFrame({"target": y_true.ravel(), "prediction": y_pred.ravel()})
    df.sort_values("prediction", ascending=False, inplace=True)
    g = normalized_weighted_gini(df)
    d = top_four_percent_captured(df)
    print(f"G: {g:.6f}, D: {d:.6f}, ALL: {0.5 * (g + d):6f}")
    if return_components:
        return g, d, 0.5 * (g + d)
    return 0.5 * (g + d)


# Preprocessing data
features_avg = [
    "B_11",
    "B_13",
    "B_14",
    "B_15",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "B_2",
    "B_20",
    "B_28",
    "B_29",
    "B_3",
    "B_33",
    "B_36",
    "B_37",
    "B_4",
    "B_42",
    "B_5",
    "B_8",
    "B_9",
    "D_102",
    "D_103",
    "D_105",
    "D_111",
    "D_112",
    "D_113",
    "D_115",
    "D_118",
    "D_119",
    "D_121",
    "D_124",
    "D_128",
    "D_129",
    "D_131",
    "D_132",
    "D_133",
    "D_139",
    "D_140",
    "D_141",
    "D_143",
    "D_144",
    "D_145",
    "D_39",
    "D_41",
    "D_42",
    "D_43",
    "D_44",
    "D_45",
    "D_46",
    "D_47",
    "D_48",
    "D_49",
    "D_50",
    "D_51",
    "D_52",
    "D_56",
    "D_58",
    "D_62",
    "D_70",
    "D_71",
    "D_72",
    "D_74",
    "D_75",
    "D_79",
    "D_81",
    "D_83",
    "D_84",
    "D_88",
    "D_91",
    "P_2",
    "P_3",
    "R_1",
    "R_10",
    "R_11",
    "R_13",
    "R_18",
    "R_19",
    "R_2",
    "R_26",
    "R_27",
    "R_28",
    "R_3",
    "S_11",
    "S_12",
    "S_22",
    "S_23",
    "S_24",
    "S_26",
    "S_27",
    "S_5",
    "S_7",
    "S_8",
]
features_min = [
    "B_13",
    "B_14",
    "B_15",
    "B_16",
    "B_17",
    "B_19",
    "B_2",
    "B_20",
    "B_22",
    "B_24",
    "B_27",
    "B_28",
    "B_29",
    "B_3",
    "B_33",
    "B_36",
    "B_4",
    "B_42",
    "B_5",
    "B_9",
    "D_102",
    "D_103",
    "D_107",
    "D_109",
    "D_110",
    "D_111",
    "D_112",
    "D_113",
    "D_115",
    "D_118",
    "D_119",
    "D_121",
    "D_122",
    "D_128",
    "D_129",
    "D_132",
    "D_133",
    "D_139",
    "D_140",
    "D_141",
    "D_143",
    "D_144",
    "D_145",
    "D_39",
    "D_41",
    "D_42",
    "D_45",
    "D_46",
    "D_48",
    "D_50",
    "D_51",
    "D_53",
    "D_54",
    "D_55",
    "D_56",
    "D_58",
    "D_59",
    "D_60",
    "D_62",
    "D_70",
    "D_71",
    "D_74",
    "D_75",
    "D_78",
    "D_79",
    "D_81",
    "D_83",
    "D_84",
    "D_86",
    "D_88",
    "D_96",
    "P_2",
    "P_3",
    "P_4",
    "R_1",
    "R_11",
    "R_13",
    "R_17",
    "R_19",
    "R_2",
    "R_27",
    "R_28",
    "R_4",
    "R_5",
    "R_8",
    "S_11",
    "S_12",
    "S_23",
    "S_25",
    "S_3",
    "S_5",
    "S_7",
    "S_9",
]
features_max = [
    "B_1",
    "B_11",
    "B_13",
    "B_15",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "B_2",
    "B_22",
    "B_24",
    "B_27",
    "B_28",
    "B_29",
    "B_3",
    "B_31",
    "B_33",
    "B_36",
    "B_4",
    "B_42",
    "B_5",
    "B_7",
    "B_9",
    "D_102",
    "D_103",
    "D_105",
    "D_109",
    "D_110",
    "D_112",
    "D_113",
    "D_115",
    "D_121",
    "D_124",
    "D_128",
    "D_129",
    "D_131",
    "D_139",
    "D_141",
    "D_144",
    "D_145",
    "D_39",
    "D_41",
    "D_42",
    "D_43",
    "D_44",
    "D_45",
    "D_46",
    "D_47",
    "D_48",
    "D_50",
    "D_51",
    "D_52",
    "D_53",
    "D_56",
    "D_58",
    "D_59",
    "D_60",
    "D_62",
    "D_70",
    "D_72",
    "D_74",
    "D_75",
    "D_79",
    "D_81",
    "D_83",
    "D_84",
    "D_88",
    "D_89",
    "P_2",
    "P_3",
    "R_1",
    "R_10",
    "R_11",
    "R_26",
    "R_28",
    "R_3",
    "R_4",
    "R_5",
    "R_7",
    "R_8",
    "S_11",
    "S_12",
    "S_23",
    "S_25",
    "S_26",
    "S_27",
    "S_3",
    "S_5",
    "S_7",
    "S_8",
]
features_last = [
    "B_1",
    "B_11",
    "B_12",
    "B_13",
    "B_14",
    "B_16",
    "B_18",
    "B_19",
    "B_2",
    "B_20",
    "B_21",
    "B_24",
    "B_27",
    "B_28",
    "B_29",
    "B_3",
    "B_30",
    "B_31",
    "B_33",
    "B_36",
    "B_37",
    "B_38",
    "B_39",
    "B_4",
    "B_40",
    "B_42",
    "B_5",
    "B_8",
    "B_9",
    "D_102",
    "D_105",
    "D_106",
    "D_107",
    "D_108",
    "D_110",
    "D_111",
    "D_112",
    "D_113",
    "D_114",
    "D_115",
    "D_116",
    "D_117",
    "D_118",
    "D_119",
    "D_120",
    "D_121",
    "D_124",
    "D_126",
    "D_128",
    "D_129",
    "D_131",
    "D_132",
    "D_133",
    "D_137",
    "D_138",
    "D_139",
    "D_140",
    "D_141",
    "D_142",
    "D_143",
    "D_144",
    "D_145",
    "D_39",
    "D_41",
    "D_42",
    "D_43",
    "D_44",
    "D_45",
    "D_46",
    "D_47",
    "D_48",
    "D_49",
    "D_50",
    "D_51",
    "D_52",
    "D_53",
    "D_55",
    "D_56",
    "D_59",
    "D_60",
    "D_62",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
    "D_70",
    "D_71",
    "D_72",
    "D_73",
    "D_74",
    "D_75",
    "D_77",
    "D_78",
    "D_81",
    "D_82",
    "D_83",
    "D_84",
    "D_88",
    "D_89",
    "D_91",
    "D_94",
    "D_96",
    "P_2",
    "P_3",
    "P_4",
    "R_1",
    "R_10",
    "R_11",
    "R_12",
    "R_13",
    "R_16",
    "R_17",
    "R_18",
    "R_19",
    "R_25",
    "R_28",
    "R_3",
    "R_4",
    "R_5",
    "R_8",
    "S_11",
    "S_12",
    "S_23",
    "S_25",
    "S_26",
    "S_27",
    "S_3",
    "S_5",
    "S_7",
    "S_8",
    "S_9",
]
features_categorical = [
    "B_30_last",
    "B_38_last",
    "D_114_last",
    "D_116_last",
    "D_117_last",
    "D_120_last",
    "D_126_last",
    "D_63_last",
    "D_64_last",
    "D_66_last",
    "D_68_last",
]

for i in [0, 1] if INFERENCE else [0]:
    # i == 0 -> process the train data
    # i == 1 -> process the test data
    df = pd.read_feather(
        ["data/archive/train_data.ftr", "data/archive/test_data.ftr"][i]
    )
    cid = pd.Categorical(df.pop("customer_ID"), ordered=True)
    last = cid != np.roll(cid, -1)  # mask for last statement of every customer
    if i == 0:  # train
        target = df.loc[last, "target"]
    print("Read", i)
    gc.collect()
    df_avg = (
        df.groupby(cid)
        .mean()[features_avg]
        .rename(columns={f: f"{f}_avg" for f in features_avg})
    )
    print("Computed avg", i)
    gc.collect()
    df_max = (
        df.groupby(cid)
        .max()[features_max]
        .rename(columns={f: f"{f}_max" for f in features_max})
    )
    print("Computed max", i)
    gc.collect()
    df_min = (
        df.groupby(cid)
        .min()[features_min]
        .rename(columns={f: f"{f}_min" for f in features_min})
    )
    print("Computed min", i)
    gc.collect()
    df_last = (
        df.loc[last, features_last]
        .rename(columns={f: f"{f}_last" for f in features_last})
        .set_index(np.asarray(cid[last]))
    )
    df = None  # we no longer need the original data
    print("Computed last", i)

    df_categorical = df_last[features_categorical].astype(object)
    features_not_cat = [f for f in df_last.columns if f not in features_categorical]
    if i == 0:  # train
        ohe = OneHotEncoder(
            drop="first", sparse=False, dtype=np.float32, handle_unknown="ignore"
        )
        ohe.fit(df_categorical)
        with open("ohe.pickle", "wb") as f:
            pickle.dump(ohe, f)
    df_categorical = pd.DataFrame(
        ohe.transform(df_categorical).astype(np.float16), index=df_categorical.index
    ).rename(columns=str)
    print("Computed categorical", i)

    df = pd.concat(
        [df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1
    )

    # Impute missing values
    df.fillna(value=0, inplace=True)

    del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat

    if i == 0:  # train
        # Free the memory
        df.reset_index(drop=True, inplace=True)  # frees 0.2 GByte
        df.to_feather("data/archive/train_processed.ftr")
        df = None
        gc.collect()

train = pd.read_feather("data/archive/train_processed.ftr")
test = df
del df, ohe

print("Shapes:", train.shape, target.shape)


class AmexDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class FFModel(nn.Module):
    """Feed Forward Neural Network with:

    3 hidden layers
    1 dropout layer
    """

    def __init__(self, in_feat, hid_dim=128, activation=nn.ReLU(), dropout=0.1):
        super(FFModel, self).__init__()
        self.encode = nn.Linear(in_feat, hid_dim)
        self.hidden1 = nn.Linear(hid_dim, hid_dim)
        self.hidden2 = nn.Linear(hid_dim, 64)
        self.drop = nn.Dropout(dropout)
        self.hidden3 = nn.Linear(64 + hid_dim, 16)
        self.pred = nn.Linear(16, 2)

        self.activation = activation

    def forward(self, x):
        e = self.activation(self.encode(x))
        h1 = self.activation(self.hidden1(e))
        h2 = self.activation(self.hidden2(h1))
        d = self.drop(torch.concat([h1, h2], dim=-1))
        h3 = self.activation(self.hidden3(d))
        return self.pred(h3)


# Define model:
# 4 hidden layers
# 1 skip connection
# 1 Dropout layer
class my_model(nn.Module):
    def __init__(self, in_feat, hid_dim=128, activation=nn.ReLU(), dropout=0.1):
        super(my_model, self).__init__()
        self.encode = nn.Linear(in_feat, hid_dim)
        self.hidden1 = nn.Linear(hid_dim, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.drop = nn.Dropout(dropout)
        self.hidden3 = nn.Linear(64 + hid_dim, 16)
        self.pred = nn.Linear(16, 2)

        self.activation = activation

    def forward(self, x):
        h0 = self.activation(self.encode(x))
        h1 = self.activation(self.hidden2(self.activation(self.hidden1(h0))))
        h = self.drop(torch.concat([h0, h1], dim=-1))
        h = self.activation(self.hidden3(h))
        return self.pred(h)


class early_stopper(object):
    def __init__(self, patience=12, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        # self.val_preds = []
        # self.val_logits = []

    def earlystop(self, loss, value, model=None):  # , preds, logits):
        """
        value: evaluation value on valiation dataset
        """
        cv = value
        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to("cpu")
            # self.val_preds = preds
            # self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print(f"EarlyStoper count: {self.count:02d}")
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to("cpu")
            # self.val_preds = preds
            # self.val_logits = logits
            self.count = 0


def fit_model(train_nn, train_y, test_nn, params):
    # Device config, sets device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Prints additional device info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    # Creates out of fold tensor to store preds from validation set
    oof_predictions = torch.from_numpy(np.zeros([len(train_nn), 2])).float().to(device)

    # Creates test tensor to store preds from test set
    # Store on CPU to save GPU memory
    test_predictions = (
        torch.from_numpy(np.zeros([len(test_nn), 2])).float().to(torch.device("cpu"))
    )

    # Uses stratified k-fold due to imbalanced data
    kfold = StratifiedKFold(
        n_splits=params["n_fold"], shuffle=True, random_state=params["seed"]
    )

    #
    features_numerical = [
        f for f in train_nn.columns if f != "target" and f != "customer_ID"
    ]
    y_target = train_y.target.to_numpy()
    num_feat = train_nn[features_numerical]
    y = train_y.target
    labels = torch.from_numpy(y.to_numpy()).long().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(np.array([118828, 340085])).float()
    ).to(device)

    # N fold cross validation
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_nn, y_target)):
        print(f"Training fold {fold + 1}")
        x_train, x_val = num_feat.iloc[trn_idx], num_feat.iloc[val_idx]
        y_train, y_val = labels[trn_idx], labels[val_idx]

        scaler = StandardScaler()
        x_train = torch.from_numpy(scaler.fit_transform(x_train)).float().to(device)
        x_val = torch.from_numpy(scaler.transform(x_val)).float().to(device)

        train_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
            np.ones(len(trn_idx)),
            num_samples=len(trn_idx),
            replacement=False,
        )

        train_dataloader = torch.utils.data.DataLoader(
            np.array(range(len(trn_idx))),
            batch_size=params["batch_size"],
            num_workers=0,
            sampler=train_sample_strategy,
            drop_last=False,
        )

        val_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
            np.ones(len(val_idx)),
            num_samples=len(val_idx),
            replacement=False,
        )

        val_dataloader = torch.utils.data.DataLoader(
            np.array(range(len(val_idx))),
            batch_size=params["batch_size"],
            num_workers=0,
            sampler=val_sample_strategy,
            drop_last=False,
        )

        model = FFModel(x_train.shape[1]).to(device)
        # model = my_model(x_train.shape[1]).to(device)
        lr = params["lr"] * np.sqrt(params["batch_size"] / 2048)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=params["wd"])
        lr_scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=[800, 1600, 2400, 3200, 4000, 4800, 5600, 6400, 7200],
            gamma=0.6,
        )

        earlystoper = early_stopper(patience=params["early_stopping"], verbose=True)

        start_epoch = 0
        for epoch in range(start_epoch, params["max_epochs"]):
            train_loss_list = []
            # train_acc_list = []
            model.train()
            for step, input_seeds in enumerate(train_dataloader):
                # Forward step
                batch_inputs = x_train[input_seeds].to(device)
                batch_labels = y_train[input_seeds].to(device)
                train_batch_logits = model(batch_inputs)
                train_loss = criterion(train_batch_logits, batch_labels)

                # Backward Step
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                # tr_batch_pred = None

                if step % 10 == 0:
                    tr_batch_pred = (
                        torch.sum(
                            torch.argmax(train_batch_logits.clone().detach(), dim=1)
                            == batch_labels
                        )
                        / batch_labels.shape[0]
                    )
                    score = (
                        torch.softmax(train_batch_logits.clone().detach(), dim=1)[:, 1]
                        .cpu()
                        .numpy()
                    )
                    print(
                        f"In epoch:{epoch:03d}|batch:{step:04d}, "
                        f"train_loss:{np.mean(train_loss_list):4f}, "
                        f"train_ap:{average_precision_score(batch_labels.cpu().numpy(), score):.4f}, "
                        f"train_acc:{tr_batch_pred.detach():.4f}, "
                        f"train_auc:{roc_auc_score(batch_labels.cpu().numpy(), score):.4f}"
                    )

            # Pred on val set in batches to save GPU memory
            val_loss_list = 0
            val_acc_list = 0
            # val_correct_list = 0
            val_all_list = 0

            model.eval()
            with torch.no_grad():
                for step, input_seeds in enumerate(val_dataloader):
                    batch_inputs = x_val[input_seeds].to(device)
                    batch_labels = y_val[input_seeds].to(device)

                    val_batch_logits = model(batch_inputs)

                    oof_predictions[val_idx[input_seeds]] = val_batch_logits

                    val_loss_list = val_loss_list + criterion(
                        val_batch_logits, batch_labels
                    )

                    val_batch_pred = torch.sum(
                        torch.argmax(val_batch_logits, dim=1) == batch_labels
                    ) / torch.tensor(batch_labels.shape[0])

                    val_acc_list = val_acc_list + val_batch_pred * torch.tensor(
                        batch_labels.shape[0]
                    )

                    val_all_list = val_all_list + batch_labels.shape[0]

                    if step % 10 == 0:
                        score = (
                            torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                                :, 1
                            ]
                            .cpu()
                            .numpy()
                        )

                        print(
                            f"In epoch:{epoch:03d}|batch:{step:04d}, "
                            f"val_loss:{val_loss_list / val_all_list:4f}, "
                            f"val_ap:{average_precision_score(batch_labels.cpu().numpy(), score):.4f}, "
                            f"val_acc:{val_batch_pred.detach():.4f}, "
                            f"val_auc:{roc_auc_score(batch_labels.cpu().numpy(), score):.4f}"
                        )
                # tmp_predictions = model(test_feature).cpu().numpy()
            # infold_preds[fold] = tmp_predictions
            # test_predictions += tmp_predictions / params['n_fold']
            val_predictions = (
                torch.softmax(oof_predictions[val_idx, :].detach(), dim=-1)[:, 1]
                .cpu()
                .numpy()
            )

            earlystoper.earlystop(
                val_loss_list,
                amex_metric(y_val.float().cpu().numpy(), val_predictions),
                model,
            )

            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break

        print(f"Best val_metric is: {earlystoper.best_cv:.7f}")

        # Evaluating on test data
        test_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
            np.ones(len(test_nn)),
            num_samples=len(test_nn),
            replacement=False,
        )

        test_dataloader = torch.utils.data.DataLoader(
            np.array(range(len(test_nn))),
            batch_size=params["batch_size"],
            num_workers=0,
            sampler=test_sample_strategy,
            drop_last=False,
        )

        test_num_feat = (
            torch.from_numpy(scaler.transform(test_nn[features_numerical]))
            .float()
            .to(torch.device("cpu"))
        )

        b_model = earlystoper.best_model.to(torch.device("cpu"))
        b_model.eval()
        with torch.no_grad():
            for step, input_seeds in enumerate(test_dataloader):
                batch_inputs = test_num_feat[input_seeds].to(torch.device("cpu"))

                test_batch_logits = b_model(batch_inputs)

                test_predictions[input_seeds] = test_batch_logits
                # test_batch_pred = torch.sum(torch.argmax(test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])

                # if step % 10 == 0:
                #     print(f"In test batch:{step:04d}")

    # my_acc = acc(y, oof_predictions)
    my_ap = average_precision_score(
        y_target, torch.softmax(oof_predictions, dim=1).cpu()[:, 1]
    )

    print("NN out of fold AP is:", my_ap)

    return (
        earlystoper.best_model.to(torch.device("cpu")),
        oof_predictions,
        test_predictions,
    )


if __name__ == "__main__":
    params = {
        "batch_size": 2048,
        "lr": 0.1,
        "wd": 4e-4,
        #'device': 'cpu',
        "device": "cuda:0",
        "early_stopping": 20,
        "n_fold": 5,
        "seed": 2022,
        "max_epochs": 200,
    }

    b_models, val_nn_0, test_nn_0 = fit_model(
        train_nn=train,
        train_y=pd.DataFrame(target),
        test_nn=test,
        params=params,
    )

    # Empty GPU cache
    torch.cuda.empty_cache()
