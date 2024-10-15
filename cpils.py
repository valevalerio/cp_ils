import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import os
import tempfile

import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from itertools import combinations, chain
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flatten = lambda m: [item for row in m for item in row]
        
class LinearEnc(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearEnc, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.weight = nn.Parameter(nn.init.uniform_(torch.Tensor(latent_dim, input_dim), 
                                                    a=-1./np.sqrt(input_dim),b=1./np.sqrt(input_dim)))

    def encode(self, x):
        z = torch.mm(x, torch.t(self.weight))
        return self.weight, z
    def forward(self, x):
        w, z = self.encode(x)
        return w, z    

def compute_similarity_Z(Z, sigma=1):
    D = 1 - F.cosine_similarity(Z[:, None, :], Z[None, :, :], dim=-1)
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_X(X, idx_cat=None, sigma=1):

    D_class = torch.cdist(X[:,-1].reshape(-1,1), X[:,-1].reshape(-1,1))
    X = X[:, :-1]
    if idx_cat:
        X_cat = X[:, idx_cat]
        X_cont = X[:, np.delete(range(X.shape[1]),idx_cat)]
        h = X_cat.shape[1]
        m = X.shape[1]
        D_cat = torch.cdist(X_cat, X_cat, p=0)/h
        D = h/m * D_cat  + D_class
        if h<m:
            D_cont = 1 - F.cosine_similarity(X_cont[:, None, :], X_cont[None, :, :], dim=-1)
            D += ((m-h)/m) * D_cont
    else:
        D_features = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1) 
        D = D_features + D_class
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def kld_loss_function(X, Z, idx_cat=None, sigma=1):
    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    Sx = compute_similarity_X(X, idx_cat, sigma)
    Sz = compute_similarity_Z(Z, sigma)
    loss = similarity_KLD(torch.log(Sz), Sx)
    return loss

class CP_ILS(torch.nn.Module):
    def __init__(self, bb_predict, bb_predict_proba, latent_dim=2, max_epochs=1000, early_stopping=5, 
                learning_rate=0.001, batch_size=1024, sigma=1):
        super().__init__()

        self.bb_predict = bb_predict
        self.bb_predict_proba = bb_predict_proba

        self.latent_dim=latent_dim

        self.max_epochs=max_epochs
        self.early_stopping=early_stopping

        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.sigma=sigma

    def _predict(self, x, scaler=None, return_proba=False):
        if scaler:
            x = scaler.inverse_transform(x)
        if return_proba:
            return self.bb_predict_proba(x)[:,1].ravel()
        else: 
            return self.bb_predict(x).ravel().ravel()

    def _set(self, X, idx_num_cat, init_path=None):

        self.idx_num_cat = idx_num_cat
        self.idx_cat = flatten([l for l in self.idx_num_cat if len(l)>1])
        if len(self.idx_cat)==0:
            self.idx_cat = None
        self.idx_num = flatten([l for l in self.idx_num_cat if len(l)==1])
        if len(self.idx_num)==0:
            self.idx_num = None

        self.X_train_bb, self.X_test_bb = X
        self.y_train_bb = self._predict(self.X_train_bb, return_proba=True)
        self.y_test_bb = self._predict(self.X_test_bb, return_proba=True)
        
        #idx_num_cat += [[X_train_bb.shape[1]]]

        # if self.idx_num:
        #     self.scaler = MinMaxScaler()
        #     self.X_train = self.scaler.fit_transform(self.X_train_bb[:, self.idx_num])
        #     self.X_test = self.scaler.transform(self.X_test_bb[:, self.idx_num])
        #     if self.idx_cat:
        #         self.X_train = np.hstack((self.X_train, self.X_train_bb[:, self.idx_cat]))
        #         self.X_test = np.hstack((self.X_test, self.X_test_bb[:, self.idx_cat]))
        # else:
        #    self.scaler = None
        #    self.X_train = self.X_train_bb.copy()
        #    self.X_test = self.X_test_bb.copy()

        # self.X_train = np.hstack((self.X_train, self.y_train_bb.reshape(-1,1)))
        # self.X_test = np.hstack((self.X_test, self.y_test_bb.reshape(-1,1)))

        self.scaler = MinMaxScaler()
        self.X_train = np.hstack((self.scaler.fit_transform(self.X_train_bb), self.y_train_bb.reshape(-1,1)))
        self.X_test = np.hstack((self.scaler.transform(self.X_test_bb), self.y_test_bb.reshape(-1,1)))

        self.input_dim = self.X_train.shape[1]

        if 'model' not in self.__dict__['_modules']:

            self.model = LinearEnc(self.input_dim, self.latent_dim).to(device)

        if init_path!=None:
            assert(os.path.isfile(init_path))
            self.model.load_state_dict(torch.load(init_path))

    def load(self, X, idx_num_cat, init_path):

        self._set(X, idx_num_cat, init_path)


    def fit(self, X, idx_num_cat, init_path=None, seed=None):

        self._set(X, idx_num_cat, init_path)

        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_losses, test_losses = self._train()

        self.model = self.model.cpu()

        return train_losses, test_losses

    def _train(self):

        train_dataset = TensorDataset(torch.tensor(self.X_train).float().to(device))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 

        test_dataset = TensorDataset(torch.tensor(self.X_test).float().to(device))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 

        model_params = list(self.model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=self.learning_rate)

        epoch_train_losses = []
        epoch_test_losses = []
        epoch = 1
        best = np.inf

        # progress bar
        pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
            postfix=["Epoch:", {'value':0}, "Train Loss", {'value':0}, "Test Loss", {'value':0}, "Early Stopping", {"value":0}])

        with tempfile.TemporaryDirectory(dir = './') as dname:
            # start training
            while epoch <= self.max_epochs:
                # ------- TRAIN ------- #
                # set model as training mode
                self.model.train()
                batch_loss = []
                for batch, (X_batch,) in enumerate(train_loader):
                    optimizer.zero_grad()

                    W_batch, Z_batch = self.model.encode(X_batch)
                    loss = kld_loss_function(X_batch, Z_batch, self.idx_cat, self.sigma)
                    
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss.append(loss.item())
                # save result
                epoch_train_losses.append(np.mean(batch_loss))
                pbar.postfix[3]["value"] = np.mean(batch_loss)
                # -------- VALIDATION --------
                # set model as testing mode
                self.model.eval()
                batch_loss = []
                for batch, (X_batch,) in enumerate(test_loader):
                    with torch.no_grad():
                        W_batch, Z_batch = self.model.encode(X_batch)
                        loss = kld_loss_function(X_batch, Z_batch, self.idx_cat, self.sigma)

                        batch_loss.append(loss.item())
                # save result
                epoch_test_losses.append(np.mean(batch_loss))
                pbar.postfix[5]["value"] = np.mean(batch_loss)
                pbar.postfix[1]["value"] = epoch 
                if epoch_test_losses[-1] < best:
                    wait = 0
                    best = epoch_test_losses[-1]
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), dname+'/LinearTransparentTemp.pt')
                else:
                    wait += 1
                pbar.postfix[7]["value"] = wait
                if wait == self.early_stopping:
                    break    
                epoch += 1
                pbar.update()
            self.model.load_state_dict(torch.load(dname+'/LinearTransparentTemp.pt'))

        return epoch_train_losses, epoch_test_losses

    def transform(self, X):
        X = np.hstack((X, self._predict(X, return_proba=True).reshape(-1,1)))
        if self.scaler:
            X[:,:-1] = self.scaler.transform(X[:,:-1])
        self.model.eval()
        with torch.no_grad():
            W, Z = self.model.encode(torch.tensor(X).float())
        return W.cpu().detach().numpy(), Z.cpu().detach().numpy() 

    def _compute_cf(self, q, indexes, max_steps=50):
        q_pred = self._predict(q.values, self.scaler, return_proba=True)
        q_cf = q.copy()
        q_cf_preds = []
        q_cf_preds.append(float(self._predict(q_cf.values, self.scaler, return_proba=True)))
        q_cf['prediction'] = q_pred
        if q_pred > 0.5:
            m = -1
        else:
            m = +1
        for iteration in range(max_steps):
            if np.round(q_pred) == np.round(q_cf_preds[-1]):
                # compute the vector to apply
                adapt_coeff = float(abs(q_cf_preds[-1]-0.5))
                with torch.no_grad():
                    w, z = self.model.encode(torch.tensor(q_cf.values).float())
                w, z = w.cpu().detach().numpy(), z.cpu().detach().numpy()
                y_contrib = w[:,-1]/np.linalg.norm(w[:,-1])
                v = (z + m*y_contrib*adapt_coeff).ravel()
                # compute the changes delta in the input space
                c_l = [v[l] - np.sum(q_cf.values*w[l,:]) for l in range(self.latent_dim)]
                M = []
                for l in range(self.latent_dim):
                    M.append([np.sum(w[k,indexes]*w[l,indexes]) for k in range(self.latent_dim)])
                M = np.vstack(M)
                lambda_k = np.linalg.solve(M, c_l)
                delta_i = [np.sum(lambda_k*w[:,i]) for i in indexes]
                q_cf[q_cf.columns[indexes]] += delta_i

                #preserve one-hot encoding
                if self.idx_cat:
                    q_cf.iloc[:, self.idx_cat] = pd.concat([pd.Series((np.arange(q_cf.iloc[:,idx].iloc[0].size)==q_cf.iloc[:,idx].iloc[0].argmax()).astype(int),
                                         index=q_cf.iloc[:,idx].iloc[0].index, name=q_cf.iloc[:,idx].index[0]).to_frame().T
                            for idx in self.idx_num_cat if len(idx)>1], axis=1)

                #preserve minmax(0,1)
                if self.idx_num:
                    q_cf.iloc[:, self.idx_num] = np.clip(q_cf.iloc[:, self.idx_num], 0, 1)
                
                # check changes or null effects in the prediction
                if float(self._predict(q_cf.iloc[:,:-1].values, self.scaler, return_proba=True)) in q_cf_preds:
                    return q_cf.iloc[:,:-1]
                
                q_cf_preds.append(float(self._predict(q_cf.iloc[:,:-1].values, self.scaler, return_proba=True)))
                q_cf[q_cf.columns[-1]] = q_cf_preds[-1]
            else:
                break
        return q_cf.iloc[:,:-1]

    def _cdist(self, XA, XB, metric=('euclidean', 'jaccard'), w=None):
        metric_continuous = metric[0]
        metric_categorical = metric[1]

        if self.idx_cat:
            dist_categorical = cdist(XA[:, self.idx_cat], XB[:, self.idx_cat],
                                 metric=metric_categorical, w=w)
            ratio_categorical = len(self.idx_cat) / (self.input_dim-1)
            dist = ratio_categorical * dist_categorical

            if self.idx_num:
                dist_continuous = cdist(XA[:, self.idx_num], XB[:, self.idx_num],
                                    metric=metric_continuous, w=w)
                ratio_continuous = len(self.idx_num) / (self.input_dim-1)
                dist += ratio_continuous * dist_continuous 
        else:
            dist = cdist(XA, XB, metric=metric_continuous, w=w)

        return dist

    def _greedy_kcover(self, x, cf_list_all, k=5, lambda_par=1.0, submodular=True, knn_dist=True):

        def selected_cf_distance(x, selected, lambda_par=1.0, knn_dist=False, knn_list=None, lconst=None):

            if not knn_dist:
                dist_ab = 0.0
                dist_ax = 0.0
                for i in range(len(selected)):
                    a = np.expand_dims(selected[i], 0)
                    for j in range(i + 1, len(selected)):
                        b = np.expand_dims(selected[j], 0)
                        dist_ab += self._cdist(a, b)[0][0]
                    dist_ax += self._cdist(a, x)[0][0]

                coef_ab = 1 / (len(selected) * len(selected)) if len(selected) else 0.0
                coef_ax = lambda_par / len(selected) if len(selected) else 0.0

            else:
                dist_ax = 0.0
                common_cfs = set()
                for i in range(len(selected)):
                    a = np.expand_dims(selected[i], 0)
                    knn_a = knn_list[a.tobytes()]
                    common_cfs |= knn_a
                    dist_ax +=  self._cdist(a, x)[0][0]
                dist_ab = len(common_cfs)

                coef_ab = 1.0
                coef_ax = 2.0 * lconst

            dist = coef_ax * dist_ax - coef_ab * dist_ab
            # dist = coef_ab * dist_ab - coef_ax * dist_ax
            return dist

        def get_best_cf(x, selected, cf_list_all, lambda_par=1.0, submodular=True, knn_dist=False, knn_list=None, lconst=None):

            min_d = np.inf
            best_i = None
            best_d = None
            d_w_a = selected_cf_distance(x, selected, lambda_par, knn_dist, knn_list, lconst)
            for i, cf in enumerate(cf_list_all):
                d_p_a = selected_cf_distance(x, selected + [cf], lambda_par)
                d = d_p_a - d_w_a if submodular else d_p_a  # submudular -> versione derivata
                if d < min_d:
                    best_i = i
                    best_d = d_p_a
                    min_d = d

            return best_i, best_d

    #   x = np.expand_dims(x, 0)
    #   nx = scaler.inverse_transform(x)
        nx = x.reshape(1, -1)

    #   ncf_list_all = scaler.transform(cf_list_all)
        ncf_list_all = cf_list_all.copy()

        lconst = None
        knn_list = None
        if knn_dist:
            dist_x_cf = self._cdist(nx, ncf_list_all)
            d0 = np.argmin(dist_x_cf)
            lconst = 0.5 / (-d0) if d0 != 0.0 else 0.5

            # cf_dist_matrix = np.mean(self.cdist(ncf_list_all, ncf_list_all,
            #                                     metric='euclidean', w=None), axis=0)
            cf_dist_matrix = self._cdist(ncf_list_all, ncf_list_all)

            knn_list = dict()
            for idx, knn in enumerate(np.argsort(cf_dist_matrix, axis=1)[:, 1:k+1]):
                cf_core_key = np.expand_dims(cf_list_all[idx], 0).tobytes()
                knn_set = set([np.expand_dims(cf_list_all[nn], 0).tobytes() for nn in knn])
                knn_list[cf_core_key] = knn_set

        cf_list = list()
        cf_selected = list()
        ncf_selected = list()
        min_dist = np.inf
        while len(ncf_selected) < k:
            idx, dist = get_best_cf(nx, ncf_selected, ncf_list_all, lambda_par, submodular,
                                         knn_dist, knn_list, lconst)
            #cf_selected.append(self.scaler.inverse_transform(ncf_list_all[idx].reshape(1, -1)))
            cf_selected.append(ncf_list_all[idx].copy())
            ncf_selected.append(ncf_list_all[idx].copy())
            ncf_list_all = np.delete(ncf_list_all, idx, axis=0)
            if dist < min_dist:
                min_dist = dist
                cf_list = cf_selected

        cf_list = np.array(cf_list)

        return cf_list


    def get_counterfactuals(self, df_test, features_to_change, max_features_to_change, 
                                max_steps=50, n_cfs=-1, n_feats_sampled=5, topn_to_check=5, seed=42):

        rng = np.random.default_rng(seed)

        self.model.eval()
        
        all_cfs = []
        
        for _, row in tqdm(list(df_test.iterrows())):
            
            q_cfs = []

            q = row.to_frame().T  
            q.iloc[:, :] = self.scaler.transform(q.values)

            q_pred = self._predict(q.values, self.scaler, return_proba=False)
            s_i = [set()]
            s_f = set()
            l_i = []
            l_f = []
            
            ########

            for indexes in list(combinations(list(features_to_change),1)):    
                q_cf = self._compute_cf(q, list(indexes), max_steps)
                q_cf_pred = self._predict(q_cf.values, self.scaler, return_proba=True)
                diff_probs = float(abs(q_cf_pred-0.5))
                if q_pred:
                    if q_cf_pred<0.5:
                        q_cfs.append(q_cf)
                        s_i[-1].add(frozenset(list(indexes)))
                    else:
                        l_i.append((list(indexes), diff_probs))
                else:
                    if q_cf_pred>0.5:
                        q_cfs.append(q_cf)
                        s_i[-1].add(frozenset(list(indexes)))
                    else:
                        l_i.append((list(indexes), diff_probs))
                        
            if len(l_i)>0:
                
                r = np.argsort(np.stack(np.array(l_i,dtype=object)[:,1]).ravel())[:topn_to_check]
                l_i = np.array(l_i,dtype=object)[r,0]
            
                while len(l_i[0])<max_features_to_change:
                    for e in l_i:
                        features_to_check = list(np.delete(features_to_change, 
                                 list(map(lambda f: (features_to_change).index(f), e))))

                        for i in rng.choice(features_to_check, 
                                                  size=min(len(features_to_check), n_feats_sampled), replace=False):

                            indexes = list(e)+[i]

                            skip_i = False
                            #check if the current indices already returned a cf
                            if frozenset(indexes) in s_f:
                                skip_i = True

                            if not skip_i:
                                #check if any subset of current indices already returned a cf
                                for comb_i in chain.from_iterable(combinations(indexes, r) 
                                                                  for r in range(1, len(indexes))):
                                    if frozenset(comb_i) in s_i[len(comb_i)-1]:
                                        skip_i = True
                                        break

                            if not skip_i:
                                q_cf = self._compute_cf(q, list(indexes), max_steps)
                                q_cf_pred = self._predict(q_cf.values, self.scaler, return_proba=True)
                                diff_probs = float(abs(q_cf_pred-0.5))
                                if q_pred:
                                    if q_cf_pred<0.5:
                                        q_cfs.append(q_cf)
                                        s_f.add(frozenset(indexes))
                                    else:
                                        l_f.append((list(indexes), diff_probs))
                                else:
                                    if q_cf_pred>0.5:
                                        q_cfs.append(q_cf)
                                        s_f.add(frozenset(indexes))
                                    else:
                                        l_f.append((list(indexes), diff_probs))
                    
                    if len(l_f)==0:
                        break
                        
                    s_i.append(s_f.copy())
                    s_f = set()

                    r = np.argsort(np.stack(np.array(l_f,dtype=object)[:,1]).ravel())[:topn_to_check]
                    l_f = np.array(l_f,dtype=object)[r,0]
                    l_i = l_f.copy()
                    l_f = []
                
            if len(q_cfs)==0:
                all_cfs.append(pd.DataFrame(None, columns=q.columns))
            else:
                #q_cfs = [pd.Series(self.scaler.inverse_transform(cf)[0], index=q.columns, name=q.index[0]).to_frame().T for cf in q_cfs]
                q_cfs = pd.concat(q_cfs).drop_duplicates()
                if n_cfs > -1:
                    if len(q_cfs)>n_cfs:
                        cf_list = self._greedy_kcover(q.values, q_cfs.values.squeeze(), k=n_cfs).squeeze()
                        q_cfs = [pd.Series(cf, index=q.columns, name=q.index[0]).to_frame().T for cf in cf_list[:n_cfs]]
                        q_cfs = pd.concat(q_cfs)
                q_cfs.iloc[:,:] = self.scaler.inverse_transform(q_cfs.iloc[:,:])
                all_cfs.append(q_cfs)
        
        return pd.concat(all_cfs)

    def get_prototypes(self, df_test, n_proto=20, seed=42):

        rng = np.random.default_rng(seed)

        self.model.eval()
        with torch.no_grad():
            _, Z_train = self.model.encode(torch.tensor(self.X_train).float())

        Z_train = Z_train.cpu().detach().numpy()

        Z_train_0 = Z_train[np.round(self.y_train_bb)==0]
        Z_train_1 = Z_train[np.round(self.y_train_bb)==1]

        ncls0 = int(np.round(n_proto*(sum(np.round(self.y_train_bb)==0)/len(self.y_train_bb))))
        ncls1 = n_proto - ncls0

        if ncls0==0:
            ncls0+=1
            ncls1-=1
        elif ncls1==0:
            ncls0-=1
            ncls1+=1

        clustering_0 = KMeans(n_clusters=ncls0, random_state=seed).fit(Z_train_0)
        clustering_1 = KMeans(n_clusters=ncls1, random_state=seed).fit(Z_train_1)
        centers = np.concatenate((clustering_0.cluster_centers_, clustering_1.cluster_centers_))

        idx_latent = np.argmin(cdist(centers, Z_train), axis=1)
        proto_latent = pd.DataFrame(self.X_train[idx_latent][:,:-1], columns=df_test.columns)
        proto_pred = self._predict(proto_latent.values, self.scaler, return_proba=False)

        idx_proto = np.arange(proto_latent.shape[0])
        x_pred = self._predict(df_test.values, return_proba=False)
        
        knn_1 = [np.argmin(self._cdist(proto_latent.values[proto_pred==x_pred[i]], 
                                       self.scaler.transform(df_test.values[i].reshape(1,-1))), axis=0)
                    for i in range(df_test.shape[0])] # 
        knn_1 = np.array([idx_proto[proto_pred==x_pred[i]][kn] for i, kn in enumerate(knn_1)]).ravel() # 

        proto_latent.iloc[:,:] = self.scaler.inverse_transform(proto_latent.iloc[:,:])
        
        return proto_latent, proto_latent.iloc[knn_1].set_index(df_test.index)
        
        



