import torch
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import numpy as np
from icecream import ic
from copy import deepcopy
from hygeia.data.split import split_anndata
import anndata as ad
import os

class DatasetLoader:
    def __init__(
        self,
        path,
        label_keys,
        default_normalization=False,
        min_gene_counts=False,
        min_cell_counts=False,
        n_highly_variable_genes=False,
        use_counts=False,
        seed=42,
        subset=False,
        ):  

        self.seed=seed
        #torch.manual_seed(self.seed)

        self.path = os.path.expandvars(path)
        self.label_keys = label_keys
        self.min_gene_counts = min_gene_counts
        self.min_cell_counts = min_cell_counts
        self.n_highly_variable_genes = n_highly_variable_genes
        self.use_counts = use_counts

        self.label_mapping = {}
        self.reverse_label_mapping = {}
        self.cell_id_label_mapping = {}
        self.label_data_by_key = {}

        self.data = sc.read_h5ad(self.path)
        if subset is not False:
            n_samples = self.data.shape[0]
            subset_size = int(subset * n_samples)

            # Generate random indices
            random_indices = np.random.choice(n_samples, subset_size, replace=False)

            # Subset the AnnData object
            self.data = self.data[random_indices, :]

        self.filter_cells_and_genes()
    
        if 'counts' not in self.data.layers.keys():
            ic('Counts not found, capturing counts from X, make sure it makes sense')
            self.data.layers['counts'] = self.to_numpy(self.data.X.copy())

        else:
            self.data.layers['counts'] = self.to_numpy(self.data.layers['counts'])

        if default_normalization:
            self.default_normalization()

        self._filter_highly_variable_genes()

        self.cell_id_strings = self.data.obs.index.tolist()
        self.cell_ids = list(range(len(self.cell_id_strings)))

        self._create_cell_id_label_mapping()

        if self.use_counts:
            self.data.X = self.to_numpy(self.data.layers['counts'].copy())
        
        # Temporary stuff
        
        self.data.obs['cell_ids'] = self.cell_ids
        
    def to_numpy(self, arr):
        if isinstance(arr, np.ndarray):
            return arr 
        else:
            return arr.toarray()

    def get_labels_from_ids(self, id_tensor, label_key):
        labels_list = [self.cell_id_label_mapping[int(i.item())][label_key] for i in id_tensor]
        return torch.tensor(labels_list, dtype = torch.long)

    def _create_cell_id_label_mapping(self):
        
        ic('Creating cell mappings')
        for key in self.label_keys:
            label_series = self.data.obs[key].astype('category')
          
            self.label_data_by_key[key] = torch.tensor(label_series.cat.codes.values, dtype=torch.long)
            self.label_mapping[key] = dict(enumerate(label_series.cat.categories.tolist()))

            self.reverse_label_mapping[key] = {v:k for k,v in self.label_mapping[key].items()}

        for idx, cell_id in enumerate(self.cell_ids):
            self.cell_id_label_mapping[cell_id] = {}
            for key in self.label_keys:
                self.cell_id_label_mapping[cell_id][key] = self.label_data_by_key[key][idx].item() 

    def get_inputs(self, anndata):
        ic('Creating inputs')

        X = torch.tensor(self.to_numpy(anndata.X), dtype=torch.float32)
        cell_ids = torch.tensor(anndata.obs['cell_ids'], dtype=torch.int64)

        inputs_to_return = [cell_ids] + [X]

        return inputs_to_return

    def train_val_test(self, test_size, val_size, batch_size):

        self.data = split_anndata(self.data, test_size=test_size, val_size=val_size, seed=self.seed)

        train_inputs = self.get_inputs(self.data[self.data.obs['split_anndata'] == 'train'])

        if test_size != 0:
            test_inputs = self.get_inputs(self.data[self.data.obs['split_anndata'] == 'test'])
        else:
            test_inputs = deepcopy(train_inputs)

        if val_size != 0:
            val_inputs = self.get_inputs(self.data[self.data.obs['split_anndata'] == 'val'])
        else:
            val_inputs = deepcopy(train_inputs)

        self.train_anndata = self.data[self.data.obs['split_anndata'] == 'train'].copy()
        self.test_anndata = self.data[self.data.obs['split_anndata'] == 'test'].copy()
        train_dataset = TensorDataset(*train_inputs)
        test_dataset = TensorDataset(*test_inputs)
        val_dataset = TensorDataset(*val_inputs)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train_val_ood_key(
        self,
        batch_size,
        split_key,
        add_ctrl_dataloader=None,
        ctrl_name=None,
        cond_key=None,
        add_small_dataloader=False,
        ):

        train_anndata = self.data[self.data.obs[split_key] == 'train']
        val_anndata = self.data[self.data.obs[split_key] == 'val']
        test_anndata = self.data[self.data.obs[split_key] == 'ood']

        if len(test_anndata) == 0:
            print('Test is empty, setting it to val')
            test_anndata = val_anndata.copy()

        if add_ctrl_dataloader:
            ctrl_anndata = train_anndata[train_anndata.obs[cond_key] == ctrl_name]
            ctrl_inputs = self.get_inputs(ctrl_anndata)
            ctrl_dataset = TensorDataset(*ctrl_inputs)
            ctrl_dataloader = DataLoader(ctrl_dataset, batch_size=2056, shuffle=False)
            self.ctrl_dataloader = ctrl_dataloader
        else:
            self.ctrl_dataloader = None

        train_inputs = self.get_inputs(train_anndata)
        val_inputs = self.get_inputs(val_anndata)
        test_inputs = self.get_inputs(test_anndata)

        train_dataset = TensorDataset(*train_inputs)
        test_dataset = TensorDataset(*test_inputs)
        val_dataset = TensorDataset(*val_inputs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)


        ic(train_anndata.shape, test_anndata.shape, self.data.shape)
      
        self.train_anndata = train_anndata
        self.val_anndata = val_anndata
        self.test_anndata = test_anndata

        if add_small_dataloader:
            ic('ADDING SMALL DATALOADER')
            cat_adata = ad.concat([test_anndata, ctrl_anndata])
            cat_inputs = self.get_inputs(cat_adata)
            cat_dataset = TensorDataset(*cat_inputs)
            ctrl_dataloader = DataLoader(cat_dataset)
            self.ctrl_dataloader = ctrl_dataloader
            ic(len(cat_dataset))
            ic(len(test_anndata))
            ic(len(ctrl_anndata))
            
        if self.ctrl_dataloader is not None:
            return train_loader, val_loader, test_loader, ctrl_dataloader
        else:
            return train_loader, val_loader, test_loader

    def get_dataloader(self, batch_size):
        _inputs = self.get_inputs(self.data)
        _dataset = TensorDataset(*_inputs)
        _dataloader = DataLoader(_dataset, batch_size=batch_size, shuffle=False)

        return _dataloader
        
    def train_val_ood(
        self,
        train_size,
        batch_size,
        filter_dict,
        ):

        ic('Train val OOD', train_size, filter_dict)

        if 'filter_single' in filter_dict:
            print('FITLERING SINGLE')
            #if filter_dict['filter_single']['apply']:
            condition_name = filter_dict['filter_single']['cond_name']
            labels = filter_dict['filter_single']['labels']
            mask_test = self.data.obs[condition_name].isin(labels)

        else:
     
            mask_test = ''
            for condition_name, value in filter_dict.items():
                if condition_name == 'filter_single':
                    print('continue')
                    continue
                mask_test += f"(self.data.obs['{condition_name}'] == '{value}') & "

            mask_test = mask_test[:-3]
            mask_test = eval(mask_test)
            
        # mask_train is the inverse of mask_test
        mask_train = ~mask_test
        ic(mask_train.sum(), mask_test.sum())

        train_anndata = self.data[mask_train]
        test_anndata = self.data[mask_test]

        val_size = 1 - train_size
        train_anndata = split_anndata(train_anndata, test_size=0, val_size=val_size, seed=self.seed)

        train_inputs = self.get_inputs(train_anndata[train_anndata.obs['split_anndata'] == 'train'])
        val_inputs = self.get_inputs(train_anndata[train_anndata.obs['split_anndata'] == 'val'])
        test_inputs = self.get_inputs(test_anndata)

        train_dataset = TensorDataset(*train_inputs)
        test_dataset = TensorDataset(*test_inputs)
        val_dataset = TensorDataset(*val_inputs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        ic(train_anndata.shape, test_anndata.shape, self.data.shape)

        self.train_anndata = train_anndata
        self.test_anndata = test_anndata

        return train_loader, val_loader, test_loader

    def default_normalization(self):
        print('Default normalization (total + log1p)')
        #sc.pp.normalize_total(self.data)

        sc.pp.normalize_total(self.data, target_sum=1e4, exclude_highly_expressed=True)
        #sc.pp.normalize_total(self.data)
        sc.pp.log1p(self.data)

    def filter_cells_and_genes(self):
        
        if self.min_cell_counts:
            ic('Filter cells', self.min_cell_counts)
            print('Filtering cells')
            sc.pp.filter_cells(self.data, min_counts=self.min_cell_counts)

        if self.min_gene_counts:
            ic('Filter genes', self.min_gene_counts)
            print('Filtering genes')
            sc.pp.filter_genes(self.data, min_counts=self.min_gene_counts)

    def _filter_highly_variable_genes(self):
        if self.n_highly_variable_genes:
            sc.pp.highly_variable_genes(self.data, n_top_genes=self.n_highly_variable_genes)
            ic('Selected highly', self.n_highly_variable_genes)
            self.data = self.data[:, self.data.var['highly_variable']]
