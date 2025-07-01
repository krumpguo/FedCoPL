import argparse
import logging
import os
import sys
from logging import StreamHandler
from pathlib import Path
from collections import Counter
import copy
import time
import os.path as osp
import torch
import yaml
from accelerate import Accelerator
import numpy as np
from methods.unsupervised_learning_new import (
    TextualFPL_PL,
    FedCoPL
)

from utils import (
    Config,
    set_obj_conf,
    dataset_object,
    get_class_names,
    get_labeled_and_unlabeled_data,
    become_deterministic,
    monitor_and_accelerate,
    InstanceSelector
)
 
import torch.nn.functional as F

torch.set_num_threads(2) #NOTE To maximize efficiency, please tune the number of threads for your machine
accelerator = Accelerator()

logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")



class AccelerateHandler(StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        if accelerator.is_local_main_process:
            super().emit(record)

stream_handler = AccelerateHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)


#============= CPL Workflow =============
def prepare_dataset_UL(classes, labeled_data, unlabeled_data, test_data):
    """
    Prepare datasets for Unsupervised Learning (UL).

    Parameters:
    classes (List[str]): List of class names.
    labeled_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.
    unlabeled_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.
    test_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.

    Returns:
    Tuple[List[str], List[int], List[str], List[int], Dict[str, int]]: Returns a tuple containing lists of test files, 
    test labels, train files, train labels and a dictionary mapping class names to indices.
    """
    labeled_files, labeles = zip(*labeled_data)
    # unseen_labeled_files, unseen_labeles = zip(*unlabeled_data)     #unlabeled == unseen
    # test_labeled_files, test_labeles = zip(*test_data)
    
    # define datasets for UL:
    UL_test_files, UL_test_lbs = zip(*test_data)
    UL_train_files =  labeled_files # for UL we use all the trian data
    UL_train_lbs_true =  labeles
    label_to_idx = {c: idx for idx, c in enumerate(classes)}
    
    return (UL_test_files, UL_test_lbs, 
            UL_train_files, UL_train_lbs_true, 
            label_to_idx)

def prepare_dataset_UL_fed(classes, labeled_data, unlabeled_data, test_data, n_parties = 10, partition='iid', beta=0.1):
    """
    Prepare datasets for Unsupervised Learning (UL).

    Parameters:
    classes (List[str]): List of class names.
    labeled_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.
    unlabeled_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.
    test_data (List[Tuple[str, int]]): List of tuples where each tuple contains a file path and its corresponding label.

    Returns:
    Tuple[List[str], List[int], List[str], List[int], Dict[str, int]]: Returns a tuple containing lists of test files, 
    test labels, train files, train labels and a dictionary mapping class names to indices.
    """
    labeled_files, labeles = zip(*labeled_data)
    # unseen_labeled_files, unseen_labeles = zip(*unlabeled_data)     #unlabeled == unseen
    # test_labeled_files, test_labeles = zip(*test_data)
    
    # define datasets for UL:
    UL_test_files, UL_test_lbs = zip(*test_data)
    UL_train_files =  labeled_files # for UL we use all the trian data
    UL_train_lbs_true = labeles
    label_to_idx = {c: idx for idx, c in enumerate(classes)}

    UL_train_lbs_true_id = [0] * len(UL_train_lbs_true)
    UL_test_lbs_true_id = [0] * len(UL_test_lbs)
    for i in range(len(UL_train_lbs_true)):
        UL_train_lbs_true_id[i] = label_to_idx[UL_train_lbs_true[i]]
    for i in range(len(UL_test_lbs)):
        UL_test_lbs_true_id[i] = label_to_idx[UL_test_lbs[i]]        
    n_train = len(UL_train_files)
    n_test = len(UL_test_lbs)
    UL_train_files_fed = {i: [] for i in range(n_parties)}
    UL_train_lbs_true_fed = {i: [] for i in range(n_parties)}
    UL_train_lbs_true_fed_id = {i: [] for i in range(n_parties)}
    UL_test_files_fed = {i: [] for i in range(n_parties)}
    UL_test_lbs_true_fed = {i: [] for i in range(n_parties)}
    UL_test_lbs_true_fed_id = {i: [] for i in range(n_parties)}
    if partition == "homo" or partition == "iid":
        min_size = 0
        min_require_size = 10
        K = len(classes)

        N =len(UL_train_files) #total number of samples
        N_test =len(UL_test_files) #total number of samples
        party2dataidx = {}
        party2dataidx_test = {}

        idx_batch = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = [idx for idx, value in enumerate(UL_train_lbs_true_id) if value == k]
            idx_k_test = [idx for idx, value in enumerate(UL_test_lbs_true_id) if value == k]
            np.random.shuffle(idx_k)
            np.random.shuffle(idx_k_test)
            proportions = np.full(n_parties, float(1 / n_parties))
            proportions_train = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions) * len(idx_k_test)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            party2dataidx[j] = idx_batch[j]
            np.random.shuffle(idx_batch_test[j])
            party2dataidx_test[j] = idx_batch_test[j]

    elif partition == "noniid-labeldir" : 
        min_size = 0 
        party2dataidx = {}
        party2dataidx_test = {}
        least_samples =10
        min_require_size = 10
        num_classes = len(classes)
        class_per_client = num_classes * 0.2
        idxs = np.array(range(len(UL_train_files)))
        idx_for_each_class = []
        idx_for_each_class_test = []
        for k in range(len(classes)):
            idx_k = [idx for idx, value in enumerate(UL_train_lbs_true_id) if value == k]
            idx_k_test = [idx for idx, value in enumerate(UL_test_lbs_true_id) if value == k]
            idx_for_each_class.append(idx_k)
            idx_for_each_class_test.append(idx_k_test)

        class_num_per_client = [class_per_client for _ in range(n_parties)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(n_parties):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(n_parties/num_classes*class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients

            num_all_samples_test = len(idx_for_each_class_test[i])
            num_per_test = num_all_samples_test / num_selected_clients


            # if balance:
            #     num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            # else:
                # if dataset == 'cifar10':
                #     num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_all_samples, num_selected_clients-1).tolist()
                # else: 
            num_samples = np.random.randint(max(num_per/n_parties, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            num_samples_test = np.random.randint(max(num_per_test/n_parties, least_samples/num_classes), num_per_test, num_selected_clients-1).tolist()
            num_samples_test.append(num_all_samples_test-sum(num_samples_test))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in party2dataidx.keys():
                    party2dataidx[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    party2dataidx[client] = np.append(party2dataidx[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1
                
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples_test):
                if client not in party2dataidx_test.keys():
                    party2dataidx_test[client] = idx_for_each_class_test[i][idx:idx+num_sample]
                else:
                    party2dataidx_test[client] = np.append(party2dataidx_test[client], idx_for_each_class_test[i][idx:idx+num_sample], axis=0)
                idx += num_sample


    elif partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = len(classes)

        N =len(UL_train_files) #total number of samples
        N_test =len(UL_test_files) #total number of samples
        party2dataidx = {}
        party2dataidx_test = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = [idx for idx, value in enumerate(UL_train_lbs_true_id) if value == k]
                idx_k_test = [idx for idx, value in enumerate(UL_test_lbs_true_id) if value == k]
                np.random.shuffle(idx_k)
                np.random.shuffle(idx_k_test)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions_train = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions_test = np.array([p * (len(idx_j) < N_test / n_parties) for p, idx_j in zip(proportions, idx_batch_test)])
                proportions_train = proportions_train / proportions_train.sum()
                proportions_test = proportions_test / proportions_test.sum()
                proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            party2dataidx[j] = idx_batch[j]
            np.random.shuffle(idx_batch_test[j])
            party2dataidx_test[j] = idx_batch_test[j]

    datadistribution_train = np.zeros((n_parties, len(classes), 2))
    datadistribution_test = np.zeros((n_parties, len(classes), 2))
    for j in range(n_parties): 
        for i in range(len(party2dataidx[j])):       
            UL_train_files_fed[j].append(UL_train_files[party2dataidx[j][i]])
            UL_train_lbs_true_fed[j].append(UL_train_lbs_true[party2dataidx[j][i]])
            UL_train_lbs_true_fed_id[j].append(UL_train_lbs_true_id[party2dataidx[j][i]])
        count_dict = Counter(UL_train_lbs_true_fed_id[j])
        for num in range(len(classes)):
            datadistribution_train[j][num][0] = num
            datadistribution_test[j][num][0] = num
        for num, count in count_dict.items():
            datadistribution_train[j][num][1] = count
        for k in range(len(party2dataidx_test[j])):  
            UL_test_files_fed[j].append(UL_test_files[party2dataidx_test[j][k]])
            UL_test_lbs_true_fed[j].append(UL_test_lbs[party2dataidx_test[j][k]])  
            UL_test_lbs_true_fed_id[j].append(UL_test_lbs_true_id[party2dataidx_test[j][k]])  
        count_dict = Counter(UL_test_lbs_true_fed_id[j])   
        for num, count in count_dict.items():
            
            datadistribution_test[j][num][1] = count

    log.info(f"\n----------------------Train data distribution-----------------------\n")
    log.info(datadistribution_train)
    log.info(f"\n----------------------Test data distribution-----------------------\n")
    log.info(datadistribution_test)

    return (UL_test_files, UL_test_lbs, UL_test_files_fed, UL_test_lbs_true_fed, 
            UL_train_files_fed, UL_train_lbs_true_fed, 
            label_to_idx)    

def average_weights(w,client_num,fed_avg_freqs,islist=False):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for idx in range(client_num):

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs[0]
            else:
                w_avg += w[idx] * fed_avg_freqs[idx]
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs[0]
            else:
                for key in w_avg:
                    w_avg[key] += w[idx][key] * fed_avg_freqs[idx]

    return w_avg

def workflow_new(dataset_dir, obj_conf):
    # Get dataset name
    # We get the dataset name from the dev_config.py
    os.environ["CUDA_VISIBLE_DEVICES"] = obj_conf.device_id

    dataset = obj_conf.DATASET_NAME
    train_dataset_fed = {}
    train_dataset_fed_use = {}
    test_dataset_fed = {}
    # Get class names of target task
    # define function for each dataset
    classes, seen_classes, unseen_classes = get_class_names(dataset, dataset_dir)
    # Create dict classes to pass as variable
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")

    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get labeled data (seen classes)
    # Get unlabeled data (unseen classes)
    # Get test data (both seen and unseen classes)
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    )
    # 1. Create datasets
    (UL_test_files, UL_test_lbs, 
    UL_train_files, UL_train_lbs_true, 
    label_to_idx) = prepare_dataset_UL(classes, labeled_data, unlabeled_data, test_data)

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    
    train_dataset = DatasetObject(
    UL_train_files,       #NOTE here we use all the train data in UL
    data_folder,
    transform=None,                 
    augmentations=None,
    train=True,
    labels=UL_train_lbs_true,
    label_map=label_to_idx)

    (UL_test_files, UL_test_lbs, UL_fed_test_files, UL_fed_test_lbs, 
    UL_fed_train_files, UL_fed_train_lbs_true, 
    label_to_idx) = prepare_dataset_UL_fed(classes, labeled_data, unlabeled_data, test_data, n_parties=obj_conf.client_num,  partition=obj_conf.partition, beta=obj_conf.beta)

    for client_idx in range(obj_conf.client_num):
        train_dataset_fed[client_idx] = DatasetObject(
            UL_fed_train_files[client_idx],       #NOTE here we use all the train data in UL
            data_folder,
            transform=None,                 
            augmentations=None,
            train=True,
            labels=UL_fed_train_lbs_true[client_idx],
            label_map=label_to_idx,
        )
        test_dataset_fed[client_idx] = DatasetObject(
            UL_fed_test_files[client_idx],       #NOTE here we use all the train data in UL
            data_folder,
            transform=None,                 
            augmentations=None,
            train=False,
            labels=UL_fed_test_lbs[client_idx],
            label_map=label_to_idx,
        )
        train_dataset_fed_use[client_idx] = copy.deepcopy(train_dataset_fed[client_idx])
    # Test set (test seen and unseen)
    test_dataset = DatasetObject(
        UL_test_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=UL_test_lbs,
        label_map=label_to_idx,
    )
    
    # Define model
    device = "cuda:" + str(obj_conf.device_id) if torch.cuda.is_available() else "cpu"
    log.info(f"\n----------------------MODEL INFO-----------------------\n")    

    if obj_conf.MODEL == "our_dual_local_prompt":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        local_trainer = FedCoPL(
            obj_conf, 
            label_to_idx, 
            device=device, 
            **dict_classes
        )
        local_trainer.define_model(obj_conf, local_trainer.clip_model, classes)
        initial_global_weights = copy.deepcopy(local_trainer.model.state_dict())

        global_weights = {}
        local_prompt_weights_per = [{} for i in range(obj_conf.client_num)]
        initial_learning_rate = copy.deepcopy(obj_conf.LR)

        uploaded_models_0 = [[] for i in range(obj_conf.client_num)]
        uploaded_models_1 = [[] for i in range(obj_conf.client_num)]

        train_probs_pres = {}
        train_probs = {}
        train_output_logits = {}

        for client in range(obj_conf.client_num):
            train_probs_pres[client] = []
            train_probs[client] = []
            train_output_logits[client] = []            
        for round in range(obj_conf.round):
            log.info(f"------------------------Global Round {round}--------------------------------")
            if round % obj_conf.num_repesudo_round == 0:
                log.info(f"*************************generate pesudolabels for every client******************************")
                train_dataset_fed_use = {}
                pesudo_label_acc_pre = []
                local_selector_pre = []
                data_distributions = []
                local_prototypes = []
                local_sample_per_class = []          
                local_prototypes_use = []
                local_filepaths = []
                local_probs = [] 
                local_output_logits = []
                local_initial_features = []
                true_label_distributions = [] ##
                pesudo_label_distributions= [] ##
                selected_data_distributions = [] ##
                true_label_of_selected_data_distributions = [] ##

                for client in range(obj_conf.client_num):
                    if round == 0:
                        local_trainer.model.load_state_dict(initial_global_weights,strict=False)
                    else:
                        local_trainer.model.load_state_dict(global_weights,strict=False) # yihui buhuilai 
                        local_trainer.model.load_state_dict(local_prompt_weights_per[client],strict=False)  # yihui buhuilai 
                    local_selector_pre.append(InstanceSelector(label_to_idx=label_to_idx, cfg=obj_conf, device=device))
                    monitor_and_accelerate(UL_fed_train_lbs_true[client], train_dataset_fed[client], local_trainer, local_selector_pre[client])

                    train_dataset_fed_use[client] = copy.deepcopy(train_dataset_fed[client])
                    local_prototype, data_distribution, acc, filepaths, probs, output_logits, initial_features = local_trainer._create_training_dataset_pre(
                    train_dataset_fed_use[client], copy.deepcopy(train_dataset_fed_use[client]), 
                    iter_num= int(round/obj_conf.num_repesudo_round), Selector=local_selector_pre[client])
                    local_filepaths.append(filepaths)
                    local_probs.append(probs)
                    local_output_logits.append(output_logits)
                    local_initial_features.append(initial_features)
                    pesudo_label_acc_pre.append(acc)
                    data_distributions.append(data_distribution)
                    local_prototypes.append(copy.deepcopy(local_prototype))
                    mask = data_distribution != 0
                    expanded_mask = mask.view(local_prototype.shape[0], 1).expand(local_prototype.shape[0], local_prototype.shape[1]).float()
                    expanded_mask = expanded_mask.bool()
                    local_prototypes_use.append(copy.deepcopy(local_prototype / torch.where(expanded_mask, local_prototype.norm(dim=-1, keepdim=True).view(local_prototype.shape[0], 1), torch.ones_like(data_distribution).view(local_prototype.shape[0], 1))))

                total_local_prototypes = torch.stack(local_prototypes).sum(dim=0)
                total_data_distributions = torch.stack(data_distributions).sum(dim=0)

                avg_pesudo_label_acc_pre = sum(pesudo_label_acc_pre) / len(pesudo_label_acc_pre)
                log.info(f"------------------------pre pesudolabels acc of all clients is {avg_pesudo_label_acc_pre}--------------------------------")
                mask = total_data_distributions != 0
                expanded_mask = mask.view(total_local_prototypes.shape[0], 1).expand(total_local_prototypes.shape[0], total_local_prototypes.shape[1]).float()
                expanded_mask = expanded_mask.bool()

                global_prototype = total_local_prototypes / torch.where(expanded_mask, total_data_distributions.view(total_local_prototypes.shape[0], 1), torch.ones_like(total_data_distributions).view(total_local_prototypes.shape[0], 1))
                zero_rows = torch.nonzero((global_prototype == 0).all(dim=1))
                zero_text_features = local_trainer.get_zero_shot_text_features()
                global_prototype[zero_rows] = zero_text_features[zero_rows]

                global_prototype = global_prototype / global_prototype.norm(dim=-1, keepdim=True)

                local_selector = []
                pesudo_label_acc = []
                uploaded_weights = []
                tot_samples=0
                pesudo_label_acc_avg=0
                similarity_between_true_and_pesudo_distribution = []
                similarity_between_true_and_pesudo_distribution_avg = 0

                total_sample_per_class = torch.ceil(torch.sum(total_data_distributions) / len(total_data_distributions))

                log.info(f"*************************The total pesudolabel of each class is : {total_sample_per_class}******************************")
                for client in range(obj_conf.client_num):
                  
                    distribution = (torch.ceil(torch.where(mask, data_distributions[client] / total_data_distributions * total_sample_per_class, len(train_dataset) / len(classes) / 4 * (int(round / obj_conf.num_repesudo_round)+1) / obj_conf.client_num))).int()
                    local_sample_per_class.append(distribution)

                for client in range(obj_conf.client_num):
                    log.info(f"------------------------generate pesudolabels for client {client}--------------------------------")
                    obj_conf.N_PSEUDOSHOTS = copy.deepcopy(local_sample_per_class[client])
                    train_dataset_fed_use[client] = copy.deepcopy(train_dataset_fed[client])
                    local_selector.append(InstanceSelector(label_to_idx=label_to_idx, cfg=obj_conf, device=device, N_PSEUDOSHOTS=True))

                    monitor_and_accelerate(UL_fed_train_lbs_true[client], train_dataset_fed[client], local_trainer, local_selector[client])                    

                    train_dataset_fed_use[client], partialY, acc, true_label_distribution, pesudo_label_distribution, selected_data_distribution, true_label_of_selected_data_distribution = local_trainer.create_training_dataset_prototype(        
                    train_dataset_fed_use[client], copy.deepcopy(train_dataset_fed_use[client]), 
                    iter_num= int(round/obj_conf.num_repesudo_round), Selector=local_selector[client], global_prototype=None, sample_per_class=local_sample_per_class[client], filepaths=local_filepaths[client], probs=local_probs[client], output_logits=local_output_logits[client], image_features=local_initial_features[client])     

                    true_label_distributions.append(true_label_distribution)
                    pesudo_label_distributions.append(pesudo_label_distribution)
                    selected_data_distributions.append(selected_data_distribution)
                    true_label_of_selected_data_distributions.append(true_label_of_selected_data_distribution)

                    pesudo_label_acc.append(acc)
                    similarity_between_true_and_pesudo_distribution.append(F.cosine_similarity(true_label_distribution.unsqueeze(0), selected_data_distribution.unsqueeze(0)))
                    log.info(f"\t similarity between true and pesudo distribution of client {client}: {F.cosine_similarity(true_label_distribution.unsqueeze(0), selected_data_distribution.unsqueeze(0))}")
                    uploaded_weights.append(len(partialY))
                    tot_samples+=len(partialY)

            
                for i, w in enumerate(uploaded_weights):
                    uploaded_weights[i] = w / tot_samples
                    pesudo_label_acc_avg+=pesudo_label_acc[i]
                    similarity_between_true_and_pesudo_distribution_avg+=similarity_between_true_and_pesudo_distribution[i]
                pesudo_label_acc_avg = pesudo_label_acc_avg / float(len(pesudo_label_acc))
                log.info(f"------------------------pesudolabels acc of all clients is {pesudo_label_acc_avg}--------------------------------")
                obj_conf.LR = copy.deepcopy(initial_learning_rate)

            results = []

            for client in range(obj_conf.client_num):
                log.info(f"--------------------------Trian for client {client}----------------------------")
                if round == 0:
                    local_trainer.model.load_state_dict(initial_global_weights,strict=False)
                else:
                    local_trainer.model.load_state_dict(global_weights,strict=False) 
                    local_trainer.model.load_state_dict(local_prompt_weights_per[client],strict=False) 

                # Initialize the loss function
                log.info(f"Building loss function..")
                local_trainer.loss_func = local_trainer.build_loss()

                local_trainer.train(
                    train_data=train_dataset_fed_use[client], 
                    val_data=None,  #all the train data
                    only_seen=False,
                    round=round,
                    selector=local_selector[client],
                )
                local_weight = local_trainer.model.state_dict()
                uploaded_models_0[client] = copy.deepcopy(local_weight['image_encoder.transformer.ctx_learner.ctx']) 
                local_prompt_weights_per[client]['prompt_learner.ctx_learner.ctx'] = copy.deepcopy(local_weight['prompt_learner.ctx_learner.ctx']) 

            global_weights['image_encoder.transformer.ctx_learner.ctx'] = average_weights(uploaded_models_0, obj_conf.client_num, uploaded_weights, islist=True) 
            for client in range(obj_conf.client_num): 
                local_trainer.model.load_state_dict(global_weights,strict=False) 
                local_trainer.model.load_state_dict(local_prompt_weights_per[client],strict=False) 
                acc = local_trainer.test_predictions(test_dataset_fed[client])
                log.info(f"The local test acc of client {client} in round {round} is : {acc}")
                results.append(acc)
            std_predictions = sum(results) / len(results)
            log.info(f"The avg local test acc in round {round} is : {std_predictions}") 


#============= Set logger and config =============
def log_args_and_env(cfg):
    log.info('************')
    log.info('** Config **')
    log.info('************')
    log.info(cfg)
    log.info('Collecting env info ...')
 
def setup_log_path(dir=None):
    if dir is None:
        return

    if dir.endswith(".txt") or dir.endswith(".log"):
        fpath = dir
    else:
        fpath = osp.join(dir, "log.txt")

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")
    return fpath

def set_logger(obj_conf):
    if obj_conf.OUTPUT_DIR != "":
        obj_conf.OUTPUT_DIR = obj_conf.OUTPUT_DIR
    else:
        obj_conf.OUTPUT_DIR = f"logs/for_DEBUG1/{obj_conf.LEARNING_PARADIGM}/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/', '-')}_seed{obj_conf.OPTIM_SEED}"

    if not os.path.exists(obj_conf.OUTPUT_DIR):
        os.makedirs(obj_conf.OUTPUT_DIR)
    fpath = setup_log_path(dir=obj_conf.OUTPUT_DIR)
    file_handler = logging.FileHandler(fpath)

    file_handler.setFormatter(formatter)
    # Add the FileHandler to the logger
    logger_.addHandler(file_handler)




#============= Main function =============
def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument(
        "--model_config",
        type=str,
        default="global_text_config_PLL.yml",
        help="Name of model config file",
    )

    args = parser.parse_args()
    try:
        with open(f"methods_config/{args.model_config}", "r") as file:
            config = yaml.safe_load(file)
    except:
        with open(f"enhanceCLIPwithCLIP/methods_config/{args.model_config}", "r") as file:
            config = yaml.safe_load(file)


    # ===========Cast configs to object===========
    obj_conf, dataset_dir = set_obj_conf(args, config)
    
    # Set the file path for the log file
    set_logger(obj_conf)

    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Dataset dir: {dataset_dir}")

    log_args_and_env(obj_conf)

    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception("`dataset_dir` does not exist..")

    # Set random seeds:
    become_deterministic(obj_conf.OPTIM_SEED)
    log.info('Setting fixed seed: {}'.format(obj_conf.OPTIM_SEED))

    accelerator.wait_for_everyone()

    # ===========run workflow===========
    workflow_new(dataset_dir, obj_conf)


if __name__ == "__main__":
    main()

