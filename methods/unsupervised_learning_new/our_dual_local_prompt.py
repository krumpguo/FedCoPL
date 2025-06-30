import logging
import torch
from accelerate import Accelerator
from torch import nn
import torch.nn.functional as F
import clip
import copy
from torchvision import transforms
import re
from utils.loss import PLL_loss

accelerator = Accelerator()
from methods.unsupervised_learning_new.training_strategies import (
    AverageMeter
)
from utils import make_scheduler, seed_worker
from collections import OrderedDict
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from utils import (
    make_scheduler, 
    seed_worker, 
    gererate_partialY,
    compute_unlabled_logits
)
from collections import Counter


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(x, list):
                x = [x]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def update_lr(self, names=None, epoch=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step(epoch=epoch)

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            log.info(f"Loss is infinite or NaN!")
            raise FloatingPointError("Loss is infinite or NaN!")

class Our_Dual_Local_Prompt(TrainerBase):
    def __init__(
        self,
        config,
        label_to_idx,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        """This class define Coop baseline.

        :param config: dictionaries of prameters in models_config/coop_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """
        super().__init__()
        self.config = config
        self.device = device
        self.label_to_idx = label_to_idx
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        seen_to_idx = {c: idx for idx, c in enumerate(self.seen_classes)}
        self.idx_to_real = {
            seen_to_idx[c]: self.label_to_idx[c] for c in self.seen_classes #将seen_to_idx中的索引映射到实际的标签索引（self.label_to_idx)
        }
        self.real_to_idx = {
            self.label_to_idx[c]: seen_to_idx[c] for c in self.seen_classes #反向映射
        }      
        self.n_ctx = self.config.PREFIX_SIZE
        self.clip_model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )
        self.clip_model = self.clip_model.float()
        self.clip_model.encoder_name = self.config.VIS_ENCODER
        self.transform_train = self.modify_transform(self.transform)
        self.template = self.config.PROMPT_TEMPLATE

    def modify_transform(self, transform):
        """
        Modify an existing transform.
        
        Parameters:
        transform (torchvision.transforms.Compose): The existing transform
    
        Returns:
        torchvision.transforms.Compose: The modified transform
        """
        # Get the normalization transform from the existing transform
        normalize = [t for t in transform.transforms if isinstance(t, transforms.Normalize)][0]
        # Get the Resize transform from the existing transform
        resize_transform = [t for t in transform.transforms if isinstance(t, transforms.CenterCrop)][0]
        # Parse the size from the Resize transform's print information
        size_info = re.search(r'size=\((\d+), (\d+)\)', str(resize_transform))
        H, W = map(int, size_info.groups())

        # Build the new transform
        transform_new = transforms.Compose([
            transforms.RandomResizedCrop(size=(H, W), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize  # Use the same normalization as the existing transform
        ])
        
        return transform_new

    def define_model(self, cfg, clip_model, classnames=None):
        """ This function initialized the model
        depending on the prompt modality.

        :param modality: either text or image
        :param classes: the list of classes for textual model
        """

        self.model = CustomCLIP_Selected_CoVPTDeep(cfg, classnames, clip_model, self.device)

        log.info(f"Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "image_encoder.transformer.ctx_learner"  not in name and "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        self.model.to(self.device)

        self.optim = torch.optim.SGD(
            self.model.image_encoder.transformer.ctx_learner.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.DECAY,
            momentum=0.9,
        )
        self.sched = make_scheduler(self.optim, cfg)

        self.optim2 = torch.optim.SGD(
            self.model.prompt_learner.ctx_learner.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.DECAY,
            momentum=0.9,
        )
        self.sched2 = make_scheduler(self.optim2, cfg)    

        self.register_model("image_encoder.transformer.ctx_learner", self.model.image_encoder.transformer.ctx_learner, self.optim, self.sched)
        self.register_model("prompt_learner.ctx_learner", self.model.prompt_learner, self.optim2, self.sched2)

    def build_loss(self):

        criterion = torch.nn.CrossEntropyLoss()     

        self.loss_func = criterion
        return self.loss_func

    def _before_train(self, train_data, val_data=None, train_transform=None, val_transform=None):
        # Declare the data pre processing for train and validation data
        train_data.transform = train_transform
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=8,
            drop_last=False,
            pin_memory=(torch.cuda.is_available()),
        )
        if val_data is not None:
            val_data.transform = val_transform
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                drop_last=False,
            )
        else:
            val_loader = None
        
        accelerator.wait_for_everyone()
        
        if val_loader is not None:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")
        
        return train_loader, val_loader

    def train(
        self,
        train_data,
        val_data=None,
        only_unlabelled=False,
        only_seen=False,
        round=None,
        selector=None,
    ):
        """This function defines the current training iteration of self.model.

        Args:
            train_data (CustomDataset): The labeled training dataset.
            unlabeled_data (CustomDataset): The unlabeled dataset.
            val_data (CustomDataset, optional): The validation dataset. Default is None.
            only_unlabelled (bool, optional): If True, train only with unlabeled data. Default is False.
            only_seen (bool, optional): If True, train only with seen classes. Default is False.
            iter_num (int, optional): The current iteration number. Default is None.
        """
        #2. prepare train loader
        train_loader, val_loader = self._before_train(
            train_data, val_data, 
            train_transform=self.transform, 
            val_transform=self.transform
        )
        # best_val_accuracy = 0
        loss = None

        # 3. start training:
        for epoch in range((round%self.config.num_repesudo_round)*10, (round%self.config.num_repesudo_round)*10 + self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER

            loss, total_losS = self._train_epoch(
                loss,
                total_loss,
                train_loader,
                epoch,
                selector=selector,
            )
            accelerator.wait_for_everyone()
            self._after_epoch(                
                train_data,
                epoch,
                selector)

            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")

    @torch.no_grad()
    def _after_epoch(self, train_data, epoch, selector):
        if not hasattr(self.loss_func, 'losstype') or '_' not in self.loss_func.losstype:
            """the loss_func do not need post-epoch processing (update conf)"""
            return

        elif epoch >= 0:
            train_loader, val_loader = self._before_train(train_data, val_data=None, 
                                                          train_transform=self.transform)

            acc_cum = AverageMeter()
            forward_method = self.get_clip_forward(target_class=self.classes)
            for i, (img, aug_1, idxs, label, img_path) in enumerate(train_loader):
                gt_label = self._get_gt_label(img_path, dtype=label.dtype, selector=selector)

                logits = forward_method(img)
                self.loss_func.check_conf_update(img, label, idxs, output=logits)   

                acc_cum.update(compute_accuracy(logits, gt_label)[0].item())
                if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                    log.info(
                        f"EVAL on epoch [{epoch}/{self.config.EPOCHS}] [{(i + 1)}/{len(train_loader)}]\t" 
                        f"acc {acc_cum.val:.3f} ({acc_cum.avg:.3f})\t"
                    )

            self.loss_func.clean_conf()

    def create_training_dataset(self, train_data, unlabeled_data, iter_num, Selector=None):
        """
        Create the dataset for training including pseudolabels for unseen classes.

        Args:
            train_data (Dataset): The dataset of the training seen classes.
            unlabeled_data (Dataset): The dataset of unlabeled data for unseen classes.
            iter_num (int): The iteration number.

        Raises:
            NotImplementedError: If the learning paradigm is not 'ul'.

        Returns:
            Dataset, Tensor: The updated training dataset and the selected pseudolabels.
        """
        if self.config.LEARNING_PARADIGM != "ul":
            raise NotImplementedError

        forward_method = self.get_clip_forward(target_class=self.classes, iter_num=iter_num)
        filepaths, probs, output_logits = compute_unlabled_logits(
            dataset=copy.deepcopy(unlabeled_data),
            transform=self.transform,
            clip_model=self.clip_model,
            forward_method=forward_method,
        )

        train_data_, PL_labels_selected, info = self._create_training_dataset_single_hard(
        train_data, iter_num,
        filepaths, probs, output_logits, Selector
        )
        return train_data_, PL_labels_selected, info


    def create_training_dataset_prototype(self, train_data, unlabeled_data, iter_num, Selector=None, global_prototype=None, local_prototype=None, sample_per_class=None, filepaths=None, probs=None, output_logits=None, image_features=None):
        """
        Create the dataset for training including pseudolabels for unseen classes.

        Args:
            train_data (Dataset): The dataset of the training seen classes.
            unlabeled_data (Dataset): The dataset of unlabeled data for unseen classes.
            iter_num (int): The iteration number.

        Raises:
            NotImplementedError: If the learning paradigm is not 'ul'.

        Returns:
            Dataset, Tensor: The updated training dataset and the selected pseudolabels.
        """
        mask_idxs = False
        
        
        train_data_, PL_labels_selected, info, true_label_distribution, pesudo_label_distribution, selected_data_distribution, true_label_of_selected_data_distribution = self._create_training_dataset_single_hard(
                        train_data, iter_num,
                        filepaths, probs, output_logits, Selector, mask_idxs
                    )
            
        return train_data_, PL_labels_selected, info, true_label_distribution, pesudo_label_distribution, selected_data_distribution, true_label_of_selected_data_distribution

    def _create_training_dataset_pre(self, train_data, unlabeled_data, iter_num, Selector=None, multi_prompt=False, train_probs_pre=[]):
        """
        Create the dataset for training by merging pseudo-labels and labeled data.

        Args:
            train_data (Dataset): The dataset of the training seen classes.
            iter_num (int): The iteration number.
            filepaths (list): List of file paths for the data.
            probs (Tensor): Probabilities from the model.
            output_logits (Tensor): Logits from the model.

        Returns:
            Dataset, Tensor, info dict: The updated training dataset, the selected pseudolabels
        """

        forward_method = self.get_clip_forward(target_class=self.classes, iter_num=iter_num)

        filepaths, probs, output_logits, features = compute_unlabled_logits(
            dataset=copy.deepcopy(unlabeled_data),
            transform=self.transform,
            clip_model=self.clip_model,
            forward_method=forward_method,
            pre=True,
        )

        target_quantile_confidence = self.config.CONF_QUANTILE_CONFIDENCE
        target_quantile_entropy = self.config.CONF_QUANTILE_ENTROPY

        max_values, max_indices = torch.max(probs, dim=1)
        entropys =  -(output_logits.softmax(1) * output_logits.log_softmax(1)).sum(1)

        conf_thr = torch.quantile(probs.max(dim=1).values, target_quantile_confidence/100).cpu().item()

        conf_thr_1 = torch.quantile(entropys, (1 - target_quantile_entropy/100)).cpu().item()

        indices = torch.nonzero((max_values > conf_thr) & (entropys < conf_thr_1)).squeeze()

        selected_labels = max_indices[indices]
        selected_features = features[indices]
        data_distribution = torch.zeros(len(self.classes)).to(self.device)
        local_feature = torch.zeros(len(self.classes), selected_features.shape[1]).to(self.device)

        for i in range(len(selected_labels)):
            label = selected_labels[i]
            local_feature[label] += selected_features[i]
            data_distribution[label] += 1

        correct=0.0
        filepaths_new = [filepaths[i] for i in indices.tolist()]
        gt_labels = self._get_gt_label(impath=filepaths_new, dtype=torch.long, selector=Selector)
        for i in range(len(selected_labels)):
            if selected_labels[i] == gt_labels[i]:
                correct += 1        
        acc = correct / len(selected_labels)   

        return local_feature, data_distribution, acc, filepaths, probs, output_logits, features

    def _create_training_dataset_single_hard(self, train_data, iter_num,
                                       filepaths, probs, output_logits, Selector, mask_idxs=False):
        """
        Create the dataset for training by merging pseudo-labels and labeled data.

        Args:
            train_data (Dataset): The dataset of the training seen classes.
            iter_num (int): The iteration number.
            filepaths (list): List of file paths for the data.
            probs (Tensor): Probabilities from the model.
            output_logits (Tensor): Logits from the model.

        Returns:
            Dataset, Tensor, info dict: The updated training dataset, the selected pseudolabels
        """

        true_label_distribution = torch.zeros(len(self.classes))
        pesudo_label_distribution = torch.zeros(len(self.classes))
        true_label_of_selected_data_distribution = torch.zeros(len(self.classes))
        selected_data_distribution = torch.zeros(len(self.classes))
        gt_labels = self._get_gt_label(impath=filepaths, dtype=torch.long, selector=Selector)
        for idx in range(len(gt_labels)):
            true_label_distribution[gt_labels[idx]] += 1

        # class-level select
        labels = torch.zeros(probs.shape[0], len(self.classes))
        max_values, max_indices = torch.max(probs, dim=1)
        max_indices = max_indices.to('cpu')
        labels = torch.eye(len(self.classes))[max_indices].to(self.device)
        for idx in range(len(max_indices)):
            pesudo_label_distribution[max_indices[idx]] += 1      
        if torch.is_tensor(mask_idxs) == False:
            selected_idxs, info_2 = Selector.select_topk_for_eachcls(
                PL_labels=(labels > 1e-7).float(),
                output_all=output_logits,
                indexs_all=torch.arange(len(filepaths)),
                K_max=self.config.N_PSEUDOSHOTS,
                N_iter=iter_num,
                multi_k=True,
            )
        else:
            selected_idxs, info_2 = Selector.select_topk_for_eachcls(
                PL_labels=(labels > 1e-7).float()[mask_idxs],
                output_all=output_logits[mask_idxs],
                indexs_all=torch.arange(len(filepaths))[mask_idxs],
                K_max=self.config.N_PSEUDOSHOTS,
                N_iter=iter_num,
                multi_k=True,
            )

        # Update the training dataset
        selected_labels = labels[selected_idxs, :] 

        filepaths_new = [filepaths[i] for i in selected_idxs.tolist()]
        train_data.update_xy(labels=selected_labels.cpu(), filepaths=filepaths_new)

        for idx in range(len(selected_labels)):
            max_values, max_indices = torch.max(selected_labels[idx],dim=0)
            selected_data_distribution[max_indices] += 1

        correct=0.0
        gt_labels = self._get_gt_label(impath=filepaths_new, dtype=torch.long, selector=Selector)
        for i in range(len(selected_labels)):
            max_values, max_indices = torch.max(selected_labels[i], dim=0)
            if max_indices == gt_labels[i]:
                correct += 1        
    
        acc = correct / len(selected_labels)
        log.info(f"\t label_estimate_acc: {acc}")

        for idx in range(len(gt_labels)):
            true_label_of_selected_data_distribution[gt_labels[idx]] += 1
        log.info(f"\n true_label_distribution: {true_label_distribution}")
        log.info(f"\n pesudo_label_distribution: {pesudo_label_distribution}")    
        log.info(f"\n selected_data_distribution: {selected_data_distribution}")    
        log.info(f"\n true_label_of_selected_data_distribution: {true_label_of_selected_data_distribution}")

        return train_data, selected_labels, acc, true_label_distribution, pesudo_label_distribution, selected_data_distribution, true_label_of_selected_data_distribution


    def check_partialY_acc(self, PL_labels, filepaths, selector):
        # check the accuracy of pseudolabels
        gt_labels = self._get_gt_label(impath=filepaths, dtype=torch.long, selector=selector)

        # initialize a list to store the results
        results = []
        distribution = []
        # iterate over each row of PL_labels and the corresponding gt_labels
        for i in range(PL_labels.shape[0]):
            # get the indices where the values are 1.0 in the current row
            indices = torch.nonzero(PL_labels[i], as_tuple=True)

            # test if the corresponding gt_label is in these indices
            is_in = gt_labels[i] in indices[0]
            distribution.extend(indices[0].tolist())

            # append the result to the list
            results.append(is_in)
        
        results = torch.tensor(results)
        coverage_acc = results.sum() / results.shape[0]
        ct = Counter(distribution)
        ct = sorted(ct.items(), key=lambda x: x[0])
        partial_avgnum = (PL_labels > 1e-7).sum(dim=1).float()

        log.info(f"\t label_estimate_acc: {coverage_acc}")
        # log.info(f"coverage distribution: {ct}")
        partialR = partial_avgnum.mean().item()/PL_labels.shape[1]

        return {"label_estimate_acc": coverage_acc.item(), 
                "partial_ratio": partialR, 
                }


    def define_loss_function(self, logits, label, idxs):
        """Return the loss value for the given batch."""
        loss= self.loss_func(logits, label)
        return loss

    def _get_gt_label(self, impath, dtype, selector):
        """
        Retrieves the ground truth labels for a given list of image paths.

        :param impath: A list of image paths for which the ground truth labels are to be retrieved.
        :param dtype: The data type to be used for the returned tensor of labels.
        :return: A tensor containing the ground truth labels for the provided image paths, 
                converted to the specified data type and moved to the model's device.
        """
        gt_label_list = []
        for ip in impath:
            gt_label = selector.all_gt_label_dict[ip]
            gt_label_list.append(gt_label)
        gt_label = torch.tensor(gt_label_list, dtype=dtype).to(self.device)
        return gt_label

    def _train_epoch(
        self, 
        loss, 
        total_loss, 
        train_loader, 
        epoch, 
        selector=None, 
    ):
        """This function defines the training epoch of self.model.

        :param loss: float loss (average across batches)
        :param total_loss: float total loss
        :param train_loader: Dataloader object - training data defined in self.train
        :param accum_iter: number of accumulation steps minimum 1
        :param epoch: current epoch
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        :param only_seen: boolean.  It is True if the training only involves seen data
        """
        acc_cum = AverageMeter()
        loss_cum = AverageMeter()
        forward_method = self.get_clip_forward(target_class=self.classes)
        self.update_lr(epoch=epoch)
        for i, (img, aug_1, idxs, label, img_path) in enumerate(train_loader):
            gt_label = self._get_gt_label(img_path, dtype=label.dtype, selector=selector)

            img, label = img.to(self.device), label.to(self.device)

            logits = forward_method(img)
            loss = self.define_loss_function(logits, label, idxs)

            total_loss += loss.item()
            accelerator.wait_for_everyone()
            
            self.model_backward_and_update(loss)

            # compute accuracy:
            acc_cum.update(compute_accuracy(logits[:len(img_path)], gt_label)[0].item())
            loss_cum.update(loss.item())
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                log.info(
                    f"epoch [{epoch}/{(self.config.EPOCHS * self.config.num_repesudo_round)}][{(i + 1)}/{len(train_loader)}]  \t" 
                    f"loss {loss_cum.val:.3f} ({loss_cum.avg:.3f})\t"
                    f"acc {acc_cum.val:.3f} ({acc_cum.avg:.3f})\t"
                )

        accelerator.wait_for_everyone() 

        return loss, total_loss
    
    def get_clip_forward(self, target_class, iter_num=2, dtype=torch.float32):
        """
        This function returns the forward method for CLIP under the correct settings.
        """
        
        def clip_forward_text(img, pre=False):
            if pre:
                logits, image_features = self.model(img, pre)
                return logits, image_features
            else:
                logits = self.model(img)
                return logits
           
        def clip_zsl_forward(img, pre=False):
            prompts = [self.config.PROMPT_TEMPLATE.format(c.replace("_", " ")) for c in target_class]
            # log.info(f"clip_zsl Prompts: {prompts[0:10]}")
            text = clip.tokenize(prompts).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.encode_text(text).type(dtype)
                text_features = (text_features / text_features.norm(dim=-1, keepdim=True))

                if img.dim() == 4:
                    image_features = self.clip_model.encode_image(img.to(self.device))
                    image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True).type(dtype)
                elif img.dim() == 2:
                    image_features = img.to(self.device).type(dtype)
                else:
                    raise ValueError(f"Image dimension {img.dim()} not supported.")

            # cosine similarity as logits:
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            if pre:
                return logits, image_features
            else:
                return logits
            
        # 2. return the correct forward method:
        if iter_num == 0:
            forward_method = clip_zsl_forward
            log.info(f"Use zero-shot prompt template: {self.config.PROMPT_TEMPLATE}")
        else:
            forward_method = clip_forward_text

        return forward_method


    def get_zero_shot_text_features(self, dtype=torch.float32):
        prompts = [self.config.PROMPT_TEMPLATE.format(c.replace("_", " ")) for c in self.classes]
        text = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text).type(dtype)
            text_features = (text_features / text_features.norm(dim=-1, keepdim=True))
        
        return text_features

    @torch.no_grad()
    def test_predictions(self, data, standard_zsl=False, zero_test=False):
        """
        Computes predictions on the test dataset and evaluates the model's performance.

        Args:
            data: A dataset object representing the test dataset.
            standard_zsl (bool): temp var to be removed

        Returns:
            The harmonic mean of seen and unseen classes' accuracies in TRZSL setting, 
            or overall accuracy in other settings.
        """

        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE,
            num_workers=8,
            drop_last=False,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        log.info(f"Start inference for test data")

        predictions, labels_true, logits_all = [], [], []

        forward_method = self.get_clip_forward(target_class=self.classes)
        for img, aug_1, idxs, label, img_path in test_loader:
            label = label.to(self.device)
            img = img.to(self.device)
            logits = forward_method(img)
            pred = torch.argmax(logits, dim=1)

            predictions.append(pred)
            labels_true.append(label)
            logits_all.append(logits)

        accelerator.wait_for_everyone()

        predictions = torch.cat(predictions, dim=0)
        labels_true = torch.cat(labels_true, dim=0)
        logits_all = torch.cat(logits_all, dim=0)   

        overall_acc = (predictions == labels_true).sum() / predictions.shape[0]
        return overall_acc.item()
    
    

def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res

class AverageMeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count

class CustomCLIP_Selected_CoVPTDeep(nn.Module):
    def __init__(self, cfg, classnames, clip_model, devices):
        super().__init__()
        self.class_prompt_num = cfg.PREFIX_SIZE
        device = devices
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # visual
        self.image_encoder = ImageEncoder_VPTD(cfg, classnames, clip_model)
        # visual end
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, pre=False):
        image = image.to(next(self.image_encoder.parameters()).device)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features_norm.t()
        if pre:
            return logits, image_features
        else:
            return logits


class ImageEncoder_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = Transformer_VPTD(cfg, classnames, clip_model)
        self.ln_post = clip_model.visual.ln_post
        self.proj = ProjLearner(clip_model)
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class_embedding is class token.
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # only take class token which is awsome.

        x = self.proj(x)

        return x

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class Transformer_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = 5
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.layers = clip_model.visual.transformer.layers

        # model
        transformer = clip_model.visual.transformer
        self.resblocks: nn.Sequential = transformer.resblocks
        self.layers = transformer.layers
        
        self.ctx_learner = VPTDeepPromptLearner(cfg, classnames, clip_model)
        self.class_prompt_num = 5
        self.n_ctx = self.n_ctx
        self.bottom_limit = 11
        
    def forward(self, x):
        ctx = self.ctx_learner()
        ctx = ctx.unsqueeze(0).expand(x.shape[1], -1, -1, -1) # batch layers n_ctx feature 
        ctx = ctx.permute(1, 2, 0, 3)
        n_ctx = self.n_ctx

        for i in range(self.bottom_limit):
            # print(ctx[i].shape, x.shape)
            x = torch.cat([x, ctx[i]], dim=0)
            x = self.resblocks[i](x)
            x = x[:-n_ctx, :, :]
            # print("bottom", x.shape)
        
        n_ctx = self.class_prompt_num
        
        for i in range(self.layers-self.bottom_limit):
            x = self.resblocks[i+self.bottom_limit](x)
            if n_ctx != 0:
                x = x[:-n_ctx, :, :]
        
        return x

class VPTDeepPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = 5
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 76
        self.bottom_limit = 11
        
        ctx_vectors = torch.empty(self.bottom_limit, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self):
        ctx = self.ctx
        return ctx

class ProjLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.proj = clip_model.visual.proj
        
    def forward(self,x):
        if self.proj is not None:
            x = x @ self.proj
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.PREFIX_SIZE
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.device = device

        prompt_prefix = " ".join(["X"] * n_ctx)    
        self.ctx_learner = TextPromptLearner(cfg, classnames, clip_model)

        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])   
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx_learner()

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts
    
class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, ctx_vectors=None):
        super().__init__()
        # hyper param
        self.n_ctx = 16
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self):
        ctx = self.ctx
        return ctx


    

