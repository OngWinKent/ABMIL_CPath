from torchmetrics.classification import Accuracy,Precision,Recall,CohenKappa
from torchmetrics import AUROC,Metric, MetricCollection
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.wrappers.bootstrapping import BootStrapper
from typing import Dict,Any
import torch
from torch import Tensor
from sksurv.metrics import concordance_index_censored
from lightning_utilities import apply_to_collection
from collections import OrderedDict

def _bootstrap_sampler(
    size: int,
    generator: Any,
    sampling_strategy: str = "poisson",
) -> Tensor:
    """Resample a tensor along its first dimension with replacement.

    Args:
        size: number of samples
        sampling_strategy: the strategy to use for sampling, either ``'poisson'`` or ``'multinomial'``

    Returns:
        resampled tensor

    """
    if sampling_strategy == "poisson":
        p = torch.distributions.Poisson(1)
        n = p.sample((size,))
        return torch.arange(size).repeat_interleave(n.long(), dim=0)
    if sampling_strategy == "multinomial":
        return torch.multinomial(torch.ones(size), num_samples=size, replacement=True,generator=generator)
    raise ValueError("Unknown sampling strategy")

class DeterministicBootStrapper(BootStrapper):
    def __init__(self, seed,**kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of the base metric.

        Any tensor passed in will be bootstrapped along dimension 0.

        """
        args_sizes = apply_to_collection(args, Tensor, len)
        kwargs_sizes = apply_to_collection(kwargs, Tensor, len)
        if len(args_sizes) > 0:
            size = args_sizes[0]
        elif len(kwargs_sizes) > 0:
            size = next(iter(kwargs_sizes.values()))
        else:
            raise ValueError("None of the input contained tensors, so could not determine the sampling size")
        
        generator = torch.Generator()
        generator.manual_seed(self.seed) 

        for idx in range(self.num_bootstraps):
            sample_idx = _bootstrap_sampler(size, generator=generator,sampling_strategy=self.sampling_strategy).to(self.device)
            if sample_idx.numel() == 0:
                continue
            new_args = apply_to_collection(args, Tensor, torch.index_select, dim=0, index=sample_idx)
            new_kwargs = apply_to_collection(kwargs, Tensor, torch.index_select, dim=0, index=sample_idx)
            self.metrics[idx].update(*new_args, **new_kwargs)

class ConcordanceIndexCensored(Metric):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.add_state("censorships", default=[], dist_reduce_fx=None)
        self.add_state("event_times", default=[], dist_reduce_fx=None)
        self.add_state("risk_scores", default=[], dist_reduce_fx=None)

    def update(self, censorships, event_times, risk_scores):
        self.censorships = censorships
        self.event_times = event_times
        self.risk_scores = risk_scores
    
    def compute(self):
        c_index = concordance_index_censored(self.censorships.numpy(), self.event_times.numpy(), self.risk_scores.numpy(), tied_tol=1e-08)[0]
        c_index = torch.tensor(c_index, dtype=torch.float32)
        self.reset()
        return c_index
    
    def reset(self):
        # Reset the state by clearing the lists
        self.censorships = []
        self.event_times = []
        self.risk_scores = []

def get_surv_metrics(args,censorships, event_times, risk_scores,bootstrap=False,raw=False):
    _metrics = metrics_base(args,bootstrap,raw=raw)
    _results = _metrics(censorships, event_times, risk_scores)

    for key in _results:
        if hasattr(_results[key], 'cpu'):  
            _results[key] = _results[key].cpu()

    if bootstrap:
        if raw:
            pass
        else:
            return [_results['mean'],_results['std']]
    else:
        return _results['C-index']

def get_cls_metrics(logits,labels,bootstrap,device,n_classes,seed,num_bootstrap):
    _metrics = metrics_base(n_classes= n_classes,seed= seed, num_bootstrap= num_bootstrap, bootstrap=bootstrap,device=device)
    if n_classes == 2:
         logits = logits[:,1]

    _results = _metrics(logits,labels)

    for key in _results:
        if hasattr(_results[key], 'cpu'): 
            _results[key] = _results[key].cpu()

    if bootstrap:
        return [_results['Acc_mean'],_results['Acc_std']],[_results['AUC_mean'],_results['AUC_std']],[_results['Precision_mean'],_results['Precision_std']],[_results['Recall_mean'],_results['Recall_std']],[_results['F1_mean'],_results['F1_std']],[_results['CK_mean'],_results['CK_std']],[_results['Acc_micro_mean'],_results['Acc_micro_std']]
    else:
        return _results['Acc'], _results['AUC'], _results['Precision'], _results['Recall'], _results['F1'], _results['CK'], _results['Acc_micro']

def metrics_base(n_classes, seed, num_bootstrap, bootstrap=False,device=None):
    _surv = False
    if n_classes == 2:
        metrics: Dict[str, Metric] = {
                "Acc": Accuracy(top_k=1, num_classes=int(n_classes), task='binary').to(device),
                "F1": F1Score(num_classes=int(n_classes),task='binary').to(device),
                "AUC": AUROC(num_classes=int(n_classes),task='binary').to(device),
                "Precision": Precision(num_classes=int(n_classes),task='binary').to(device),
                "Recall": Recall(num_classes=int(n_classes),task='binary').to(device),
                "CK": CohenKappa(num_classes=int(n_classes),task='binary').to(device),
                "Acc_micro": Accuracy(top_k=1, num_classes=int(n_classes), task='binary').to(device),
        }
    else:
        metrics: Dict[str, Metric] = {
                "Acc": Accuracy(top_k=1, num_classes=int(n_classes), task='multiclass',average='macro').to(device),
                "F1": F1Score(num_classes=int(n_classes), average="macro", task="multiclass").to(device),
                "AUC": AUROC(num_classes=int(n_classes), average="macro", task="multiclass").to(device),
                "Precision": Precision(num_classes=int(n_classes), average="macro", task="multiclass").to(device),
                "Recall": Recall(num_classes=int(n_classes), average="macro", task="multiclass").to(device),
                "CK": CohenKappa(num_classes=int(n_classes),task="multiclass").to(device),
                "Acc_micro": Accuracy(top_k=1, num_classes=int(n_classes), task='multiclass').to(device),
        }

    # boot strap wrap
    if bootstrap:
        for k, m in metrics.items():
            metrics[k] = DeterministicBootStrapper(seed=seed,base_metric=m, num_bootstraps=num_bootstrap, sampling_strategy="multinomial",raw=True,sync_on_compute=False if _surv else True).to(device)
    metrics = MetricCollection(metrics)
    return metrics

def get_metric_val(args, bag_logit, bag_labels, model, status, early_stopping, epoch, loss_cls_meter, device, n_classes, seed, num_bootstrap, best_metric_index):
    if status == 'test':
        _bootstrap = True
    else:
        _bootstrap = False
    suffix = ''
    threshold_optimal = None

    # Get classification metrics
    accuracy, auc_value, precision, recall, fscore, ck, acc_micro = get_cls_metrics(
        bag_logit, bag_labels, _bootstrap, device=device, n_classes= n_classes, seed= seed, num_bootstrap= num_bootstrap)

    if status == 'val':
        # Early stop 
        if early_stopping is not None:
            if best_metric_index == 0:
                early_stopping(args, epoch, -auc_value, model)
            elif best_metric_index == 1:
                early_stopping(args, epoch, -accuracy, model)
            stop = early_stopping.early_stop
        else:
            stop = False

        rowd = OrderedDict([
            ("acc", accuracy),
            ("precision", precision),
            ("recall", recall),
            ("fscore", fscore),
            ("auc", auc_value),
            ("ck", ck),
            ("acc_micro", acc_micro),
            ("loss", loss_cls_meter.avg),
        ])
        rowd = OrderedDict([(_k + str(suffix), _v) for _k, _v in rowd.items()])
        return [auc_value, accuracy, precision, recall, fscore, ck, acc_micro], stop, loss_cls_meter.avg, threshold_optimal, rowd

    else:
        rowd = OrderedDict([
            ("acc", accuracy[0]),
            ("precision", precision[0]),
            ("recall", recall[0]),
            ("fscore", fscore[0]),
            ("auc", auc_value[0]),
            ("ck", ck[0]),
            ("acc_micro", acc_micro[0]),
            ("loss", loss_cls_meter.avg),
            ("acc_std", accuracy[1]),
            ("fscore_std", fscore[1]),
            ("auc_std", auc_value[1]),
            ("ck_std", ck[1]),
            ("acc_micro_std", acc_micro[1]),
        ])
        rowd = OrderedDict([(_k + str(suffix), _v) for _k, _v in rowd.items()])
        return [auc_value[0], accuracy[0], precision[0], recall[0], fscore[0], ck[0], acc_micro[0], auc_value[1], accuracy[1], fscore[1], ck[1], acc_micro[1]], loss_cls_meter.avg, rowd