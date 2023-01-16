from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch

from patronus.evaluating.mask import IEvaluator
from patronus.evaluating.module import metrica
from patronus.logging.mask import ILogger


class IREvaluator(IEvaluator):
    def __init__(self):
        pass


class SEMIEvaluator(IREvaluator):
    """Semantic Information Retrieval Evaluator"""

    def __init__(
        self,
        model_name_or_path: Union[Path, str] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        logger: Optional[ILogger] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                f"Please install <a href=\"https://pypi.org/project/sentence-transformers/\">sentence-transformers</a> first."
            )
        super(SEMIEvaluator, self).__init__()
        self.model = SentenceTransformer(model_name_or_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.logger = logger if logger is not None else None
        self.model_name = self.model.__class__.__name__ if Path(model_name_or_path).is_dir() else model_name_or_path

    def evaluate(self, dataset: List[Dict]):
        pass


class STIREvaluator(IREvaluator):
    """STatic Information Retrieval Evaluator"""

    metrics = {}

    def __init__(self, metrics: List[Union[str, Callable]], **kwargs):
        super(STIREvaluator, self).__init__()
        for fn in metrics:
            if not isinstance(fn, str):
                assert isinstance(fn, Callable), "The metric you provide is neither Callable not a name"
                metrics[str(fn)] = fn
                continue
            assert fn in metrica.keys(), f"The metric name \"{str(fn)}\" is not implemented"
            metrics[fn] = metrica[str(fn)]

    def evaluate(self, dataset: List[Dict], mode: str = "hard", return_every_score=False):
        ans_step = []
        ans = {k: 0.0 for k in self.registered_metrics}
        for i, e in enumerate(dataset):
            preds, labels = e["preds"], e["labels"]

            cur_ans = {}

            for naming in self.registered_metrics:
                fun = self.metrics[naming]
                score = fun(preds, labels, mode=mode)
                cur_ans[naming] = score
                if self.logger:
                    self.logger.log_metrics({f"{naming}": score}, step=i)

            if return_every_score:
                ans_step.append(cur_ans)

            for k, v in cur_ans.items():
                ans[k] += v * 1.0 / len(dataset)

        if return_every_score:
            return ans, ans_step
        return ans


__all__ = ["SEMIEvaluator", "STIREvaluator"]
