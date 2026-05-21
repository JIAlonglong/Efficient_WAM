"""Probe dataset with disk caching for extracted Motus tokens.

MotusProbeDataset extracts tokens from a frozen Motus model once, caches
them to disk, and serves them as a standard PyTorch Dataset for fast
probe training.

Reference: semantic-wm's TrajectoryProbeDataset (probe_dataset.py)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .extract_motus_tokens import extract_motus_tokens

logger = logging.getLogger(__name__)


class MotusProbeDataset(Dataset):
    """Dataset that caches extracted Motus tokens to disk.

    On first instantiation, runs the frozen Motus model over the entire
    dataloader to extract tokens, then saves them to a cache file.
    Subsequent instantiations load from cache.

    Parameters
    ----------
    motus_model : nn.Module or None
        Frozen Motus model. Only needed for initial extraction.
    dataloader : DataLoader or None
        Source dataloader for extraction. Only needed for initial extraction.
    device : torch.device
        Device for extraction.
    cache_dir : str or Path
        Directory to store cached tokens.
    token_types : list of str, optional
        Which token types to extract. Default: all four.
    force_extract : bool
        If True, re-extract even if cache exists.
    """

    TOKEN_TYPES = ["video_before", "video_after", "und_before", "und_after"]

    def __init__(
        self,
        motus_model=None,
        dataloader=None,
        device: torch.device = torch.device("cpu"),
        cache_dir: str | Path = "probe_cache",
        token_types: Optional[List[str]] = None,
        force_extract: bool = False,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_types = token_types or self.TOKEN_TYPES
        self.device = device

        cache_path = self.cache_dir / "tokens.pt"

        if cache_path.exists() and not force_extract:
            logger.info("Loading cached tokens from %s", cache_path)
            self._load_cache(cache_path)
        else:
            if motus_model is None or dataloader is None:
                raise ValueError(
                    "motus_model and dataloader are required for initial "
                    "token extraction (no cache found at %s)" % cache_path
                )
            logger.info("Extracting tokens and caching to %s", cache_path)
            self._extract_and_cache(motus_model, dataloader, device, cache_path)

    def _load_cache(self, cache_path: Path) -> None:
        """Load cached tensors from disk."""
        data = torch.load(cache_path, map_location="cpu", weights_only=False)

        for token_type in self.TOKEN_TYPES:
            if token_type in data:
                setattr(self, token_type, data[token_type])
            else:
                setattr(self, token_type, None)

        self.actions = data["actions"]
        self._length = len(self.actions)
        logger.info(
            "Loaded %d cached samples. Token shapes: %s",
            self._length,
            {t: getattr(self, t).shape if getattr(self, t) is not None else None
             for t in self.TOKEN_TYPES},
        )

    def _extract_and_cache(
        self,
        model,
        loader,
        device: torch.device,
        cache_path: Path,
    ) -> None:
        """Run model forward pass to extract and cache all tokens."""
        collected: Dict[str, List[torch.Tensor]] = {t: [] for t in self.TOKEN_TYPES}
        collected["actions"] = []

        model.eval()
        for batch in tqdm(loader, desc="Extracting Motus tokens"):
            tokens = extract_motus_tokens(
                model, batch, device,
                extract_before=any(t.endswith("_before") for t in self.token_types),
                extract_after=any(t.endswith("_after") for t in self.token_types),
            )

            for token_type in self.token_types:
                if token_type in tokens:
                    collected[token_type].append(tokens[token_type].cpu())

            # Actions: expected shape [B, chunk_size, action_dim]
            actions = batch["actions"]
            if isinstance(actions, torch.Tensor):
                collected["actions"].append(actions.cpu())

        # Concatenate all batches
        for token_type in self.TOKEN_TYPES:
            if collected[token_type]:
                tensor = torch.cat(collected[token_type], dim=0)
                setattr(self, token_type, tensor)
            else:
                setattr(self, token_type, None)

        self.actions = torch.cat(collected["actions"], dim=0)
        self._length = len(self.actions)

        # Save to disk
        save_dict = {t: getattr(self, t) for t in self.TOKEN_TYPES if getattr(self, t) is not None}
        save_dict["actions"] = self.actions
        torch.save(save_dict, cache_path)
        logger.info("Saved %d samples to %s", self._length, cache_path)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tokens and action for a single sample.

        Returns
        -------
        dict with keys matching self.token_types + "action".
        """
        item = {}
        for token_type in self.TOKEN_TYPES:
            tensor = getattr(self, token_type)
            if tensor is not None:
                item[token_type] = tensor[idx]
        item["action"] = self.actions[idx]
        return item


def build_probe_datasets(
    motus_model=None,
    train_loader=None,
    test_loader=None,
    device: torch.device = torch.device("cpu"),
    cache_dir: str = "probe_cache",
    token_types: Optional[List[str]] = None,
    force_extract: bool = False,
) -> Tuple[MotusProbeDataset, MotusProbeDataset]:
    """Build train and test probe datasets with caching.

    Convenience function that creates both train and test datasets,
    extracting tokens if needed.

    Parameters
    ----------
    motus_model : nn.Module or None
        Frozen Motus model (needed only for extraction).
    train_loader, test_loader : DataLoader or None
        Source dataloaders (needed only for extraction).
    device : torch.device
        Device for extraction.
    cache_dir : str
        Base cache directory. Train/test subdirectories are created inside.
    token_types : list of str, optional
        Token types to extract.
    force_extract : bool
        Force re-extraction.

    Returns
    -------
    (train_dataset, test_dataset) : Tuple[MotusProbeDataset, MotusProbeDataset]
    """
    train_dataset = MotusProbeDataset(
        motus_model=motus_model,
        dataloader=train_loader,
        device=device,
        cache_dir=str(Path(cache_dir) / "train"),
        token_types=token_types,
        force_extract=force_extract,
    )
    test_dataset = MotusProbeDataset(
        motus_model=motus_model,
        dataloader=test_loader,
        device=device,
        cache_dir=str(Path(cache_dir) / "test"),
        token_types=token_types,
        force_extract=force_extract,
    )
    return train_dataset, test_dataset
