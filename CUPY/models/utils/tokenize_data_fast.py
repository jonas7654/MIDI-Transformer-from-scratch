from miditok import REMI
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
import numpy as np
from tqdm import tqdm
from symusic import Score


SCORE_LOADING_EXCEPTION = (
    RuntimeError,
    ValueError,
    OSError,
    FileNotFoundError,
    IOError,
    EOFError,
)

def tokenize_dataset_to_bin(self, files_paths: str | Path | Sequence[str | Path],
                            validation_fn=None,
                            save_programs=None,
                            verbose=True):
    """
    @Jonas
    Custom method to tokenize files and return a NumPy array.
    This only works with a tokenizer object
    """
    self._verbose = verbose

    # Resolve file paths
    if not isinstance(files_paths, Sequence):
        if isinstance(files_paths, str):
            files_paths = Path(files_paths)
        files_paths = [
            path
            for path in files_paths.glob("**/*")
            if path.suffix in ['.mid', '.midi']
        ]

    if save_programs is None:
        save_programs = not self.config.use_programs

    all_ids = []  # To store token IDs
    max_length = 0  # Track the longest sequence for padding

    desc = "Tokenizing music files (in-memory)"
    for file_path in tqdm(files_paths, desc=desc):
        file_path = Path(file_path)
        try:
            score = Score(file_path)
        except FileNotFoundError:
            if self._verbose:
                warnings.warn(f"File not found: {file_path}", stacklevel=2)
            continue
        except SCORE_LOADING_EXCEPTION:
            continue

        # Validate the score
        if validation_fn is not None and not validation_fn(score):
            continue

        # Tokenize the Score
        tokens = self.encode(score)
        
        # Collect token IDs
        token_ids = tokens[0].ids
        all_ids.append(token_ids)
        max_length = max(max_length, len(token_ids))

    # Convert collected token IDs to a padded NumPy array
    token_array = np.array(
        [ids + [0] * (max_length - len(ids)) for ids in all_ids], dtype=np.int32
    )

    self._verbose = False
    return token_array


REMI.tokenize_dataset_to_bin = tokenize_dataset_to_bin