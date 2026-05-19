
import pickle
from typing import Any

def save_object(obj: Any, file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_object(file_path: str) -> Any:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
