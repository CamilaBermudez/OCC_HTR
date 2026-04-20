import os
import re
from pathlib import Path
from typing import Tuple, Union


def fixed_file_naming(base_name: str, padding: int = 2) -> str:
    padded_name = re.sub(r'^\d+', lambda m: m.group().zfill(padding), base_name)
    processed_name = padded_name.replace(" ", "").replace("-", "_").replace("f.", "f_")
    return processed_name

def format_filename(base_name: str, output_folder: Union[str, Path], padding: int = 2) -> Tuple[Path, str, str]:
    """
    Format a base filename for Kraken output
    Returns: (output_path, output_filename, processed_name)
    """
    output_folder = Path(output_folder)
    processed_name = fixed_file_naming(base_name, padding)
    output_filename = f"{processed_name}.json"
    output_path = os.path.join(output_folder, output_filename) 
    return output_path, output_filename, processed_name

def format_for_cli(*paths: Union[str, Path]) -> Tuple[str, ...]:
    """
    Convert paths for CLI tools.
    Always returns a tuple of formatted strings.
    """
    return tuple(Path(p).as_posix() for p in paths)