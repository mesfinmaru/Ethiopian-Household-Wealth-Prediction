"""
sav_reader.py
═══════════════════════════════════════════════════════════════════════════════
Pure-Python SPSS SAV decoder for Wave 2 (ETH_2013_ESS_v03_M_SPSS).
No external dependencies beyond numpy and pandas.

CRISP-DM Phase 1 (Data Understanding): This module is the entry point for
Wave 2 data. It decodes the binary SPSS format into a pandas DataFrame,
then applies W2_COL_RENAME from config so all downstream code sees
consistent column names across all 5 waves.

Supported format: bytecode compression (compress=1, bias=100) — the only
compression type used by the World Bank ESS W2 files.

String-variable multi-slot handling:
  type_code == -1  → continuation record (adds one more 8-byte slot to
                     the previous variable; needed for long string fields)
  Slot-aware decompression: code 253 reads ASCII for string slots and
  IEEE-754 double for numeric slots. Getting this wrong silently corrupts
  all numeric values that happen to share a record with string variables.
═══════════════════════════════════════════════════════════════════════════════
"""

import struct
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def read_sav(filepath) -> pd.DataFrame:
    """
    Read an SPSS .sav file and return a pandas DataFrame.

    W2_COL_RENAME from config is applied automatically so that downstream
    code (data_loader.load_section) receives standard column names.

    Parameters
    ----------
    filepath : str or Path to the .sav file

    Returns
    -------
    pd.DataFrame with lowercase column names.
    System-missing values (|val| > 1e15) replaced with NaN.

    Raises
    ------
    ValueError           : bad magic bytes or malformed file
    NotImplementedError  : unsupported compression type (not bytecode)
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as fh:
        raw = fh.read()

    if raw[:4] != b"$FL2":
        raise ValueError(f"{filepath.name}: not an SPSS SAV file (bad magic bytes)")

    compress = struct.unpack_from("<i", raw, 72)[0]
    if compress != 1:
        raise NotImplementedError(
            f"{filepath.name}: compression={compress} unsupported; "
            "only bytecode (1) is handled."
        )
    bias = struct.unpack_from("<d", raw, 84)[0]   # always 100.0 for bytecode

    var_names, var_types, var_blocks, data_start = _parse_header(raw)

    if not var_names or data_start is None:
        return pd.DataFrame()

    is_string = _slot_type_map(var_types, var_blocks)
    elements  = _decompress(raw, data_start, is_string, bias)
    rows      = _reshape(elements, var_types, var_blocks)

    df = pd.DataFrame(rows, columns=var_names)
    df = _clean_numeric(df)

    # Apply W2 rename map (imported here to keep sav_reader self-contained)
    from config import W2_COL_RENAME
    return df.rename(columns=W2_COL_RENAME)


# ── Header parser ──────────────────────────────────────────────────────────────

def _parse_header(raw: bytes):
    """
    Walk SPSS record types to collect variable metadata and locate data_start.

    Record types handled:
      2   → variable descriptor (name, type_code, label, missing specs)
      3   → value labels
      4   → value-label indices
      6   → document records
      7   → extension records (long var names, etc.)
      999 → data start marker

    type_code == -1 → string continuation slot (extends previous var's block count)
    type_code ==  0 → numeric variable
    type_code ==  n → string variable of n chars (needs ceil(n/8) slots)
    """
    offset     = 176
    var_names  : List[str] = []
    var_types  : List[int] = []
    var_blocks : List[int] = []
    data_start : Optional[int] = None
    end        = len(raw)

    while offset < end - 4:
        rec_type = struct.unpack_from("<i", raw, offset)[0]

        if rec_type == 2:
            if offset + 32 > end:
                break
            type_code = struct.unpack_from("<i", raw, offset +  4)[0]
            has_label = struct.unpack_from("<i", raw, offset +  8)[0]
            n_miss    = struct.unpack_from("<i", raw, offset + 12)[0]
            raw_name  = raw[offset + 24: offset + 32]
            name      = (raw_name.rstrip(b"\x00 ")
                                 .decode("ascii", "replace")
                                 .strip().lower())
            skip = 32
            if has_label and offset + skip + 4 <= end:
                llen  = struct.unpack_from("<i", raw, offset + skip)[0]
                skip += 4 + ((llen + 3) // 4) * 4
            if n_miss > 0:
                skip += n_miss * 8
            offset += skip

            if type_code == -1:
                if var_blocks:
                    var_blocks[-1] += 1
            elif name and name.isprintable():
                var_names.append(name)
                var_types.append(type_code)
                var_blocks.append(1)

        elif rec_type == 3:
            n = struct.unpack_from("<i", raw, offset + 4)[0]
            offset += 8 + n * 16
        elif rec_type == 4:
            n = struct.unpack_from("<i", raw, offset + 4)[0]
            offset += 8 + n * 4
        elif rec_type == 6:
            n = struct.unpack_from("<i", raw, offset + 4)[0]
            offset += 8 + n * 80
        elif rec_type == 7:
            size_ = struct.unpack_from("<i", raw, offset +  8)[0]
            cnt_  = struct.unpack_from("<i", raw, offset + 12)[0]
            offset += 16 + size_ * cnt_
        elif rec_type == 999:
            data_start = offset + 8
            break
        else:
            offset += 4

    return var_names, var_types, var_blocks, data_start


def _slot_type_map(var_types: List[int], var_blocks: List[int]) -> List[bool]:
    """
    Build a per-slot boolean list: True = string slot, False = numeric slot.
    This is critical for correct decompression of code 253 (raw 8 bytes).
    """
    result: List[bool] = []
    for vt, blk in zip(var_types, var_blocks):
        for _ in range(blk):
            result.append(vt != 0)
    return result


# ── Decompressor ───────────────────────────────────────────────────────────────

def _decompress(raw: bytes, data_start: int,
                is_string: List[bool], bias: float) -> List:
    """
    Decode SPSS bytecode compression.

    Byte codes (each governs one 8-byte slot in the output):
      1–251  → numeric shorthand: float(code) - bias
      253    → raw 8 bytes follow (IEEE-754 double OR ASCII, slot-aware)
      254    → blank: '        ' for string, NaN for numeric
      255    → system-missing → NaN
      0/252  → end-of-block signal (no more data in this code group)
    """
    elements: List  = []
    slot_idx: int   = 0
    n_slots   = len(is_string)
    i         = data_start
    end       = len(raw)

    while i <= end - 8:
        codes = raw[i: i + 8]
        i    += 8

        for code in codes:
            if code in (0, 252):
                return elements

            s = is_string[slot_idx % n_slots] if n_slots else False

            if 1 <= code <= 251:
                elements.append(float(code) - bias)
            elif code == 253:
                if i + 8 > end:
                    return elements
                chunk = raw[i: i + 8]
                elements.append(
                    chunk.decode("ascii", "replace") if s
                    else struct.unpack_from("<d", chunk)[0]
                )
                i += 8
            elif code == 254:
                elements.append("        " if s else np.nan)
            else:  # 255 = system-missing
                elements.append(np.nan)

            slot_idx += 1

    return elements


# ── Row assembler ──────────────────────────────────────────────────────────────

def _reshape(elements: List, var_types: List[int],
             var_blocks: List[int]) -> List[List]:
    """Reshape flat element list into a 2-D list of rows."""
    blocks_per_case = sum(var_blocks)
    if blocks_per_case == 0:
        return []

    n_rows = len(elements) // blocks_per_case
    rows: List[List] = []
    idx = 0

    for _ in range(n_rows):
        row = []
        for vt, blk in zip(var_types, var_blocks):
            if vt == 0:   # numeric
                row.append(elements[idx] if idx < len(elements) else np.nan)
                idx += 1
            else:         # string (possibly multi-slot)
                parts = []
                for _ in range(blk):
                    e = elements[idx] if idx < len(elements) else " " * 8
                    parts.append(e if isinstance(e, str) else " " * 8)
                    idx += 1
                row.append("".join(parts).rstrip())
        rows.append(row)

    return rows


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace SPSS system-missing large-magnitude values with NaN."""
    for col in df.select_dtypes(include="number").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].abs() > 1e15, col] = np.nan
    return df
