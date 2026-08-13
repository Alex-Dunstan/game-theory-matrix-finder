import itertools
import numpy as np
import csv
from typing import List, Tuple, Iterator, Generator


# ---------------------------------------------------------------------------
# Core n×m generation
# ---------------------------------------------------------------------------

def generate_dual_matrices(rows: int, cols: int, min_val: int = 0, max_val: int = 5) -> List[np.ndarray]:
    """
    Generate ALL possible n×m matrices where every cell holds a (player1, player2)
    payoff pair.  Both payoffs range from min_val to max_val inclusive.

    Returns a list of numpy arrays with shape (rows, cols, 2).

    Use generate_dual_matrices_iter() instead when the full list would be too
    large to hold in RAM (e.g. 0-10 range with 214M matrices).
    """
    values = range(min_val, max_val + 1)
    position_pairs = list(itertools.product(values, values))
    num_cells = rows * cols

    matrices = []
    for perm in itertools.product(position_pairs, repeat=num_cells):
        matrix = np.array(perm, dtype=np.int32).reshape(rows, cols, 2)
        matrices.append(matrix)
    return matrices


def generate_dual_matrices_iter(
    rows: int, cols: int, min_val: int = 0, max_val: int = 5
) -> Iterator[np.ndarray]:
    """
    Generator version of generate_dual_matrices().
    Yields one (rows, cols, 2) numpy array at a time — never holds the full
    dataset in memory.  Use this for large ranges (e.g. 0–10).
    """
    values = range(min_val, max_val + 1)
    position_pairs = list(itertools.product(values, values))
    num_cells = rows * cols

    for perm in itertools.product(position_pairs, repeat=num_cells):
        yield np.array(perm, dtype=np.int32).reshape(rows, cols, 2)


def generate_dual_matrices_batched(
    rows: int,
    cols: int,
    min_val: int = 0,
    max_val: int = 5,
    batch_size: int = 50_000,
) -> Generator[np.ndarray, None, None]:
    """
    Yields batches of matrices as a single numpy array of shape (B, rows, cols, 2).

    B = batch_size for all batches except possibly the last one.
    This is the most efficient input form for find_nash_batch_vectorized().

    Example
    -------
    for batch in generate_dual_matrices_batched(2, 2, 0, 5, batch_size=50_000):
        is_ne, counts = find_nash_batch_vectorized(batch)
        # batch.shape == (50_000, 2, 2, 2) for all but the last chunk
    """
    values = range(min_val, max_val + 1)
    position_pairs = list(itertools.product(values, values))
    num_cells = rows * cols
    buf: list[np.ndarray] = []

    for perm in itertools.product(position_pairs, repeat=num_cells):
        buf.append(np.array(perm, dtype=np.int32))
        if len(buf) == batch_size:
            yield np.stack(buf).reshape(batch_size, rows, cols, 2)
            buf = []

    if buf:
        n = len(buf)
        yield np.stack(buf).reshape(n, rows, cols, 2)


# ---------------------------------------------------------------------------
# Nash equilibrium analysis — single matrix (Python loops)
# ---------------------------------------------------------------------------

def check_nash_equilibrium(matrix: np.ndarray, position: Tuple[int, int]) -> bool:
    """
    Return True if (row, col) is a pure-strategy Nash equilibrium.

    A cell is a NE when:
      - The row player cannot improve their payoff by switching to any other row
        (keeping the column fixed).
      - The column player cannot improve their payoff by switching to any other
        column (keeping the row fixed).

    Works for any n×m matrix shape.
    """
    row, col = position
    current_row_payoff = int(matrix[row, col, 0])
    current_col_payoff = int(matrix[row, col, 1])

    # Can the row player do better by moving to a different row?
    for other_row in range(matrix.shape[0]):
        if other_row != row and int(matrix[other_row, col, 0]) > current_row_payoff:
            return False

    # Can the column player do better by moving to a different column?
    for other_col in range(matrix.shape[1]):
        if other_col != col and int(matrix[row, other_col, 1]) > current_col_payoff:
            return False

    return True


def find_nash_equilibria(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find all pure-strategy Nash equilibria in an n×m payoff matrix.
    Returns a (possibly empty) list of (row, col) positions.
    """
    equilibria = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if check_nash_equilibrium(matrix, (row, col)):
                equilibria.append((row, col))
    return equilibria


# ---------------------------------------------------------------------------
# Nash equilibrium analysis — vectorised batch (numpy, no Python loops)
# ---------------------------------------------------------------------------

def find_nash_batch_vectorized(
    batch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find pure-strategy Nash equilibria for a BATCH of matrices in one numpy call.

    Parameters
    ----------
    batch : np.ndarray, shape (N, rows, cols, 2)
        N matrices, each of shape (rows, cols, 2) where the last axis is
        [player1_payoff, player2_payoff].

    Returns
    -------
    is_ne : np.ndarray, shape (N, rows, cols), dtype bool
        True wherever a cell is a Nash equilibrium.
    ne_counts : np.ndarray, shape (N,), dtype int
        Number of Nash equilibria per matrix.

    How it works
    ------------
    A cell (r, c) in matrix n is a NE when:
      • P1 cannot strictly improve by switching rows:
            p1[n, r, c] == max over r' of p1[n, r', c]
      • P2 cannot strictly improve by switching columns:
            p2[n, r, c] == max over c' of p2[n, r, c']

    Both conditions are expressed as a single element-wise equality after
    broadcasting the per-column and per-row maxima — no Python loops needed.

    This is equivalent to find_nash_equilibria() but ~10–15× faster when
    processing large batches (e.g. 50 000 matrices at a time).
    """
    p1 = batch[:, :, :, 0]                          # (N, R, C)
    p2 = batch[:, :, :, 1]                          # (N, R, C)

    # Best P1 can achieve in each column (broadcast over rows dimension)
    p1_col_max = p1.max(axis=1, keepdims=True)      # (N, 1, C)
    # Best P2 can achieve in each row (broadcast over cols dimension)
    p2_row_max = p2.max(axis=2, keepdims=True)      # (N, R, 1)

    # A cell is NE iff both players are already at their column/row maximum
    is_ne = (p1 == p1_col_max) & (p2 == p2_row_max)  # (N, R, C)
    ne_counts = is_ne.sum(axis=(1, 2)).astype(np.int32)  # (N,)

    return is_ne, ne_counts


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _build_headers(rows: int, cols: int) -> List[str]:
    """Return column headers for a rows×cols payoff matrix CSV."""
    headers = []
    for r in range(rows):
        for c in range(cols):
            headers += [f"r{r}c{c}_p1", f"r{r}c{c}_p2"]
    headers += ["num_equilibria", "equilibrium_positions", "category"]
    return headers


def save_dual_matrices_to_csv(matrices: List[np.ndarray], filename: str) -> None:
    """
    Analyse each matrix for Nash equilibria and write results to a CSV file.

    CSV columns (2×2 example):
        r0c0_p1, r0c0_p2, r0c1_p1, r0c1_p2,
        r1c0_p1, r1c0_p2, r1c1_p1, r1c1_p2,
        num_equilibria, equilibrium_positions, category

    Args:
        matrices: list of numpy arrays with shape (rows, cols, 2)
        filename: output CSV path
    """
    if not matrices:
        return

    rows, cols = matrices[0].shape[0], matrices[0].shape[1]
    headers = _build_headers(rows, cols)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for matrix in matrices:
            _write_matrix_row(writer, matrix)


def save_dual_matrices_iter_to_csv(
    matrix_iter: Iterator[np.ndarray],
    filename: str,
    rows: int,
    cols: int,
    progress_interval: int = 1_000_000,
) -> int:
    """
    Stream matrices from an iterator directly into a CSV — constant RAM usage
    regardless of dataset size.  Prints progress every progress_interval rows.

    Returns the total number of matrices written.
    """
    headers = _build_headers(rows, cols)
    count = 0

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for matrix in matrix_iter:
            _write_matrix_row(writer, matrix)
            count += 1
            if count % progress_interval == 0:
                print(f"  {count:,} matrices written…")

    return count


def save_batched_to_csv(
    rows: int,
    cols: int,
    min_val: int,
    max_val: int,
    filename: str,
    batch_size: int = 50_000,
    progress_interval: int = 1_000_000,
) -> int:
    """
    Generate all matrices for (rows, cols, min_val, max_val) and write to CSV
    using batched vectorised NE detection.  Streams to disk — constant RAM.

    Significantly faster than save_dual_matrices_iter_to_csv() for large ranges.
    Returns total number of matrices written.
    """
    headers = _build_headers(rows, cols)
    count = 0

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for batch in generate_dual_matrices_batched(rows, cols, min_val, max_val, batch_size):
            is_ne, ne_counts = find_nash_batch_vectorized(batch)
            B = len(batch)

            for i in range(B):
                flat = batch[i].reshape(-1).tolist()
                n_eq = int(ne_counts[i])
                positions = [
                    (r, c)
                    for r in range(rows)
                    for c in range(cols)
                    if is_ne[i, r, c]
                ]
                category = "Solved" if n_eq > 0 else "Unsolved"
                writer.writerow(
                    flat + [n_eq, str(positions) if positions else "None", category]
                )
                count += 1
                if count % progress_interval == 0:
                    print(f"  {count:,} matrices written…")

    return count


def _write_matrix_row(writer: csv.writer, matrix: np.ndarray) -> None:
    """Write one matrix as a CSV row (helper used by both save functions)."""
    flat_values = matrix.reshape(-1).tolist()
    equilibria = find_nash_equilibria(matrix)
    category = "Solved" if equilibria else "Unsolved"
    writer.writerow(
        flat_values
        + [len(equilibria), str(equilibria) if equilibria else "None", category]
    )


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------

def print_dual_matrices(matrices: List[np.ndarray]) -> None:
    """Print matrices in a human-readable format."""
    for i, matrix in enumerate(matrices):
        print(f"Matrix {i + 1}:")
        for row in matrix:
            print([f"({pair[0]},{pair[1]})" for pair in row])
        print()


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------

def generate_2x2_dual_matrices(min_val: int = 0, max_val: int = 5) -> List[np.ndarray]:
    """Alias for generate_dual_matrices(2, 2, ...) — kept for compatibility."""
    return generate_dual_matrices(2, 2, min_val, max_val)
