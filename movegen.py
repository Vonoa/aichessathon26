"""A numba-jitted bitboard move generator, built and validated on its own before it
touches the search (docs/PLAN.md, Phase 3 -- "the hardest phase").

The engine currently pays python-chess for `board.legal_moves` + `board.push` / `pop`
at every node -- ~30-50 us of Python. This module replaces that inner loop with jitted
bitboard arithmetic on a compact representation.

Phase A (this file): pseudo-legal generation, make / unmake, legal generation and a
jitted `perft`, validated node-for-node against python-chess. No search integration yet
-- a movegen bug is a lost game, so it does not go near `search.py` until perft and the
legal-move sets match exactly.

Board representation
    bb     uint64[2, 6]  -- bb[colour, piece_type - 1]; colour 0 = White, 1 = Black;
                            piece_type 1..6 = P N B R Q K (same layout as evaluate._encode)
    state  int64[4]      -- [turn, castling, ep_square, halfmove]
                            turn      0 = White to move, 1 = Black
                            castling  bit0 WK, bit1 WQ, bit2 BK, bit3 BQ
                            ep_square 0..63, or -1
                            halfmove  plies since the last pawn move or capture

Move encoding (int32)
    bits  0-5   from square
    bits  6-11  to square
    bits 12-14  promotion piece (0 none, 1 N, 2 B, 3 R, 4 Q)
    bits 15-17  flag (0 normal / capture, 1 double pawn push, 2 en passant, 3 castle)
"""

from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt
from numba import njit

# --- constants ---------------------------------------------------------------

_U1 = np.uint64(1)

_BIT: npt.NDArray[np.uint64] = np.array([1 << i for i in range(64)], dtype=np.uint64)
_KNIGHT_ATK: npt.NDArray[np.uint64] = np.array(chess.BB_KNIGHT_ATTACKS, dtype=np.uint64)
_KING_ATK: npt.NDArray[np.uint64] = np.array(chess.BB_KING_ATTACKS, dtype=np.uint64)
_WPAWN_ATK: npt.NDArray[np.uint64] = np.array(
    list(chess.BB_PAWN_ATTACKS[chess.WHITE]), dtype=np.uint64
)
_BPAWN_ATK: npt.NDArray[np.uint64] = np.array(
    list(chess.BB_PAWN_ATTACKS[chess.BLACK]), dtype=np.uint64
)
_BISHOP_DIRS: npt.NDArray[np.int64] = np.array(
    [(1, 1), (1, -1), (-1, 1), (-1, -1)], dtype=np.int64
)
_ROOK_DIRS: npt.NDArray[np.int64] = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int64)

_MAX_MOVES = 256


# --- jitted primitives -----------------------------------------------------------


@njit(cache=False)
def _popcount(bb: np.uint64) -> np.int64:
    bb = bb - ((bb >> _U1) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + (
        (bb >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((bb * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=False)
def _lsb_sq(bb: np.uint64) -> int:
    """Square index of the lowest set bit."""
    return int(_popcount((bb & (~bb + _U1)) - _U1))


@njit(cache=False)
def _ray(occ: np.uint64, sq: int, df: int, dr: int) -> np.uint64:
    """Squares along one direction from `sq` up to and including the first blocker."""
    attacks = np.uint64(0)
    f = (sq & 7) + df
    r = (sq >> 3) + dr
    while 0 <= f <= 7 and 0 <= r <= 7:
        bit = _BIT[r * 8 + f]
        attacks |= bit
        if (occ & bit) != np.uint64(0):
            break
        f += df
        r += dr
    return attacks


@njit(cache=False)
def _bishop_atk(occ: np.uint64, sq: int) -> np.uint64:
    a = np.uint64(0)
    for i in range(4):
        a |= _ray(occ, sq, _BISHOP_DIRS[i, 0], _BISHOP_DIRS[i, 1])
    return a


@njit(cache=False)
def _rook_atk(occ: np.uint64, sq: int) -> np.uint64:
    a = np.uint64(0)
    for i in range(4):
        a |= _ray(occ, sq, _ROOK_DIRS[i, 0], _ROOK_DIRS[i, 1])
    return a


@njit(cache=False)
def _occ_of(bb: npt.NDArray[np.uint64], colour: int) -> np.uint64:
    return np.uint64(
        bb[colour, 0] | bb[colour, 1] | bb[colour, 2]
        | bb[colour, 3] | bb[colour, 4] | bb[colour, 5]
    )


@njit(cache=False)
def _king_sq(bb: npt.NDArray[np.uint64], colour: int) -> int:
    return _lsb_sq(bb[colour, 5])


@njit(cache=False)
def _occ3(bb: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64]) -> None:
    """Fill `out` (len 3) with [white occ, black occ, all occ] -- the shape evaluate's
    jitted path expects."""
    out[0] = _occ_of(bb, 0)
    out[1] = _occ_of(bb, 1)
    out[2] = out[0] | out[1]


# --- Zobrist hashing ---------------------------------------------------------------
#
# One 64-bit key per position, so the search's transposition table and repetition
# detection can work off (bb, state) without a python-chess board. Deterministically
# seeded -- the finals panel needs a reproducible engine.
_zrng = np.random.default_rng(0x5C4013A7)
_ZOBRIST_PIECE: npt.NDArray[np.uint64] = _zrng.integers(
    0, 1 << 64, size=(2, 6, 64), dtype=np.uint64
)
_ZOBRIST_CASTLING: npt.NDArray[np.uint64] = _zrng.integers(0, 1 << 64, size=16, dtype=np.uint64)
_ZOBRIST_EP: npt.NDArray[np.uint64] = _zrng.integers(0, 1 << 64, size=8, dtype=np.uint64)
_ZOBRIST_TURN = np.uint64(_zrng.integers(0, 1 << 64, dtype=np.uint64))


@njit(cache=False)
def _zobrist(bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64]) -> np.uint64:
    h = np.uint64(0)
    for c in range(2):
        for pt in range(6):
            b = bb[c, pt]
            while b != np.uint64(0):
                h ^= _ZOBRIST_PIECE[c, pt, _lsb_sq(b)]
                b &= b - _U1
    h ^= _ZOBRIST_CASTLING[int(state[1]) & 15]
    ep = int(state[2])
    if ep >= 0:
        turn = int(state[0])
        # matches python-chess: the ep square only distinguishes the position if the side
        # to move actually has a pawn that could capture there
        capper = _BPAWN_ATK[ep] if turn == 0 else _WPAWN_ATK[ep]
        if (capper & bb[turn, 0]) != np.uint64(0):
            h ^= _ZOBRIST_EP[ep & 7]
    if int(state[0]) == 1:
        h ^= _ZOBRIST_TURN
    return np.uint64(h)


@njit(cache=False)
def _attacked_by(bb: npt.NDArray[np.uint64], occ: np.uint64, sq: int, by: int) -> bool:
    """Is `sq` attacked by any piece of colour `by`, given occupancy `occ`?"""
    if (_KNIGHT_ATK[sq] & bb[by, 1]) != np.uint64(0):
        return True
    if (_KING_ATK[sq] & bb[by, 5]) != np.uint64(0):
        return True
    # a pawn of colour `by` attacks `sq` iff it stands where the other colour's pawn on
    # `sq` would attack
    pawn_from = _BPAWN_ATK[sq] if by == 0 else _WPAWN_ATK[sq]
    if (pawn_from & bb[by, 0]) != np.uint64(0):
        return True
    if (_bishop_atk(occ, sq) & (bb[by, 2] | bb[by, 4])) != np.uint64(0):
        return True
    return bool((_rook_atk(occ, sq) & (bb[by, 3] | bb[by, 4])) != np.uint64(0))


@njit(cache=False)
def _mv(frm: int, to: int, promo: int, flag: int) -> int:
    return frm | (to << 6) | (promo << 12) | (flag << 15)


# --- pseudo-legal generation ---------------------------------------------------


@njit(cache=False)
def _gen(
    bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64], out: npt.NDArray[np.int32]
) -> int:
    """Fill `out` with pseudo-legal moves for the side to move; return the count. A move
    may leave the mover's king in check -- the caller filters that."""
    turn = int(state[0])
    castling = int(state[1])
    ep = int(state[2])
    own = _occ_of(bb, turn)
    enemy = _occ_of(bb, 1 - turn)
    occ = own | enemy
    n = 0

    # --- pawns ---
    if turn == 0:
        start_rank = 1
        promo_rank = 7
        cap_table = _WPAWN_ATK
    else:
        start_rank = 6
        promo_rank = 0
        cap_table = _BPAWN_ATK
    p = bb[turn, 0]
    while p != np.uint64(0):
        frm = _lsb_sq(p)
        p &= p - _U1
        one = frm + 8 if turn == 0 else frm - 8
        if (occ & _BIT[one]) == np.uint64(0):
            if (one >> 3) == promo_rank:
                for promo in (4, 3, 2, 1):
                    out[n] = _mv(frm, one, promo, 0)
                    n += 1
            else:
                out[n] = _mv(frm, one, 0, 0)
                n += 1
                if (frm >> 3) == start_rank:
                    two = frm + 16 if turn == 0 else frm - 16
                    if (occ & _BIT[two]) == np.uint64(0):
                        out[n] = _mv(frm, two, 0, 1)
                        n += 1
        targets = cap_table[frm] & enemy
        while targets != np.uint64(0):
            to = _lsb_sq(targets)
            targets &= targets - _U1
            if (to >> 3) == promo_rank:
                for promo in (4, 3, 2, 1):
                    out[n] = _mv(frm, to, promo, 0)
                    n += 1
            else:
                out[n] = _mv(frm, to, 0, 0)
                n += 1
        if ep >= 0 and (cap_table[frm] & _BIT[ep]) != np.uint64(0):
            out[n] = _mv(frm, ep, 0, 2)
            n += 1

    # --- knights ---
    kn = bb[turn, 1]
    while kn != np.uint64(0):
        frm = _lsb_sq(kn)
        kn &= kn - _U1
        t = _KNIGHT_ATK[frm] & ~own
        while t != np.uint64(0):
            out[n] = _mv(frm, _lsb_sq(t), 0, 0)
            n += 1
            t &= t - _U1

    # --- bishops / rooks / queens ---
    for pt in range(2, 5):
        pieces = bb[turn, pt]
        while pieces != np.uint64(0):
            frm = _lsb_sq(pieces)
            pieces &= pieces - _U1
            if pt == 2:
                t = _bishop_atk(occ, frm) & ~own
            elif pt == 3:
                t = _rook_atk(occ, frm) & ~own
            else:
                t = (_bishop_atk(occ, frm) | _rook_atk(occ, frm)) & ~own
            while t != np.uint64(0):
                out[n] = _mv(frm, _lsb_sq(t), 0, 0)
                n += 1
                t &= t - _U1

    # --- king ---
    ksq = _king_sq(bb, turn)
    t = _KING_ATK[ksq] & ~own
    while t != np.uint64(0):
        out[n] = _mv(ksq, _lsb_sq(t), 0, 0)
        n += 1
        t &= t - _U1

    # --- castling: right held, squares between empty, king start + path not attacked ---
    if (
        turn == 0
        and (castling & 1)
        and (occ & (_BIT[5] | _BIT[6])) == np.uint64(0)
        and not _attacked_by(bb, occ, 4, 1)
        and not _attacked_by(bb, occ, 5, 1)
        and not _attacked_by(bb, occ, 6, 1)
    ):
        out[n] = _mv(4, 6, 0, 3)
        n += 1
    if (
        turn == 0
        and (castling & 2)
        and (occ & (_BIT[1] | _BIT[2] | _BIT[3])) == np.uint64(0)
        and not _attacked_by(bb, occ, 4, 1)
        and not _attacked_by(bb, occ, 3, 1)
        and not _attacked_by(bb, occ, 2, 1)
    ):
        out[n] = _mv(4, 2, 0, 3)
        n += 1
    if (
        turn == 1
        and (castling & 4)
        and (occ & (_BIT[61] | _BIT[62])) == np.uint64(0)
        and not _attacked_by(bb, occ, 60, 0)
        and not _attacked_by(bb, occ, 61, 0)
        and not _attacked_by(bb, occ, 62, 0)
    ):
        out[n] = _mv(60, 62, 0, 3)
        n += 1
    if (
        turn == 1
        and (castling & 8)
        and (occ & (_BIT[57] | _BIT[58] | _BIT[59])) == np.uint64(0)
        and not _attacked_by(bb, occ, 60, 0)
        and not _attacked_by(bb, occ, 59, 0)
        and not _attacked_by(bb, occ, 58, 0)
    ):
        out[n] = _mv(60, 58, 0, 3)
        n += 1

    return n


# --- make / unmake -----------------------------------------------------------


@njit(cache=False)
def _piece_at(bb: npt.NDArray[np.uint64], colour: int, sq: int) -> int:
    """1..6 piece type of `colour` on `sq`, or 0."""
    b = _BIT[sq]
    for pt in range(6):
        if (bb[colour, pt] & b) != np.uint64(0):
            return pt + 1
    return 0


@njit(cache=False)
def _rook_hop_from(to: int) -> int:
    if to == 6:
        return 7
    if to == 2:
        return 0
    if to == 62:
        return 63
    return 56


@njit(cache=False)
def _rook_hop_to(to: int) -> int:
    if to == 6:
        return 5
    if to == 2:
        return 3
    if to == 62:
        return 61
    return 59


@njit(cache=False)
def _make(
    bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64], mv: int
) -> int:
    """Apply `mv` in place; return an undo word for _unmake."""
    m = int(mv)
    frm = m & 0x3F
    to = (m >> 6) & 0x3F
    promo = (m >> 12) & 7
    flag = (m >> 15) & 7
    mover = int(state[0])
    enemy = 1 - mover

    old_castling = int(state[1])
    old_ep = int(state[2])
    old_half = int(state[3])
    movpt = _piece_at(bb, mover, frm)

    if flag == 2:
        cap_sq = to - 8 if mover == 0 else to + 8
        cappt = 1
    else:
        cap_sq = to
        cappt = _piece_at(bb, enemy, to)

    undo = cappt | (cap_sq << 3) | ((old_ep + 1) << 9) | (old_castling << 16) | (old_half << 20)

    if cappt != 0:
        bb[enemy, cappt - 1] &= ~_BIT[cap_sq]

    bb[mover, movpt - 1] &= ~_BIT[frm]
    if promo != 0:
        bb[mover, promo] |= _BIT[to]
    else:
        bb[mover, movpt - 1] |= _BIT[to]

    if flag == 3:
        rf = _rook_hop_from(to)
        rt = _rook_hop_to(to)
        bb[mover, 3] &= ~_BIT[rf]
        bb[mover, 3] |= _BIT[rt]

    castling = old_castling
    if movpt == 6:
        castling &= ~(3 if mover == 0 else 12)
    if movpt == 4:
        if frm == 7:
            castling &= ~1
        elif frm == 0:
            castling &= ~2
        elif frm == 63:
            castling &= ~4
        elif frm == 56:
            castling &= ~8
    if cappt == 4:
        if cap_sq == 7:
            castling &= ~1
        elif cap_sq == 0:
            castling &= ~2
        elif cap_sq == 63:
            castling &= ~4
        elif cap_sq == 56:
            castling &= ~8
    state[1] = castling

    state[2] = (frm + to) // 2 if flag == 1 else -1
    state[3] = 0 if (movpt == 1 or cappt != 0) else old_half + 1
    state[0] = enemy
    return undo


@njit(cache=False)
def _unmake(
    bb: npt.NDArray[np.uint64],
    state: npt.NDArray[np.int64],
    mv: int,
    undo: int,
) -> None:
    m = int(mv)
    frm = m & 0x3F
    to = (m >> 6) & 0x3F
    promo = (m >> 12) & 7
    flag = (m >> 15) & 7
    u = int(undo)
    cappt = u & 7
    cap_sq = (u >> 3) & 0x3F
    old_ep = ((u >> 9) & 0x7F) - 1
    old_castling = (u >> 16) & 0xF
    old_half = (u >> 20) & 0x7F

    mover = 1 - int(state[0])
    enemy = 1 - mover

    if promo != 0:
        bb[mover, promo] &= ~_BIT[to]
        bb[mover, 0] |= _BIT[frm]
    else:
        movpt = _piece_at(bb, mover, to)
        bb[mover, movpt - 1] &= ~_BIT[to]
        bb[mover, movpt - 1] |= _BIT[frm]

    if flag == 3:
        rf = _rook_hop_from(to)
        rt = _rook_hop_to(to)
        bb[mover, 3] &= ~_BIT[rt]
        bb[mover, 3] |= _BIT[rf]

    if cappt != 0:
        bb[enemy, cappt - 1] |= _BIT[cap_sq]

    state[0] = mover
    state[1] = old_castling
    state[2] = old_ep
    state[3] = old_half


# --- legal generation + perft --------------------------------------------------


@njit(cache=False)
def _gen_legal(
    bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64], out: npt.NDArray[np.int32]
) -> int:
    """Fill `out` with fully legal moves for the side to move; return the count."""
    pseudo = np.empty(_MAX_MOVES, dtype=np.int32)
    m = _gen(bb, state, pseudo)
    mover = int(state[0])
    n = 0
    for i in range(m):
        undo = _make(bb, state, pseudo[i])
        occ = _occ_of(bb, 0) | _occ_of(bb, 1)
        if not _attacked_by(bb, occ, _king_sq(bb, mover), 1 - mover):
            out[n] = pseudo[i]
            n += 1
        _unmake(bb, state, pseudo[i], undo)
    return n


@njit(cache=False)
def _perft(bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64], depth: int) -> int:
    if depth == 0:
        return 1
    out = np.empty(_MAX_MOVES, dtype=np.int32)
    n = _gen(bb, state, out)
    total = 0
    mover = int(state[0])
    for i in range(n):
        undo = _make(bb, state, out[i])
        occ = _occ_of(bb, 0) | _occ_of(bb, 1)
        if not _attacked_by(bb, occ, _king_sq(bb, mover), 1 - mover):
            total += 1 if depth == 1 else _perft(bb, state, depth - 1)
        _unmake(bb, state, out[i], undo)
    return total


# --- python glue (not jitted) --------------------------------------------------

_PROMO_FROM_CODE: tuple[int | None, ...] = (
    None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN,
)


def encode(board: chess.Board) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """chess.Board -> (bb, state) for the jitted routines."""
    bb: npt.NDArray[np.uint64] = np.zeros((2, 6), dtype=np.uint64)
    for pt in range(1, 7):
        bb[0, pt - 1] = board.pieces_mask(pt, chess.WHITE)
        bb[1, pt - 1] = board.pieces_mask(pt, chess.BLACK)
    castling = 0
    if board.castling_rights & chess.BB_H1:
        castling |= 1
    if board.castling_rights & chess.BB_A1:
        castling |= 2
    if board.castling_rights & chess.BB_H8:
        castling |= 4
    if board.castling_rights & chess.BB_A8:
        castling |= 8
    ep = board.ep_square if board.ep_square is not None else -1
    turn = 0 if board.turn == chess.WHITE else 1
    state: npt.NDArray[np.int64] = np.array(
        [turn, castling, ep, board.halfmove_clock], dtype=np.int64
    )
    return bb, state


def decode_move(code: int) -> chess.Move:
    frm = code & 0x3F
    to = (code >> 6) & 0x3F
    promo = (code >> 12) & 7
    return chess.Move(frm, to, promotion=_PROMO_FROM_CODE[promo] if promo else None)


def perft(board: chess.Board, depth: int) -> int:
    bb, state = encode(board)
    return int(_perft(bb, state, depth))


def zobrist(board: chess.Board) -> int:
    """The 64-bit position key for a chess.Board -- used to seed the search's history so
    repetition detection can run off the bitboard state."""
    bb, state = encode(board)
    return int(_zobrist(bb, state))


def legal_ucis(board: chess.Board) -> set[str]:
    """The jitted legal-move set as UCI strings, for validating against python-chess."""
    bb, state = encode(board)
    out: npt.NDArray[np.int32] = np.empty(_MAX_MOVES, dtype=np.int32)
    n = _gen_legal(bb, state, out)
    return {decode_move(int(out[i])).uci() for i in range(n)}


def perft_reference(board: chess.Board, depth: int) -> int:
    """python-chess perft -- the oracle the jitted version is checked against."""
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += perft_reference(board, depth - 1)
        board.pop()
    return total


def _warm_up() -> None:
    """Compile the jitted routines the search will use, at import, with the real dtypes.
    `_perft` is deliberately not warmed here -- it is a recursive validation-only routine
    and compiling it adds ~15 s; it compiles on its first call from a test or a script.
    """
    bb, state = encode(chess.Board())
    out: npt.NDArray[np.int32] = np.empty(_MAX_MOVES, dtype=np.int32)
    occ3: npt.NDArray[np.uint64] = np.empty(3, dtype=np.uint64)
    _gen(bb, state, out)
    _gen_legal(bb, state, out)
    _occ3(bb, occ3)
    _zobrist(bb, state)
    undo = _make(bb, state, out[0])
    _unmake(bb, state, out[0], undo)


_warm_up()
