#!/usr/bin/env python3
import itertools
import json
import sys
import numpy as np

p = 17
n = 32
m = 34
t = 16
s = 4
no = n - t - s

def rref_inplace(A, ncols):
    r, piv = 0, []
    for col in range(ncols):
        if r >= A.shape[0]:
            break
        nz = np.nonzero(A[r:, col])[0]
        if len(nz) == 0:
            continue
        k = r + nz[0]
        A[[r, k]] = A[[k, r]]
        A[r] = A[r] * pow(int(A[r, col]), -1, p) % p
        f = A[:, col].copy()
        f[r] = 0
        idx = np.nonzero(f)[0]
        A[idx] = (A[idx] - np.outer(f[idx], A[r])) % p
        piv.append(col)
        r += 1
    return A, piv

def rank(M):
    A = np.array(M, dtype=np.int64) % p
    return len(rref_inplace(A, A.shape[1])[1])

def nullspace(M):
    A = np.array(M, dtype=np.int64) % p
    c = A.shape[1]
    A, piv = rref_inplace(A, c)
    free = [j for j in range(c) if j not in piv]
    out = []
    for fj in free:
        v = np.zeros(c, dtype=np.int64)
        v[fj] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-A[i, fj]) % p
        out.append(v)
    return np.array(out, dtype=np.int64).reshape(len(out), c)

def inv_mat(M):
    k = len(M)
    A = np.concatenate([np.array(M, dtype=np.int64) % p, np.eye(k, dtype=np.int64)], 1)
    A, piv = rref_inplace(A, k)
    if len(piv) != k:
        raise ValueError("ma tran suy bien")
    return A[:, k:]

def extend_to_basis(rows, dim, count):
    cur, added = [r for r in rows], []
    for i in range(dim):
        e = np.zeros(dim, dtype=np.int64)
        e[i] = 1
        if rank(np.array(cur + [e])) == len(cur) + 1:
            cur.append(e)
            added.append(e)
        if len(cur) == count:
            break
    return np.array(added)

def independent_rows(rows):
    B = []
    for r in rows:
        if rank(np.array(B + [r])) == len(B) + 1:
            B.append(r)
    return np.array(B)

def load(path):
    d = json.load(open(path))
    return d["public_key"]["polynomials"], d["ciphertext"]["blocks"]

def build_index(poly):
    idx = np.full((len(poly), 4), n, dtype=np.int64)
    co = np.zeros(len(poly), dtype=np.int64)
    for i, (c, mon) in enumerate(poly):
        for q, v in enumerate(mon):
            idx[i, q] = v
        co[i] = c
    return idx, co

def eval_public(index, X):
    N = len(X)
    Xe = np.concatenate([X, np.ones((N, 1), dtype=np.int64)], 1)
    out = np.zeros((N, m), dtype=np.int64)
    for k, (idx, co) in enumerate(index):
        for st in range(0, N, 100):
            xs = Xe[st:st + 100]
            v = (xs[:, idx[:, 0]] * xs[:, idx[:, 1]] % p
                 * xs[:, idx[:, 2]] % p * xs[:, idx[:, 3]] % p)
            out[st:st + 100, k] = (v @ co) % p
    return out

def step1_low_degree(polys):
    mons = {}
    for k, P in enumerate(polys):
        for c, mon in P:
            if len(mon) >= 3:
                mons.setdefault(tuple(mon), []).append((k, c))
    D = np.zeros((len(mons), m), dtype=np.int64)
    for i, lst in enumerate(mons.values()):
        for k, c in lst:
            D[i, k] = c
    lam = nullspace(D)
    assert lam.shape[0] == t
    return lam

def step2_coordinates(polys, lam):
    inv2 = pow(2, -1, p)
    Q = np.zeros((t, n, n), dtype=np.int64)
    lin = np.zeros((t, n), dtype=np.int64)
    const = np.zeros(t, dtype=np.int64)
    for k, P in enumerate(polys):
        for c, mon in P:
            L = len(mon)
            if L > 2:
                continue
            for a in range(t):
                l = int(lam[a, k])
                if not l:
                    continue
                if L == 0:
                    const[a] = (const[a] + l * c) % p
                elif L == 1:
                    lin[a, mon[0]] = (lin[a, mon[0]] + l * c) % p
                else:
                    i, j = mon
                    if i == j:
                        Q[a, i, i] = (Q[a, i, i] + l * c) % p
                    else:
                        v = l * c * inv2 % p
                        Q[a, i, j] = (Q[a, i, j] + v) % p
                        Q[a, j, i] = (Q[a, j, i] + v) % p

    stack = Q.reshape(t * n, n)
    K = nullspace(stack)
    Lsp = nullspace(K)
    assert Lsp.shape[0] == n - t

    E = extend_to_basis(list(Lsp), n, n)
    R = np.vstack([E, Lsp])
    Rinv = inv_mat(R)

    Qp = np.array([Rinv.T @ Q[a] @ Rinv % p for a in range(t)])
    linp = lin @ Rinv % p
    assert not Qp[:, :t, :].any() and not Qp[:, :, :t].any()
    alpha, beta, S = linp[:, :t], linp[:, t:], Qp[:, t:, t:]
    return dict(const=const, Lsp=Lsp, R=R, Rinv=Rinv,
                alpha_inv=inv_mat(alpha), beta=beta, S=S)

def compute_W(Y, lam, st):
    return (st["alpha_inv"] @ ((Y @ lam.T - st["const"]).T % p) % p).T % p

def monomial_names():
    names = [()] + [(i,) for i in range(n)]
    names += [(i, j) for i in range(n) for j in range(i, n)]
    return names

def step3_regression(index, lam, st, npoints=650, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, p, size=(npoints, n))
    Y = eval_public(index, X)
    uu = X @ st["Lsp"].T % p
    Z = np.concatenate([compute_W(Y, lam, st), uu], 1)
    cols = [np.ones(npoints, dtype=np.int64)] + [Z[:, i] for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            cols.append(Z[:, i] * Z[:, j] % p)
    F = np.array(cols).T % p
    A = np.concatenate([F, Y], 1) % p
    A, piv = rref_inplace(A, F.shape[1])
    assert len(piv) == F.shape[1]
    assert not A[len(piv):, F.shape[1]:].any()
    C = np.zeros((F.shape[1], m), dtype=np.int64)
    for i, pc in enumerate(piv):
        C[pc] = A[i, F.shape[1]:]
    return C, {mn: i for i, mn in enumerate(monomial_names())}

def step4_oil_space(C, nidx, seed=5):
    inv2 = pow(2, -1, p)
    N = np.zeros((m, t, t), dtype=np.int64)
    for i in range(t):
        for j in range(i, t):
            c = C[nidx[(t + i, t + j)]]
            if i == j:
                N[:, i, i] = c
            else:
                N[:, i, j] = c * inv2 % p
                N[:, j, i] = c * inv2 % p
    rng = np.random.default_rng(seed)
    kers = []
    for _ in range(3):
        M = np.tensordot(rng.integers(0, p, m), N, 1) % p
        kers.extend(nullspace(M))
    O = independent_rows(kers)
    assert O.shape[0] == no
    assert all(not (O @ N[k] @ O.T % p).any() for k in range(m))
    return N, O

def solve_block(y, lam, st, C, nidx, N, O, Pa):
    y = np.array(y, dtype=np.int64)
    Po = O.T
    inv_tab = np.array([0] + [pow(i, -1, p) for i in range(1, p)], dtype=np.int64)
    W = compute_W(y[None, :], lam, st)[0]

    c0 = C[nidx[()]].copy()
    for i in range(t):
        c0 = (c0 + C[nidx[(i,)]] * W[i]) % p
        for j in range(i, t):
            c0 = (c0 + C[nidx[(i, j)]] * W[i] * W[j]) % p
    lin = np.zeros((t, m), dtype=np.int64)
    for l in range(t):
        lin[l] = C[nidx[(t + l,)]]
        for i in range(t):
            lin[l] = (lin[l] + C[nidx[(i, t + l)]] * W[i]) % p
    c0 = (c0 - y) % p

    def g(U):
        return (c0[None, :] + U @ lin + np.einsum("bi,kij,bj->bk", U, N, U)) % p

    sols = []
    A_all = np.array(list(itertools.product(range(p), repeat=s)), dtype=np.int64)
    for st0 in range(0, len(A_all), 20000):
        A = A_all[st0:st0 + 20000]
        B = len(A)
        base = A @ Pa.T % p
        g0 = g(base)
        aug = np.zeros((B, m, no + 1), dtype=np.int64)
        for l in range(no):
            aug[:, :, l] = (g(base + Po[:, l][None, :]) - g0) % p
        aug[:, :, no] = (-g0) % p
        ok, ar = np.ones(B, bool), np.arange(B)
        for col in range(no):
            sub = aug[:, col:, col] != 0
            ok &= sub.any(1)
            first = np.argmax(sub, 1) + col
            tmp = aug[ar, first].copy()
            aug[ar, first] = aug[:, col].copy()
            aug[:, col] = tmp
            aug[:, col] = aug[:, col] * inv_tab[aug[:, col, col]][:, None] % p
            f = aug[:, :, col].copy()
            f[:, col] = 0
            aug = (aug - f[:, :, None] * aug[:, col][:, None, :]) % p
        ok &= ~aug[:, no:, no].any(1)
        for b in np.nonzero(ok)[0]:
            uu = (Pa @ A[b] + Po @ aug[b, :no, no]) % p
            q = np.array([uu @ st["S"][k] @ uu for k in range(t)]) % p
            ee = (W - st["alpha_inv"] @ (st["beta"] @ uu % p) - st["alpha_inv"] @ q) % p
            sols.append(st["Rinv"] @ np.concatenate([ee, uu]) % p)
    return sols

def decode_frame(digits):
    width = 1
    while p ** width < 256:
        width += 1
    def byte_at(start):
        v = 0
        for d in digits[start:start + width]:
            v = v * p + d
        if v > 255:
            raise ValueError("khong phai byte")
        return v
    length = int.from_bytes(bytes(byte_at(i * width) for i in range(4)), "big")
    end = (4 + length) * width
    return bytes(byte_at(i) for i in range(4 * width, end, width))

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out.txt"
    polys, blocks = load(path)
    index = [build_index(P) for P in polys]

    lam = step1_low_degree(polys)
    st = step2_coordinates(polys, lam)
    C, nidx = step3_regression(index, lam, st)
    N, O = step4_oil_space(C, nidx)
    Pa = extend_to_basis(list(O), t, t).T

    digits = []
    for bi, y in enumerate(blocks):
        sols = solve_block(y, lam, st, C, nidx, N, O, Pa)
        assert len(sols) == 1
        digits += [int(v) for v in sols[0]]

    print(decode_frame(digits).decode())

if __name__ == "__main__":
    main()