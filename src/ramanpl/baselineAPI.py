from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import warnings
import numpy as np
from numpy.polynomial import Polynomial
from scipy.ndimage import gaussian_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .schema import baseline_spec_to_runtime, normalise_baseline_spec


# --- Shared Whittaker helpers ------------------------------------------------

def _whittaker_diff_matrix(n: int, diff_order: int) -> sparse.csc_matrix:
    """Build the difference operator D of shape (n-diff_order, n) as csc."""
    if diff_order == 2 and n > 2:
        return sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc")
    # General: compose first-order difference operators D_{k-1} @ ... @ D_1
    D = sparse.diags([-1, 1], [0, 1], shape=(n - 1, n), format="csc")
    for _ in range(diff_order - 1):
        n_cur = D.shape[0]
        D = sparse.diags([-1, 1], [0, 1], shape=(n_cur - 1, n_cur), format="csc") @ D
    return D.tocsc()


# Cache D^T D by (n, diff_order) — invariant across calls with the same spectrum length.
_DtD_cache: dict = {}


def _get_cached_DtD(n: int, diff_order: int) -> sparse.csc_matrix:
    key = (n, diff_order)
    if key not in _DtD_cache:
        D = _whittaker_diff_matrix(n, diff_order)
        _DtD_cache[key] = (D.T @ D).tocsc()
    return _DtD_cache[key]


@dataclass(frozen=True)
class BaselineResult:
    """Container for baseline subtraction results."""
    y_corrected: np.ndarray
    baseline: np.ndarray


class BaselineAPI:
    """
    Centralised baseline subtraction API.

    Contract:
    - Caller is responsible for any smoothing BEFORE calling subtract()
      (keeps your current pre-processing order).
    - subtract() ALWAYS applies non-negativity clipping (unphysical negatives removed).
    """

    @staticmethod
    def parse_spec(baseline_method, poly_degree=None, gaussian_sigma=None):
        """
        Normalise baseline specifications to the runtime contract used by subtract().

        Canonical v0.3.8 user-facing form:
            {"method": "poly", "poly_order": 3}
            {"method": "gaussian", "gaussian_sigma": 10}
            {"method": "airpls", "lam": 1e6, "niter": 50, "tol": 1e-6}
        """
        if isinstance(baseline_method, (tuple, list)):
            warnings.warn(
                "Tuple/list baseline_method is deprecated. "
                "Use a dict specification instead, e.g. "
                "{'method': 'poly', 'poly_order': 3}.",
                category=FutureWarning,
                stacklevel=2,
            )

        if poly_degree is not None:
            warnings.warn(
                "poly_degree is deprecated. "
                "Use baseline_method={'method': 'poly', 'poly_order': degree} instead.",
                category=FutureWarning,
                stacklevel=2,
            )

        spec = normalise_baseline_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma,
        )
        return baseline_spec_to_runtime(spec)



    @staticmethod
    def subtract(
        x: np.ndarray,
        y: np.ndarray,
        method: str = "poly",
        *,
        clip_nonnegative: bool = True,
        mask: Optional[np.ndarray] = None,
        return_baseline: bool = True,
        **kwargs: Any,
    ) -> BaselineResult:
        """
        Compute baseline and subtract it from y.

        Parameters
        ----------
        x, y : np.ndarray
            1D arrays of the same length.
        method : str
            'poly' or 'gaussian' (extendable).
        clip_nonnegative : bool
            If True, sets negative corrected intensities to 0. Default True.
        mask : Optional[np.ndarray]
            Optional boolean array selecting points used to estimate baseline.
            If provided, baseline model is fitted on x[mask], y[mask], and then
            evaluated on full x.
        return_baseline : bool
            Kept for compatibility/extension; BaselineResult always includes baseline.
        kwargs :
            Method-specific parameters.

        Returns
        -------
        BaselineResult
            y_corrected and baseline (both length N arrays).
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape. Got {x.shape} vs {y.shape}.")

        if mask is not None:
            mask = np.asarray(mask, dtype=bool).ravel()
            if mask.shape != y.shape:
                raise ValueError(f"mask must match y shape. Got {mask.shape} vs {y.shape}.")
            x_fit = x[mask]
            y_fit = y[mask]
        else:
            x_fit = x
            y_fit = y

        method = str(method).strip().lower()

        if method == "poly":
            baseline = BaselineAPI._baseline_poly(x, x_fit, y_fit, **kwargs)

        elif method == "gaussian":
            baseline = BaselineAPI._baseline_gaussian(y, **kwargs)

        elif method in {"asls"}:
            baseline = BaselineAPI._baseline_asls(y, mask=mask, **kwargs)

        elif method in {"arpls"}:
            baseline = BaselineAPI._baseline_arpls(y, mask=mask, **kwargs)

        elif method in {"airpls"}:
            baseline = BaselineAPI._baseline_airpls(y, mask=mask, **kwargs)

        else:
            raise ValueError(
                f"Baseline method '{method}' not recognised. "
                "Use 'poly', 'gaussian', 'asls', 'arpls', or 'airpls'."
            )


        y_corr = y - baseline

        # Rule (2): always clip (default True; callers should not disable unless you decide later)
        if clip_nonnegative:
            y_corr = np.clip(y_corr, a_min=0.0, a_max=None)

        return BaselineResult(y_corrected=y_corr, baseline=baseline)

    @staticmethod
    def _baseline_poly(
        x_full: np.ndarray,
        x_fit: np.ndarray,
        y_fit: np.ndarray,
        *,
        poly_degree: int = 3,
        **_: Any,
    ) -> np.ndarray:
        """
        Polynomial baseline using the same approach as your current code:
        Polynomial.fit(...).convert().coef then np.polyval(coeffs[::-1], x_full)
        """
        if x_fit.size < poly_degree + 1:
            # Not enough points to fit requested degree; degrade gracefully
            deg = max(0, min(poly_degree, x_fit.size - 1))
        else:
            deg = poly_degree

        coeffs = Polynomial.fit(x_fit, y_fit, deg).convert().coef
        baseline = np.polyval(coeffs[::-1], x_full)
        return baseline

    @staticmethod
    def _baseline_gaussian(
        y_full: np.ndarray,
        *,
        gaussian_sigma: float = 10.0,
        **_: Any,
    ) -> np.ndarray:
        """
        Gaussian smoothing baseline using the same operator as your current code:
        gaussian_filter1d(y, sigma=gaussian_sigma)
        """
        sigma = float(gaussian_sigma)
        if sigma <= 0:
            raise ValueError(f"gaussian_sigma must be > 0. Got {sigma}.")
        return gaussian_filter1d(y_full, sigma=sigma)

    @staticmethod
    def _baseline_asls(
        y_full: np.ndarray,
        *,
        lam: float = 1e6,
        p: float = 0.01,
        niter: int = 20,
        diff_order: int = 2,
        mask: Optional[np.ndarray] = None,
        **_: Any,
    ) -> np.ndarray:
        """
        asLS baseline (Eilers-style asymmetric least squares).

        Parameters
        ----------
        lam : float
            Smoothness penalty (higher => smoother baseline).
        p : float
            Asymmetry parameter in (0, 1). Small p biases baseline below peaks.
        niter : int
            Number of reweighting iterations.
        diff_order : int
            Difference order (usually 2).
        mask : Optional[np.ndarray]
            Boolean array. Points outside mask get weight 0 (excluded).

        Returns
        -------
        baseline : np.ndarray
        """
        y = np.asarray(y_full, dtype=float).ravel()
        n = y.size

        lam = float(lam)
        p = float(p)
        niter = int(niter)
        diff_order = int(diff_order)

        if n < 3:
            return y.copy()
        if lam <= 0:
            raise ValueError(f"lam must be > 0. Got {lam}.")
        if not (0 < p < 1):
            raise ValueError(f"p must be in (0, 1). Got {p}.")
        if niter <= 0:
            raise ValueError(f"niter must be >= 1. Got {niter}.")
        if diff_order < 1:
            raise ValueError(f"diff_order must be >= 1. Got {diff_order}.")

        # DtD is constant within a call (D depends only on n, diff_order); cache across calls.
        DtD = _get_cached_DtD(n, diff_order)
        lam_DtD = lam * DtD  # W and z change each iteration; this does not

        # Initial weights
        w = np.ones(n, dtype=float)
        if mask is not None:
            m = np.asarray(mask, dtype=bool).ravel()
            if m.shape != y.shape:
                raise ValueError(f"mask must match y shape. Got {m.shape} vs {y.shape}.")
            w *= m.astype(float)

        z = np.zeros(n, dtype=float)

        for _ in range(niter):
            W = sparse.diags(w, 0, shape=(n, n), format="csc")
            Z = W + lam_DtD
            z = spsolve(Z, w * y)

            # Asymmetric weights: penalise points above baseline more strongly
            w = p * (y > z) + (1 - p) * (y < z)
            if mask is not None:
                w *= m.astype(float)

        return np.asarray(z, dtype=float)

    @staticmethod
    def _baseline_arpls(
        y_full: np.ndarray,
        *,
        lam: float = 1e6,
        niter: int = 50,
        diff_order: int = 2,
        tol: float = 1e-6,
        mask: Optional[np.ndarray] = None,
        **_: Any,
    ) -> np.ndarray:
        """
        arPLS baseline (Baek-style adaptive reweighting).

        Parameters
        ----------
        lam : float
            Smoothness penalty.
        niter : int
            Max iterations.
        diff_order : int
            Difference order (usually 2).
        tol : float
            Convergence tolerance on relative weight change.
        mask : Optional[np.ndarray]
            Boolean array. Points outside mask get weight 0 (excluded).

        Returns
        -------
        baseline : np.ndarray
        """
        y = np.asarray(y_full, dtype=float).ravel()
        n = y.size

        lam = float(lam)
        niter = int(niter)
        diff_order = int(diff_order)
        tol = float(tol)

        if n < 3:
            return y.copy()
        if lam <= 0:
            raise ValueError(f"lam must be > 0. Got {lam}.")
        if niter <= 0:
            raise ValueError(f"niter must be >= 1. Got {niter}.")
        if diff_order < 1:
            raise ValueError(f"diff_order must be >= 1. Got {diff_order}.")
        if tol <= 0:
            raise ValueError(f"tol must be > 0. Got {tol}.")

        # DtD is constant within a call (D depends only on n, diff_order); cache across calls.
        DtD = _get_cached_DtD(n, diff_order)
        lam_DtD = lam * DtD

        # Initial weights
        w = np.ones(n, dtype=float)
        if mask is not None:
            m = np.asarray(mask, dtype=bool).ravel()
            if m.shape != y.shape:
                raise ValueError(f"mask must match y shape. Got {m.shape} vs {y.shape}.")
            w *= m.astype(float)

        z = np.zeros(n, dtype=float)
        eps = 1e-12

        for _ in range(niter):
            w_prev = w.copy()

            W = sparse.diags(w, 0, shape=(n, n), format="csc")
            Z = W + lam_DtD
            z = spsolve(Z, w * y)

            d = y - z
            dn = d[d < 0]
            if dn.size == 0:
                # No negative residuals => baseline at or above signal everywhere
                break

            mu = dn.mean()
            sigma = dn.std() + eps

            # Logistic reweighting (downweight positive residuals / peaks)
            w = 1.0 / (1.0 + np.exp(2.0 * (d - (2.0 * sigma - mu)) / sigma))

            if mask is not None:
                w *= m.astype(float)

            # Convergence check
            denom = np.linalg.norm(w_prev) + eps
            if np.linalg.norm(w - w_prev) / denom < tol:
                break

        return np.asarray(z, dtype=float)

    @staticmethod
    def _baseline_airpls(
        y_full: np.ndarray,
        *,
        lam: float = 1e6,
        niter: int = 50,
        diff_order: int = 2,
        tol: float = 1e-6,
        mask: Optional[np.ndarray] = None,
        min_weight: float = 1e-6,
        ridge: float = 1e-12,
        exp_clip: float = 50.0,
        **_: Any,
    ) -> np.ndarray:
        """
        Stabilised airPLS baseline (adaptive iteratively reweighted penalised least squares).

        Key stabilisations:
          - Enforce a minimum weight floor (min_weight) to prevent W -> 0 (singular system).
          - Add a tiny ridge term (ridge * I) for numerical robustness.
          - Clip exponent (exp_clip) to avoid overflow in weight updates.

        Parameters
        ----------
        lam : float
            Smoothness penalty.
        niter : int
            Max iterations.
        diff_order : int
            Difference order (usually 2).
        tol : float
            Stop if sum of negative residual magnitudes < tol * sum(|y|).
        mask : Optional[np.ndarray]
            Boolean array. Points outside mask are excluded (weight 0).
        min_weight : float
            Minimum weight in included region to keep system well-posed.
        ridge : float
            Tiny diagonal regularisation to guarantee solvability.
        exp_clip : float
            Clip value for exponent argument to avoid overflow.

        Returns
        -------
        baseline : np.ndarray
        """
        y = np.asarray(y_full, dtype=float).ravel()
        n = y.size

        lam = float(lam)
        niter = int(niter)
        diff_order = int(diff_order)
        tol = float(tol)
        min_weight = float(min_weight)
        ridge = float(ridge)
        exp_clip = float(exp_clip)

        if n < 3:
            return y.copy()
        if lam <= 0:
            raise ValueError(f"lam must be > 0. Got {lam}.")
        if niter <= 0:
            raise ValueError(f"niter must be >= 1. Got {niter}.")
        if diff_order < 1:
            raise ValueError(f"diff_order must be >= 1. Got {diff_order}.")
        if tol <= 0:
            raise ValueError(f"tol must be > 0. Got {tol}.")
        if min_weight <= 0:
            raise ValueError(f"min_weight must be > 0. Got {min_weight}.")
        if ridge < 0:
            raise ValueError(f"ridge must be >= 0. Got {ridge}.")

        DtD = _get_cached_DtD(n, diff_order)
        # lam and ridge are call-level constants; pre-compute to avoid per-iteration sparse ops.
        lam_DtD_ridge_I = lam * DtD + ridge * sparse.eye(n, format="csc")

        # Mask handling
        if mask is not None:
            m = np.asarray(mask, dtype=bool).ravel()
            if m.shape != y.shape:
                raise ValueError(f"mask must match y shape. Got {m.shape} vs {y.shape}.")
        else:
            m = None

        # Initial weights: floor in included region, zero outside mask
        w = np.full(n, min_weight, dtype=float)
        if m is not None:
            w[~m] = 0.0

        z = np.zeros(n, dtype=float)
        eps = 1e-12
        y_scale = np.sum(np.abs(y)) + eps

        for i in range(1, niter + 1):
            # Solve (W + lam*DtD + ridge*I) z = W y
            W = sparse.diags(w, 0, shape=(n, n), format="csc")
            Z = W + lam_DtD_ridge_I
            z = spsolve(Z, w * y)

            d = y - z
            neg_idx = d < 0
            if m is not None:
                neg_idx &= m  # only consider included region

            neg_sum = np.sum(np.abs(d[neg_idx])) + eps

            # Convergence: baseline sufficiently below the signal in aggregate
            if neg_sum < tol * y_scale:
                break

            # Weight update: keep floor everywhere (included region), boost negative residuals
            w_new = np.full(n, min_weight, dtype=float)
            if m is not None:
                w_new[~m] = 0.0

            if np.any(neg_idx):
                arg = i * np.abs(d[neg_idx]) / neg_sum
                arg = np.clip(arg, 0.0, exp_clip)
                w_new[neg_idx] = np.exp(arg)

            # Endpoint stabilisation (within included region)
            if m is None or m[0]:
                w_new[0] = max(w_new[0], min_weight)
            if m is None or m[-1]:
                w_new[-1] = max(w_new[-1], min_weight)

            w = w_new

        return np.asarray(z, dtype=float)
