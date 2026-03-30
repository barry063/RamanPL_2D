from collections import Counter
import numpy as np


def fit_summary(
    obj,
    *,
    print_summary: bool = True,
    rmse_quantiles=(0.5, 0.9, 0.95, 0.99),
):
    """
    Summarise mapping fit quality and bound-sticking diagnostics.

    Works for both RamanMapping and PLMapping provided they define:
      - obj.X, obj.Y
      - obj.residual_map (RMSE in fit-space; NaN = failed/unfitted)
      - obj.fit_diagnostics_map: [Y, X] of dicts with keys:
          ok: bool
          reason: str (optional, for failures)
          n_params_at_lower_bounds / n_params_at_upper_bounds (optional)
    """
    if not hasattr(obj, "residual_map"):
        raise AttributeError("Object has no residual_map. Run fit_spectra() first.")
    if not hasattr(obj, "fit_diagnostics_map"):
        raise AttributeError("Object has no fit_diagnostics_map. Run fit_spectra() first.")

    Y, X = int(obj.Y), int(obj.X)
    total = X * Y

    residual = np.asarray(obj.residual_map, dtype=float)
    ok_mask = np.isfinite(residual)

    n_ok = int(np.count_nonzero(ok_mask))
    n_fail = total - n_ok

    # Failure reasons (best-effort)
    reasons = Counter()
    diag = obj.fit_diagnostics_map
    for jj in range(Y):
        for ii in range(X):
            d = diag[jj, ii]
            if not isinstance(d, dict):
                continue
            if d.get("ok") is False:
                reasons[str(d.get("reason", "fit_failed"))] += 1

    # Bound sticking stats (best-effort)
    lower_hits = []
    upper_hits = []
    for jj in range(Y):
        for ii in range(X):
            d = diag[jj, ii]
            if not isinstance(d, dict):
                continue
            if d.get("ok") is True:
                if "n_params_at_lower_bounds" in d:
                    lower_hits.append(int(d["n_params_at_lower_bounds"]))
                if "n_params_at_upper_bounds" in d:
                    upper_hits.append(int(d["n_params_at_upper_bounds"]))

    lower_hits = np.asarray(lower_hits, dtype=float) if lower_hits else None
    upper_hits = np.asarray(upper_hits, dtype=float) if upper_hits else None

    # RMSE stats
    rmse_vals = residual[ok_mask]
    rmse_stats = {}
    if rmse_vals.size:
        rmse_stats["mean"] = float(np.mean(rmse_vals))
        rmse_stats["median"] = float(np.median(rmse_vals))
        for q in rmse_quantiles:
            rmse_stats[f"q{int(round(q*100))}"] = float(np.quantile(rmse_vals, q))

    out = dict(
        n_total=total,
        n_ok=n_ok,
        n_fail=n_fail,
        success_rate=(n_ok / total) if total else np.nan,
        rmse_stats=rmse_stats,
        failure_reasons=dict(reasons),
        bounds=dict(
            lower=dict(
                mean=float(np.mean(lower_hits)) if lower_hits is not None and lower_hits.size else None,
                max=int(np.max(lower_hits)) if lower_hits is not None and lower_hits.size else None,
            ),
            upper=dict(
                mean=float(np.mean(upper_hits)) if upper_hits is not None and upper_hits.size else None,
                max=int(np.max(upper_hits)) if upper_hits is not None and upper_hits.size else None,
            ),
        ),
    )

    if print_summary:
        print("\n=== Fit summary ===")
        print(f"Successful fits: {n_ok} / {total} ({100*out['success_rate']:.1f}%)")

        if rmse_stats:
            q_bits = " | ".join([f"{k}: {v:.4g}" for k, v in rmse_stats.items()])
            print(f"RMSE (fit-space): {q_bits}")

        if reasons:
            print("\nFailure reasons:")
            for k, v in reasons.most_common():
                print(f"  - {k}: {v}")

        lb_mean = out["bounds"]["lower"]["mean"]
        ub_mean = out["bounds"]["upper"]["mean"]
        if lb_mean is not None or ub_mean is not None:
            print("\nBound-sticking (params at bounds per successful pixel):")
            if lb_mean is not None:
                print(f"  - lower: mean {lb_mean:.3g}, max {out['bounds']['lower']['max']}")
            if ub_mean is not None:
                print(f"  - upper: mean {ub_mean:.3g}, max {out['bounds']['upper']['max']}")

    return out