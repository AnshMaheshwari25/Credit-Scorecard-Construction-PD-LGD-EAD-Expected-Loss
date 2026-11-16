"""Expected loss: EL = PD x LGD x EAD, aggregation, sensitivity and a (proxy) back-test."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_expected_loss(df: pd.DataFrame, pd_hat, lgd_hat, ccf_hat) -> pd.DataFrame:
    """Loan-level EL. ``EAD = CCF x funded_amnt`` (exposure at default of a term loan: principal not yet repaid)."""
    out = pd.DataFrame(index=df.index)
    out["funded_amnt"] = df["funded_amnt"].to_numpy(dtype=float)
    out["pd"] = np.asarray(pd_hat, dtype=float)
    out["lgd"] = np.asarray(lgd_hat, dtype=float)
    out["ccf"] = np.asarray(ccf_hat, dtype=float)
    out["ead"] = out["ccf"] * out["funded_amnt"]
    out["el"] = out["pd"] * out["lgd"] * out["ead"]
    return out


def summarize_el(el: pd.DataFrame, by: pd.Series | None = None, label: str = "segment") -> pd.DataFrame:
    """Portfolio / segment summary. ``el_rate_%`` = EL as a percentage of funded amount."""
    g = el.groupby(by.reindex(el.index)) if by is not None else el.groupby(lambda _: "Portfolio")
    t = g.agg(loans=("el", "size"), funded=("funded_amnt", "sum"), avg_pd=("pd", "mean"), avg_lgd=("lgd", "mean"),
              avg_ccf=("ccf", "mean"), ead=("ead", "sum"), expected_loss=("el", "sum"))
    t["el_rate_%"] = 100 * t["expected_loss"] / t["funded"]
    t.index.name = label
    return t.reset_index()


def realized_loss_proxy(df: pd.DataFrame, lgd_for_uncharged: float) -> pd.Series:
    """Observed loss proxy per loan: 0 for non-bad loans; for bad loans EAD_actual x LGD_actual.

    LGD is the realised one for charged-off loans; 'Late (31-120)' and 'Default' loans have no completed workout, so the
    mean charged-off LGD of the training sample is assumed (``lgd_for_uncharged``). This is a *proxy*, not accounting loss.
    """
    lgd = df["lgd"].where(df["is_charged_off"] & df["lgd"].notna(), lgd_for_uncharged)
    loss = df["ead_at_default"] * lgd
    return loss.where(df["bad"] == 1, 0.0).rename("loss_proxy")


def sensitivity_grid(el: pd.DataFrame, pd_multipliers, lgd_adds) -> pd.DataFrame:
    """Portfolio EL (% of funded) under a PD multiplier (rows) and an additive LGD shock in pp (columns)."""
    funded = el["funded_amnt"].sum()
    rows = {}
    for m in pd_multipliers:
        pdm = np.clip(el["pd"] * m, 0, 1)
        rows[f"PD x{m:g}"] = {f"LGD +{int(round(100 * a))}pp": 100 * float(
            (pdm * np.clip(el["lgd"] + a, 0, 1) * el["ead"]).sum() / funded) for a in lgd_adds}
    return pd.DataFrame(rows).T
