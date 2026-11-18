
import os
import sys
import csv
import math
import time
from collections import Counter

import numpy as np
from tqdm import tqdm
from scipy.stats import qmc

import clr

DWSIM_PATH = r"C:\Users\pc\AppData\Local\DWSIM"
sys.path.append(DWSIM_PATH)
if os.path.isdir(DWSIM_PATH):
    os.chdir(DWSIM_PATH)

for dll in ["CapeOpen.dll", "DWSIM.Automation.dll", "DWSIM.Interfaces.dll",
            "DWSIM.GlobalSettings.dll", "DWSIM.SharedClasses.dll",
            "DWSIM.Thermodynamics.dll", "DWSIM.UnitOperations.dll",
            "DWSIM.Inspector.dll", "System.Buffers.dll"]:
    clr.AddReference(os.path.join(DWSIM_PATH, dll))

from DWSIM.Automation import Automation3


FLOWSHEET = r"C:\Users\pc\Desktop\project\simulation chaine porduciton de ciment\Gaz de Pétrole Liquéfié_original.dwxmz"
OUTPUT_CSV = r"C:\Users\pc\Desktop\project\simulation chaine porduciton de ciment\data_final.csv"

TAGS = {
    "feed1": "feed1",
    "feed2": "feed2",
    "heater": "HT-1",
    "column": "T101",
    "col_feed": "16",
    "top": "17",
    "bot": "18",          # mets "18a" si le fond s'appelle 18a dans ton fichier
    "q_cond": "E5",
    "q_reb": "E6",
    "q_heater": "E4",
}

COMPONENTS = ["Ethane", "Propane", "Isobutane", "N-butane",
              "Isopentane", "N-pentane", "N-hexane"]

FEED1_SHARE = 0.6        
DP_COLUMN_BAR = 1.4      

BASE = {
    "reflux_ratio": 2.6,
    "feed_total_kg_h": 185200.0,
    "dT_HT1_K": 20.0,
    "P_cond_barg": 19.3,
}
FEATURES = list(BASE.keys())

BOUNDS = {
    "reflux_ratio": (2.0, 3.6),
    "feed_total_kg_h": (166_680.0, 203_720.0),   # +/- 10 %
    "dT_HT1_K": (10.0, 26.0),
    "P_cond_barg": (18.6, 19.4),                 # P_reb reste <= ~20.8 barg
}

N_SAMPLES = 1500        
SEED = 42
RESUME = True

EXPECTED_BASE = {"Q_cond_kW": 20615.73, "Q_reb_kW": 10725.69}  # pour le warning


def barg_to_pa(p_barg):
    return (p_barg + 1.01325) * 1e5


def to_float(v):
    return float(v) if v is not None else math.nan


def get_object(sim, tag):
    obj = sim.GetFlowsheetSimulationObject(tag)
    if obj is None:
        raise RuntimeError(f"Objet '{tag}' introuvable dans le flowsheet")
    try:
        obj = obj.GetAsObject()
    except Exception:
        pass
    return obj


def list_tags(sim):
    tags = []
    try:
        for obj in sim.SimulationObjects.Values:
            try:
                tags.append(str(obj.GraphicObject.Tag))
            except Exception:
                pass
    except Exception:
        pass
    return sorted(tags)


def set_inputs(sim, x):
    col = get_object(sim, TAGS["column"])
    col.Specs["C"].SpecValue = float(x["reflux_ratio"])
    col.CondenserPressure = barg_to_pa(x["P_cond_barg"])
    col.ReboilerPressure = barg_to_pa(x["P_cond_barg"] + DP_COLUMN_BAR)

    total_kg_s = float(x["feed_total_kg_h"]) / 3600.0
    get_object(sim, TAGS["feed1"]).Phases[0].Properties.massflow = FEED1_SHARE * total_kg_s
    get_object(sim, TAGS["feed2"]).Phases[0].Properties.massflow = (1.0 - FEED1_SHARE) * total_kg_s

    get_object(sim, TAGS["heater"]).DeltaT = float(x["dT_HT1_K"])


def solve(interf, sim):
 
    errors = interf.CalculateFlowsheet2(sim)
    if errors is None:
        return []
    return [str(e)[:120] for e in errors]


def read_composition(sim, tag, prefix):
    comps = get_object(sim, tag).GetPhase("Overall").Compounds
    return {f"{prefix}_{c}": to_float(comps[c].MoleFraction) for c in COMPONENTS}


def read_T_C(sim, tag):
    s = get_object(sim, tag)
    try:
        t_k = s.GetTemperature()
    except Exception:
        t_k = s.Phases[0].Properties.temperature
    return to_float(t_k) - 273.15


def read_massflow_kg_h(sim, tag):
    return to_float(get_object(sim, tag).Phases[0].Properties.massflow) * 3600.0


def read_duty(sim, tag):
    try:
        return to_float(get_object(sim, tag).EnergyFlow)
    except Exception:
        return math.nan


def read_outputs(sim):
    out = {}
    out.update(read_composition(sim, TAGS["top"], "TOP"))
    out.update(read_composition(sim, TAGS["bot"], "BOT"))
    out["TOP_T_C"] = read_T_C(sim, TAGS["top"])
    out["BOT_T_C"] = read_T_C(sim, TAGS["bot"])
    out["T16_C"] = read_T_C(sim, TAGS["col_feed"])

    f16 = read_massflow_kg_h(sim, TAGS["col_feed"])
    f17 = read_massflow_kg_h(sim, TAGS["top"])
    f18 = read_massflow_kg_h(sim, TAGS["bot"])
    out["F16_kg_h"] = f16
    out["mass_balance_err"] = (f16 - f17 - f18) / f16 if f16 else math.nan

    out["Q_cond_kW"] = read_duty(sim, TAGS["q_cond"])
    out["Q_reb_kW"] = read_duty(sim, TAGS["q_reb"])
    out["Q_HT1_kW"] = read_duty(sim, TAGS["q_heater"])
    return out


def check_output(out):
 
    for k in ("Q_cond_kW", "Q_reb_kW", "F16_kg_h", "T16_C",
              "TOP_Propane", "BOT_Propane"):
        if math.isnan(out.get(k, math.nan)):
            return f"nan_{k}"
    if abs(out["mass_balance_err"]) > 1e-3:
        return "mass_balance"
    for prefix in ("TOP", "BOT"):
        s = sum(out[f"{prefix}_{c}"] for c in COMPONENTS)
        if abs(s - 1.0) > 1e-3:
            return f"sum_{prefix}"
    return ""


def attempt(interf, sim, x):

    try:
        set_inputs(sim, x)
        errs = solve(interf, sim)
        if errs:
            return "calc_error: " + errs[0], {}
        col = get_object(sim, TAGS["column"])
        try:
            if not bool(col.Calculated):
                return "column_not_calculated", {}
        except Exception:
            pass
        out = read_outputs(sim)
        reason = check_output(out)
        return ("ok" if not reason else reason), out
    except Exception as e:
        return f"exception: {str(e)[:100]}", {}


def solve_point(interf, sim, x):

    status, out = attempt(interf, sim, x)
    if status == "ok":
        return status, out
    for frac in (0.0, 0.5, 1.0):
        xi = {k: BASE[k] + frac * (x[k] - BASE[k]) for k in FEATURES}
        status, out = attempt(interf, sim, xi)
    return status, out


def restore_base(interf, sim):
    attempt(interf, sim, BASE)


PROBES = [
    ("reflux_ratio", +0.30, "Q_cond_kW"),
    ("feed_total_kg_h", +9000.0, "F16_kg_h"),
    ("dT_HT1_K", +5.0, "T16_C"),
    ("P_cond_barg", +0.30, "TOP_T_C"),
]


def preflight(interf, sim):
    print("=== PREFLIGHT ===")
    tags = list_tags(sim)
    if tags:
        print("Objets du flowsheet:", ", ".join(tags))
        missing = [t for t in TAGS.values() if t not in tags]
        if missing:
            raise SystemExit(f"Tags introuvables: {missing}. Corrige le dictionnaire TAGS.")

    col = get_object(sim, TAGS["column"])
    print("Colonne:", col.GetType().FullName)
    for key in ("C", "R"):
        try:
            sp = col.Specs[key]
            print(f"Spec '{key}': type={sp.SType} valeur={sp.SpecValue}")
        except Exception:
            pass
    try:
        print("CalcMode HT-1:", get_object(sim, TAGS["heater"]).CalcMode)
    except Exception:
        pass

    status, base_out = attempt(interf, sim, BASE)
    print("Cas de base:", status)
    if status != "ok":
        raise SystemExit("Le cas de base ne converge pas : verifie le flowsheet.")
    for k, ref in EXPECTED_BASE.items():
        v = base_out[k]
        flag = "OK" if abs(v - ref) / ref < 0.02 else "ECART > 2 % (verifie le fichier)"
        print(f"  {k} = {v:.1f} (attendu ~ {ref:.1f}) {flag}")

    bad = []
    for name, delta, key in PROBES:
        x = dict(BASE)
        x[name] += delta
        st, out = attempt(interf, sim, x)
        ref = base_out[key]
        changed = st == "ok" and abs(out[key] - ref) > 1e-6 * max(1.0, abs(ref))
        print(f"  {name:16s} {delta:+9.2f} -> {key}: {ref:.4f} -> "
              f"{out.get(key, math.nan):.4f}  {'OK' if changed else 'AUCUN EFFET / ECHEC (' + st + ')'}")
        if not changed:
            bad.append(name)
        restore_base(interf, sim)
    if bad:
        raise SystemExit(f"Inputs sans effet ou en echec: {bad}. Ne lance pas le scan avant correction.")
    print("=== PREFLIGHT OK ===\n")



def make_samples():
    lows = [BOUNDS[k][0] for k in FEATURES]
    highs = [BOUNDS[k][1] for k in FEATURES]
    unit = qmc.LatinHypercube(d=len(FEATURES), seed=SEED).random(N_SAMPLES)
    arr = qmc.scale(unit, lows, highs)
    return [dict(zip(FEATURES, row)) for row in arr]


def build_header():
    header = FEATURES + ["P_reb_barg", "F16_kg_h", "T16_C", "converged", "status",
                         "mass_balance_err", "Q_cond_kW", "Q_reb_kW", "Q_HT1_kW"]
    header += [f"TOP_{c}" for c in COMPONENTS]
    header += [f"BOT_{c}" for c in COMPONENTS]
    header += ["TOP_T_C", "BOT_T_C"]
    return header


def count_done(path, header):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip().split(",")
        if first != header:
            raise SystemExit(f"{path} existe avec un autre en-tete. Renomme-le ou supprime-le.")
        return sum(1 for _ in f)


def key_vector(out):
    return (out["Q_cond_kW"], out["Q_reb_kW"], out["TOP_Propane"], out["BOT_Propane"])


def run_scan(interf, sim):
    header = build_header()
    samples = make_samples()
    done = count_done(OUTPUT_CSV, header) if RESUME else 0
    mode = "a" if done else "w"
    if done:
        print(f"Reprise: {done} lignes deja ecrites")
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

    stats = Counter()
    prev_key = None
    t0 = time.time()

    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if mode == "w":
            writer.writeheader()

        for i in tqdm(range(done, N_SAMPLES), desc="Scan LHS", unit="sim", colour="cyan"):
            x = samples[i]
            row = {k: math.nan for k in header}
            row.update(x)
            row["P_reb_barg"] = x["P_cond_barg"] + DP_COLUMN_BAR

            status, out = solve_point(interf, sim, x)

            # resultats "figes" : identiques a la ligne precedente alors que x a change
            if status == "ok":
                kv = key_vector(out)
                if prev_key is not None and all(abs(a - b) < 1e-9 for a, b in zip(kv, prev_key)):
                    status = "stale_results"
                prev_key = kv

            row["status"] = status
            row["converged"] = int(status == "ok")
            if status == "ok":
                row.update(out)
            stats[status.split(":")[0]] += 1
            if status != "ok":
                restore_base(interf, sim)

            writer.writerow(row)
            f.flush()

    n = sum(stats.values())
    print(f"\nTermine: {n} points en {time.time() - t0:.0f} s")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    interf = Automation3()
    sim = interf.LoadFlowsheet(FLOWSHEET)
    preflight(interf, sim)
    run_scan(interf, sim)