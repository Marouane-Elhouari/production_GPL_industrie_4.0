import os
import sys
import csv
import math
import time
import subprocess
from collections import Counter

import numpy as np
from tqdm import tqdm
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.abspath(__file__)

DWSIM_PATH = r"C:\Users\pc\AppData\Local\DWSIM"
DLLS = ["CapeOpen.dll", "DWSIM.Automation.dll", "DWSIM.Interfaces.dll",
        "DWSIM.GlobalSettings.dll", "DWSIM.SharedClasses.dll",
        "DWSIM.Thermodynamics.dll", "DWSIM.UnitOperations.dll",
        "DWSIM.Inspector.dll", "System.Buffers.dll"]

Automation3 = None
System = None

FLOWSHEET = r"C:\Users\pc\Desktop\project\simulation chaine porduciton de ciment\Gaz de Pétrole Liquéfié_original.dwxmz"
OUTPUT_CSV = r"C:\Users\pc\Desktop\project\simulation chaine porduciton de ciment\data_final_v4.csv"
CLEAN_CSV = OUTPUT_CSV.replace(".csv", "_clean.csv")
SKIP_FILE = os.path.join(os.path.dirname(OUTPUT_CSV), "skip_points_v4.txt")
BEAT_FILE = os.path.join(os.path.dirname(OUTPUT_CSV), "heartbeat_v4.txt")

TIMEOUT = 240
POLL = 5
RESTART_DELAY = 3
MAX_FAILS = 2
MAX_START_FAILS = 3

TAGS = {
    "feed1": "feed1",
    "feed2": "feed2",
    "heater": "HT-1",
    "column": "T101",
    "col_feed": "16",
    "top": "17",
    "bot": "18",
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
    "duty_HT1": 7529.75,
    "P_cond_barg": 19.3,
}
FEATURES = list(BASE.keys())

BOUNDS = {
    "reflux_ratio": (2.0, 3.6),
    "feed_total_kg_h": (166_680.0, 203_720.0),
    "duty_HT1": (7000.0, 8000.0),
    "P_cond_barg": (18.6, 19.4),
}

N_SAMPLES = 1500
SEED = 42
RESUME = True

EXPECTED_BASE = {"Q_cond_kW": 20615.73, "Q_reb_kW": 10725.69}


def fatal(msg):
    print(msg)
    sys.exit(2)


def load_dwsim():
    global Automation3, System
    sys.path.append(DWSIM_PATH)
    if os.path.isdir(DWSIM_PATH):
        os.chdir(DWSIM_PATH)
    import clr
    import System as _System
    System = _System
    for dll in DLLS:
        clr.AddReference(os.path.join(DWSIM_PATH, dll))
    from DWSIM.Automation import Automation3 as _Automation3
    Automation3 = _Automation3


def touch_beat(idx):
    try:
        with open(BEAT_FILE, "w") as fh:
            fh.write(str(idx))
    except OSError:
        pass


def load_skips():
    if not os.path.exists(SKIP_FILE):
        return set()
    with open(SKIP_FILE) as fh:
        return {int(l) for l in fh if l.strip().lstrip("-").isdigit()}


def barg_to_pa(p_barg):
    return (p_barg + 1.01325) * 1e5


def pa_to_barg(p_pa):
    return to_float(p_pa) / 1e5 - 1.01325


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
                tags.append(str(obj.GraphicObject.Tag).strip())
            except Exception:
                pass
    except Exception:
        pass
    return sorted(tags)


def read_column_pressures(col):
    stages = list(col.Stages)
    return pa_to_barg(stages[0].P), pa_to_barg(stages[-1].P)


def set_column_pressure(col, p_cond_barg):
    pc = barg_to_pa(p_cond_barg)
    pr = barg_to_pa(p_cond_barg + DP_COLUMN_BAR)
    stages = list(col.Stages)
    n = len(stages)
    for i, st in enumerate(stages):
        st.P = pc + (pr - pc) * i / (n - 1)


def set_heater_duty(htr, duty_kw):
    enum_type = htr.CalcMode.GetType()
    try:
        htr.CalcMode = System.Enum.Parse(enum_type, "HeatAdded")
    except Exception:
        names = list(System.Enum.GetNames(enum_type))
        raise RuntimeError(f"Mode 'HeatAdded' introuvable. Modes disponibles: {names}")
    htr.DeltaQ = float(duty_kw)


def set_inputs(sim, x):
    col = get_object(sim, TAGS["column"])
    col.Specs["C"].SpecValue = float(x["reflux_ratio"])
    set_column_pressure(col, x["P_cond_barg"])

    total_kg_s = float(x["feed_total_kg_h"]) / 3600.0
    get_object(sim, TAGS["feed1"]).Phases[0].Properties.massflow = FEED1_SHARE * total_kg_s
    get_object(sim, TAGS["feed2"]).Phases[0].Properties.massflow = (1.0 - FEED1_SHARE) * total_kg_s

    set_heater_duty(get_object(sim, TAGS["heater"]), x["duty_HT1"])


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


def read_P_barg(sim, tag):
    return pa_to_barg(get_object(sim, tag).Phases[0].Properties.pressure)


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
    out["P_top_barg"] = read_P_barg(sim, TAGS["top"])

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
    for k in ("Q_cond_kW", "Q_reb_kW", "Q_HT1_kW", "F16_kg_h", "T16_C",
              "P_top_barg", "TOP_Propane", "BOT_Propane"):
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
        if status != "ok":
            break
    return status, out


def restore_base(interf, sim):
    attempt(interf, sim, BASE)


PROBES = [
    ("reflux_ratio", +0.30, "Q_cond_kW"),
    ("feed_total_kg_h", +9000.0, "F16_kg_h"),
    ("duty_HT1", +300.0, "T16_C"),
    ("P_cond_barg", +0.30, "P_top_barg"),
]


def preflight(interf, sim):
    global DP_COLUMN_BAR
    print("=== PREFLIGHT ===")

    tags = list_tags(sim)
    print("Objets du flowsheet:", ", ".join(tags))
    missing = [t for t in TAGS.values() if t not in tags]
    if missing:
        fatal(f"Tags introuvables: {missing}. Corrige le dictionnaire TAGS.")

    col = get_object(sim, TAGS["column"])
    print("Colonne:", col.GetType().FullName)
    for key in ("C", "R"):
        try:
            sp = col.Specs[key]
            print(f"Spec '{key}': type={sp.SType} valeur={sp.SpecValue}")
        except Exception:
            pass

    errs = solve(interf, sim)
    if errs:
        print("  [warn] calcul initial:", errs[0])

    try:
        pc, pr = read_column_pressures(col)
        if not (math.isnan(pc) or math.isnan(pr)) and pr > pc:
            DP_COLUMN_BAR = round(pr - pc, 3)
            BASE["P_cond_barg"] = round(pc, 3)
        else:
            print(f"  [warn] pressions illisibles (pc={pc}, pr={pr}) ; DP par defaut = {DP_COLUMN_BAR} bar")
    except Exception as e:
        print(f"  [warn] lecture pressions impossible ({str(e)[:80]}) ; DP par defaut = {DP_COLUMN_BAR} bar")
    print(f"  P_cond (etage 0) = {BASE['P_cond_barg']:.3f} barg | DP = {DP_COLUMN_BAR:.3f} bar "
          f"| P_reb = {BASE['P_cond_barg'] + DP_COLUMN_BAR:.3f} barg")
    print(f"  P flux 17 = {read_P_barg(sim, TAGS['top']):.3f} barg | P flux 18 = {read_P_barg(sim, TAGS['bot']):.3f} barg")

    htr = get_object(sim, TAGS["heater"])
    try:
        print("CalcMode HT-1 (fichier original):", htr.CalcMode)
    except Exception:
        pass

    q_orig = read_duty(sim, TAGS["q_heater"])
    if not math.isnan(q_orig):
        BASE["duty_HT1"] = round(q_orig, 2)
    print(f"  Duty HT-1 original = {q_orig:.2f} kW ; T16 original = {read_T_C(sim, TAGS['col_feed']):.2f} C")

    for name in ("duty_HT1", "P_cond_barg"):
        lo, hi = BOUNDS[name]
        if not (lo <= BASE[name] <= hi):
            print(f"  [warn] BASE[{name}] = {BASE[name]} hors bornes ({lo}, {hi}) : ajuste BOUNDS.")

    status, base_out = attempt(interf, sim, BASE)
    print("Cas de base:", status)
    if status != "ok":
        fatal("Le cas de base ne converge pas : verifie le flowsheet.")
    for k, ref in EXPECTED_BASE.items():
        v = base_out[k]
        flag = "OK" if abs(v - ref) / ref < 0.02 else "ECART > 2 % (verifie le fichier)"
        print(f"  {k} = {v:.1f} (attendu ~ {ref:.1f}) {flag}")
    print(f"  T16 base = {base_out['T16_C']:.2f} C | Q_HT1 base = {base_out['Q_HT1_kW']:.1f} kW")

    bad = []
    for name, delta, key in PROBES:
        x = dict(BASE)
        x[name] += delta
        st, out = attempt(interf, sim, x)
        ref = base_out[key]
        new = out.get(key, math.nan)
        changed = st == "ok" and abs(new - ref) > 1e-6 * max(1.0, abs(ref))
        print(f"  {name:16s} {delta:+9.2f} -> {key}: {ref:.4f} -> {new:.4f}  "
              f"{'OK' if changed else 'AUCUN EFFET / ECHEC (' + st + ')'}")
        if not changed:
            bad.append(name)
        restore_base(interf, sim)
    if bad:
        fatal(f"Inputs sans effet ou en echec: {bad}. Ne lance pas le scan avant correction.")
    print("=== PREFLIGHT OK ===\n")


def make_samples():
    lows = [BOUNDS[k][0] for k in FEATURES]
    highs = [BOUNDS[k][1] for k in FEATURES]
    unit = qmc.LatinHypercube(d=len(FEATURES), seed=SEED).random(N_SAMPLES)
    arr = qmc.scale(unit, lows, highs)
    return [dict(zip(FEATURES, row)) for row in arr]


def build_header():
    header = FEATURES + ["P_reb_barg", "P_top_barg", "F16_kg_h", "T16_C",
                         "converged", "status", "mass_balance_err",
                         "Q_cond_kW", "Q_reb_kW", "Q_HT1_kW"]
    header += [f"TOP_{c}" for c in COMPONENTS]
    header += [f"BOT_{c}" for c in COMPONENTS]
    header += ["TOP_T_C", "BOT_T_C"]
    return header


def count_done(path, header):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first != header:
            fatal(f"{path} existe avec un autre en-tete. Renomme-le ou supprime-le.")
        return sum(1 for _ in reader)


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

    skips = load_skips()
    stats = Counter()
    prev_key = None
    t0 = time.time()

    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        for i in tqdm(range(done, N_SAMPLES), desc="Scan LHS", unit="sim", colour="cyan"):
            touch_beat(i)
            x = samples[i]
            row = {k: math.nan for k in header}
            row.update(x)
            row["P_reb_barg"] = x["P_cond_barg"] + DP_COLUMN_BAR

            if i in skips:
                row["status"] = "skipped"
                row["converged"] = 0
                stats["skipped"] += 1
                writer.writerow(row)
                f.flush()
                continue

            status, out = solve_point(interf, sim, x)

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


def worker():
    touch_beat(-1)
    load_dwsim()
    interf = Automation3()
    sim = interf.LoadFlowsheet(FLOWSHEET)
    preflight(interf, sim)
    run_scan(interf, sim)


def read_beat():
    try:
        with open(BEAT_FILE) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def supervise(cmd=None):
    cmd = cmd or [sys.executable, SELF, "--worker"]
    fails = Counter()
    start_fails = 0

    while True:
        if os.path.exists(BEAT_FILE):
            os.remove(BEAT_FILE)
        proc = subprocess.Popen(cmd, cwd=HERE)
        last_beat = time.time()
        last_mtime = 0
        hung = False

        while proc.poll() is None:
            time.sleep(POLL)
            m = os.stat(BEAT_FILE).st_mtime_ns if os.path.exists(BEAT_FILE) else 0
            if m != last_mtime:
                last_mtime, last_beat = m, time.time()
            elif time.time() - last_beat > TIMEOUT:
                hung = True
                proc.kill()
                proc.wait()
                break

        code = proc.returncode
        if code == 0 and not hung:
            print("Scan termine.")
            return
        if code == 2 and not hung:
            print("Preflight ou configuration en echec : arret du superviseur.")
            return

        idx = read_beat()
        cause = "bloque" if hung else f"crash (code {code})"
        if idx.lstrip("-").isdigit() and int(idx) >= 0:
            start_fails = 0
            fails[idx] += 1
            print(f"[superviseur] point {idx} {cause}, essai {fails[idx]}/{MAX_FAILS}")
            if fails[idx] >= MAX_FAILS:
                with open(SKIP_FILE, "a") as fh:
                    fh.write(idx + "\n")
                print(f"[superviseur] point {idx} ajoute a {os.path.basename(SKIP_FILE)}")
        else:
            start_fails += 1
            print(f"[superviseur] echec au demarrage ({cause}), essai {start_fails}/{MAX_START_FAILS}")
            if start_fails >= MAX_START_FAILS:
                print("[superviseur] DWSIM ne demarre pas : arret.")
                return
        time.sleep(RESTART_DELAY)


def summarize():
    if not os.path.exists(OUTPUT_CSV):
        print("Aucun CSV a resumer.")
        return
    with open(OUTPUT_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    stats = Counter(r["status"].split(":")[0] for r in rows)
    ok = [r for r in rows if r.get("status") == "ok" and r.get("converged") == "1"]
    with open(CLEAN_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_header(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ok)
    print(f"\n=== DATASET : {len(rows)} lignes dans {os.path.basename(OUTPUT_CSV)} ===")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    print(f"Dataset propre ({len(ok)} points convergés) : {CLEAN_CSV}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    elif "--summary" in sys.argv:
        summarize()
    else:
        supervise()
        summarize()