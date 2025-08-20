#!/usr/bin/env python3
"""
v1.0
options_levels.py
- Adds absolute window size per symbol: --window-pts-spx, --window-pts-ndx (total width in points).
- Adds effective EM floor per symbol: --em-floor-spx, --em-floor-ndx (min EM used when window derives from EM).
- Adds forced far picks beyond the near band:
    --far-per-side (default 1)
    --far-min-mult (default 1.6)  # x EM distance from center
    --far-min-points (default 180) # or absolute distance in points (max(., mult*EM))
- Adds per-symbol plot count: --plot-count-spx, --plot-count-ndx
Keeps ZG/CW/PW/VT± and dynamic width caps from v7s.
"""

from __future__ import annotations

import argparse, datetime as dt, json, math, os, re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np, pandas as pd, requests
try:
    from zoneinfo import ZoneInfo
except Exception:
    from pytz import timezone as ZoneInfo

# Base URL for CBOE API
CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes"

# Symbols that require underscore prefix in CBOE queries
INDEX_UNDERSCORE = {"SPX","NDX","RUT","DJX","VIX","XSP"}

# Contract multipliers for different symbols
CONTRACT_MULTIPLIER = {"default":100.0,"SPX":100.0,"NDX":100.0,"RUT":100.0,"DJX":100.0,"XSP":100.0,"VIX":100.0}

def now_ny() -> dt.datetime: return dt.datetime.now(ZoneInfo("America/New_York"))

def cboe_symbol(sym:str)->str:
    """
    Convert symbol to CBOE format, adding underscore for indices if needed.
    """
    s=sym.upper().lstrip("^"); return f"_{s}" if s in INDEX_UNDERSCORE else s

# Regex for parsing OCC symbols
_OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")

def parse_occ(occ:str):
    """
    Parse an OCC symbol into underlying, expiry, type, and strike.
    """
    s=(occ or "").replace(" ","").upper(); m=_OCC.match(s)
    if not m: return None
    ul,yymmdd,typ,str8=m.groups()
    exp=dt.date(2000+int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    return ul, exp, typ, int(str8)/1000.0

def parse_expiry_from_json(o:dict):
    """
    Parse expiry date from option JSON using various keys.
    """
    for k in ("expirationDate","expiration_date","expiry","expiration"):
        v=o.get(k)
        if v is None: continue
        try: return pd.to_datetime(v).date()
        except Exception:
            try:
                s=str(v)
                if len(s)==8 and s.isdigit(): return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
                iv=int(v)
                if iv>10_000_000_000: iv//=1000
                return dt.datetime.utcfromtimestamp(iv).date()
            except Exception: pass
    return None

def fetch_chain_json(symbol:str, raw_dir:Optional[str]):
    """
    Fetch options chain JSON from CBOE and save to raw_dir if provided.
    """
    url=f"{CBOE_BASE}/options/{cboe_symbol(symbol)}.json"
    r=requests.get(url,timeout=25,headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"})
    r.raise_for_status(); j=r.json()
    ts=now_ny().strftime("%Y%m%dT%H%M%S%z")
    if raw_dir:
        os.makedirs(raw_dir,exist_ok=True)
        with open(os.path.join(raw_dir,f"cboe_{symbol}_{ts}.json"),"w",encoding="utf-8") as f: json.dump(j,f,indent=2)
    return j,ts

def json_to_frame(symbol:str,j:Dict):
    """
    Convert CBOE JSON to DataFrame of options and extract metadata.
    """
    options=j.get("data",{}).get("options") or j.get("options")
    if options is None: raise RuntimeError("no options list in Cboe JSON")
    meta={}
    for k in ("last","last_price","underlying_price","current_price"):
        v=j.get(k) or j.get("data",{}).get(k)
        if isinstance(v,(int,float)): meta[k]=float(v)
    q=j.get("data",{}).get("quote")
    if isinstance(q,dict):
        v=q.get("last") or q.get("last_price")
        if isinstance(v,(int,float)): meta["last"]=float(v)
    rows=[]
    for o in options:
        occ=o.get("option") or o.get("symbol") or ""
        p=parse_occ(occ) or (None, None, None, None)
        ul_occ,exp_occ,typ_occ,strike_occ=p
        exp=parse_expiry_from_json(o) or exp_occ
        if exp is None: continue
        typ=(o.get("type") or o.get("option_type") or typ_occ or "?").upper()[0]
        strike=o.get("strike", o.get("strike_price", strike_occ))
        if strike is None: continue
        bid,ask,mark=o.get("bid"),o.get("ask"),o.get("mark")
        if mark is None and bid is not None and ask is not None: mark=(float(bid)+float(ask))/2.0
        gamma=o.get("gamma") or (o.get("greeks") or {}).get("gamma",0.0)
        oi=o.get("open_interest", o.get("openInterest",0))
        rows.append({"underlying":ul_occ or symbol.upper(),"expiry":pd.to_datetime(exp).date(),"type":typ,
                     "strike":float(strike),
                     "bid":float(bid) if bid is not None else math.nan,
                     "ask":float(ask) if ask is not None else math.nan,
                     "mark":float(mark) if mark is not None else math.nan,
                     "open_interest":float(oi or 0.0),"gamma":float(gamma or 0.0)})
    df=pd.DataFrame(rows).sort_values(["expiry","strike","type"]).reset_index(drop=True)
    if df.empty: raise RuntimeError("no parsed options with expiry (parse failure)")
    return df,meta

def select_today_next(df:pd.DataFrame):
    """
    Select today's and next expiry from available expiries.
    """
    today=now_ny().date(); exps=sorted(df["expiry"].dropna().unique()); out=[]
    if today in exps: out.append(today)
    fut=[e for e in exps if e>today]
    if fut: out.append(fut[0])
    if not out and exps: out.append(exps[0])
    return out[:2]

def mid(a,b,m):
    """
    Compute mid price: mark if available, else bid/ask average.
    """
    if isinstance(m,(int,float)) and math.isfinite(m): return float(m)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)): return (float(a)+float(b))/2.0
    return math.nan

def atm_from_prices(df_exp:pd.DataFrame):
    """
    Find ATM strike and expected move from call/put price parity.
    """
    if df_exp.empty: return None,None
    c=df_exp[df_exp["type"]=="C"][["strike","bid","ask","mark"]].rename(columns={"bid":"c_bid","ask":"c_ask","mark":"c_mark"})
    p=df_exp[df_exp["type"]=="P"][["strike","bid","ask","mark"]].rename(columns={"bid":"p_bid","ask":"p_ask","mark":"p_mark"})
    mrg=pd.merge(c,p,on="strike",how="inner")
    if mrg.empty: return None,None
    mrg["c_mid"]=[mid(a,b,m) for a,b,m in zip(mrg["c_bid"],mrg["c_ask"],mrg["c_mark"])]
    mrg["p_mid"]=[mid(a,b,m) for a,b,m in zip(mrg["p_bid"],mrg["p_ask"],mrg["p_mark"])]
    mrg=mrg.dropna(subset=["c_mid","p_mid"])
    if mrg.empty: return None,None
    mrg["diff"]=(mrg["c_mid"]-mrg["p_mid"]).abs()
    row=mrg.iloc[mrg["diff"].idxmin()]
    return float(row["strike"]), float(row["c_mid"]+row["p_mid"])

def aggregate_by_strike(df_exp:pd.DataFrame,symbol:str)->pd.DataFrame:
    """
    Aggregate OI and GEX by strike.
    """
    mult=CONTRACT_MULTIPLIER.get(symbol.upper(),100.0)
    c=df_exp[df_exp["type"]=="C"].groupby("strike")["open_interest"].sum()
    p=df_exp[df_exp["type"]=="P"].groupby("strike")["open_interest"].sum()
    oi=(c+p).fillna(0.0); sign=df_exp["type"].map({"C":1.0,"P":-1.0}).fillna(0.0)
    g=(df_exp["gamma"].fillna(0.0)*df_exp["open_interest"].fillna(0.0)*mult*sign).groupby(df_exp["strike"]).sum()
    out=pd.DataFrame({"call_oi":c,"put_oi":p,"total_oi":oi,"gex":g}).fillna(0.0).sort_index(); out["abs_gex"]=out["gex"].abs()
    return out

def smooth_series(xs,ys,window:int=5):
    """
    Smooth series with moving average.
    """
    w=max(1,int(window)); 
    if w%2==0: w+=1
    pad=w//2; ypad=np.pad(ys,(pad,pad),mode="edge"); kern=np.ones(w)/w
    return np.convolve(ypad,kern,mode="valid")

def local_peaks(xs,ys):
    """
    Find local peak indices.
    """
    idx=[]; 
    for i in range(1,len(xs)-1):
        if ys[i]>=ys[i-1] and ys[i]>=ys[i+1]: idx.append(i)
    return idx

def pick_with_spacing(cands,min_spacing:float,k:int):
    """
    Pick top candidates with min spacing.
    """
    out=[]
    for x,score in sorted(cands,key=lambda t:t[1],reverse=True):
        if all(abs(x-y)>=min_spacing for y,_ in out): out.append((x,score))
        if len(out)>=k: break
    return out

def zero_crossings(xs,ys):
    """
    Find zero crossing positions.
    """
    zs=[]; prev=ys[0]
    for i in range(1,len(xs)):
        y=ys[i]
        if (prev<=0 and y>=0) or (prev>=0 and y<=0):
            x0,x1=xs[i-1],xs[i]; y0,y1=prev,y; t=0.5 if y1==y0 else (-y0/(y1-y0))
            zs.append(x0+t*(x1-x0))
        prev=y
    return zs

def zone_half_width_from_series(xs: np.ndarray, ys: np.ndarray, x0: float, frac: float = 0.5) -> Optional[float]:
    """
    Compute zone half-width at fraction of peak.
    """
    if xs.size==0 or ys.size==0: return None
    i0=int(np.argmin(np.abs(xs - x0)))
    y0=float(ys[i0])
    if not np.isfinite(y0) or y0<=0: return None
    thr = frac * y0
    il=i0
    while il>0 and ys[il] >= thr: il-=1
    ir=i0
    while ir < len(xs)-1 and ys[ir] >= thr: ir+=1
    left_x = float(xs[max(0, il)])
    right_x= float(xs[min(len(xs)-1, ir)])
    width = max(0.0, (right_x - left_x) * 0.5)
    return width if width>0 else None

def cap_hw(symbol:str, hw: Optional[float], em: Optional[float], cfg: dict) -> Optional[float]:
    """
    Cap half-width with config limits.
    """
    if hw is None or not np.isfinite(hw) or hw<=0: return None
    sym = symbol.upper()
    min_hw = float(cfg["box_hw_min"].get(sym, cfg["box_hw_min"]["DEFAULT"]))
    max_abs = float(cfg["box_hw_cap"].get(sym, cfg["box_hw_cap"]["DEFAULT"]))
    max_frac = float(cfg["box_hw_frac_em"].get(sym, cfg["box_hw_frac_em"]["DEFAULT"]))
    max_by_em = float(max_frac * em) if (em and em>0) else float("inf")
    hw = max(min(hw, max_abs, max_by_em), min_hw)
    return hw

@dataclass
class Level:
    run_date: dt.date
    symbol: str
    expiry: dt.date
    level: float
    label: str
    method: str
    notes: str = ""
    fut_level: Optional[float] = None
    side: Optional[str] = None
    zone_hw: Optional[float] = None

def build_plot_set(symbol, exp, xs, gex, abs_gex, tot_oi, call_oi, put_oi,
                   center, em_eff, em_raw, smooth_win, min_spacing, oi_floor,
                   near_thresh, far_thresh, vt_inner, vt_outer, combo_near_pct,
                   far_per_side, far_min_mult, far_min_points,
                   run_date, cfg):
    """
    Build levels for plotting.
    """
    sm_gex=smooth_series(xs,gex,window=smooth_win); sm_abs_gex=np.abs(sm_gex); sm_oi=smooth_series(xs,tot_oi,window=smooth_win)
    slope=np.abs(np.gradient(sm_gex,xs))
    out=[]
    def add(lbl,level,method,zone_hw=None):
        if level is not None and math.isfinite(level):
            side="+" if (center is not None and level>=center) else "-"
            out.append(Level(run_date,symbol,exp,float(level),lbl,method,"",None,side,zone_hw))

    # Core: ZG + VT at ZG
    zc=zero_crossings(xs,gex) if xs.size>1 else []
    zg=min(zc,key=lambda v:abs(v-center)) if zc else None
    add("Zero Gamma",zg,"gex_zero_win",None)
    if zg is not None: add("Volatility Trigger",zg,"gex_zero_win",None)

    # --- Walls (floor + optional fallback) ---
    use_fb = (cfg.get("wall_fallback","on") == "on")

    call_ix = np.where(call_oi >= float(oi_floor))[0]
    put_ix  = np.where(put_oi  >= float(oi_floor))[0]

    cw = xs[call_ix[np.argmax(call_oi[call_ix])]] if call_ix.size else (xs[np.argmax(call_oi)] if use_fb else None)
    pw = xs[put_ix[np.argmax(put_oi[put_ix])]]   if put_ix.size  else (xs[np.argmax(put_oi)] if use_fb else None)

    if cw is not None:
        cw_hw = cap_hw(symbol, zone_half_width_from_series(xs, sm_oi, cw, frac=cfg["wall_fwhm_frac"]), em_eff, cfg)
        add("Call Wall", cw, "max_call_oi_floor" if call_ix.size else "max_call_oi_any", cw_hw)

    if pw is not None:
        pw_hw = cap_hw(symbol, zone_half_width_from_series(xs, sm_oi, pw, frac=cfg["wall_fwhm_frac"]), em_eff, cfg)
        add("Put Wall",  pw, "max_put_oi_floor"  if put_ix.size  else "max_put_oi_any",  pw_hw)

    # VT± slope maxima in [vt_inner, vt_outer] * EM range
    if em_eff and em_eff>0:
        d=np.abs(xs-center); mask=(d>=vt_inner*em_eff)&(d<=vt_outer*em_eff)
        left=np.where((xs<center)&mask)[0]; right=np.where((xs>center)&mask)[0]
        if left.size: add("VT−", xs[left[np.argmax(slope[left])]], "slope_peak", None)
        if right.size: add("VT+", xs[right[np.argmax(slope[right])]], "slope_peak", None)

    # LG near
    near_mask=(np.abs(xs-center)<=1.2*em_eff) if (em_eff and em_eff>0) else np.ones_like(xs,dtype=bool)
    lg_idx=local_peaks(xs,sm_abs_gex)
    lg_cands=[(xs[i],sm_abs_gex[i]) for i in lg_idx if sm_abs_gex[i]>=near_thresh*float(np.max(sm_abs_gex)) and near_mask[i]]
    for side in ("L", "R"):
        if side == "L":
            pool = [(x, s) for (x, s) in lg_cands if x < center]
        else:
            pool = [(x, s) for (x, s) in lg_cands if x > center]

        keep = pick_with_spacing(pool, min_spacing, k=2)
        for r, (x, s) in enumerate(keep, start=1):
            hw_raw = zone_half_width_from_series(xs, sm_abs_gex, x, frac=cfg["lg_fwhm_frac"])
            hw = cap_hw(symbol, hw_raw, em_eff, cfg)
            add(f"Large Gamma {r}", x, "lg_near", hw)


    # LG far — forced picks
    if em_eff and em_eff>0 and far_per_side>0:
        far_gate = max(far_min_mult*em_eff, far_min_points)  # absolute distance in points from center
        far_mask = np.abs(xs-center) >= far_gate
        lg_far=[(xs[i],sm_abs_gex[i]) for i in lg_idx if sm_abs_gex[i]>=far_thresh*float(np.max(sm_abs_gex)) and far_mask[i]]
        for side in ("L", "R"):
            if side == "L":
                side_cands = [(x, s) for (x, s) in lg_far if x < center]
            else:
                side_cands = [(x, s) for (x, s) in lg_far if x > center]

            keep = pick_with_spacing(side_cands, min_spacing, k=far_per_side)
            for i, (x, s) in enumerate(keep[:far_per_side], start=1):
                hw_raw = zone_half_width_from_series(xs, sm_abs_gex, x, frac=cfg["lg_fwhm_frac"])
                hw = cap_hw(symbol, hw_raw, em_eff, cfg)
                add("Large Gamma (far)", x, "lg_far_forced", hw)


    # Combo clusters (near + far)
    thr = cfg["combo_gate_frac"] * float(np.max(sm_oi)) if np.max(sm_oi)>0 else float("inf")
    i=0; clusters=[]
    while i < len(xs):
        if sm_oi[i] >= thr:
            j=i
            while j+1 < len(xs) and sm_oi[j+1] >= thr: j+=1
            seg_x = xs[i:j+1]; seg_y = sm_oi[i:j+1]
            mass = float(np.trapz(seg_y, seg_x))
            if mass>0:
                cx = float(np.trapz(seg_x*seg_y, seg_x) / mass)
                hw_raw = (seg_x[-1] - seg_x[0]) * 0.5
                clusters.append((cx, mass, j-i+1, hw_raw))
            i=j+1
        else:
            i+=1
    if clusters:
        max_mass=max(m for _,m,_,_ in clusters)
        keep=[(x,m,c,hw) for (x,m,c,hw) in clusters if m>=cfg["combo_keep_frac"]*max_mass and c>=3]
        near=[(x,m,hw) for (x,m,c,hw) in keep if (em_eff and abs(x-center)<=combo_near_pct*em_eff)]
        far=[(x,m,hw) for (x,m,c,hw) in keep if not (em_eff and abs(x-center)<=combo_near_pct*em_eff)]
        def pick_side(arr,k=1):
            arr_sorted=sorted(arr, key=lambda t: abs(t[0]-center))
            return arr_sorted[:k]
        for left in pick_side([t for t in near if t[0]<center],1):
            add("Combo", left[0], "combo_cluster", cap_hw(symbol,left[2],em_eff,cfg))
        for right in pick_side([t for t in near if t[0]>center],1):
            add("Combo", right[0], "combo_cluster", cap_hw(symbol,right[2],em_eff,cfg))
        # far combos forced (at most 1 per side)
        for left in pick_side([t for t in far if t[0]<center],1):
            add("Combo (far)", left[0], "combo_cluster_far", cap_hw(symbol,left[2],em_eff,cfg))
        for right in pick_side([t for t in far if t[0]>center],1):
            add("Combo (far)", right[0], "combo_cluster_far", cap_hw(symbol,right[2],em_eff,cfg))

    # scoring
    S=np.abs(np.gradient(sm_gex,xs)); Smax=float(np.max(S)) if len(S) else 1.0
    max_abs=float(np.max(np.abs(abs_gex))) if len(abs_gex) else 1.0
    def nearest_idx(x): return int(np.argmin(np.abs(xs-x)))
    for L in out:
        i=nearest_idx(L.level)
        gex_norm=(abs(abs_gex[i])/(max_abs or 1.0)); oi_norm=float(tot_oi[i])/((float(np.max(tot_oi))) or 1.0); slope_norm=float(S[i])/(Smax or 1.0)
        dist=abs(L.level-center)/(em_eff if (em_eff and em_eff>0) else max(1.0,center*0.01))
        edge_bonus=0.12 if (1.3<=dist<=2.5) else (0.10 if (0.8<=dist<=1.3) else 0.0)
        mbonus={"Zero Gamma":0.40,"Volatility Trigger":0.40,"VT+":0.40,"VT−":0.40,"Call Wall":0.30,"Put Wall":0.30,
                "Large Gamma (far)":0.28,"Combo (far)":0.24,"Combo":0.20}
        L._score=0.46*gex_norm+0.24*oi_norm+0.30*slope_norm+edge_bonus+mbonus.get(L.label,0.10)  # type: ignore
    out.sort(key=lambda L:getattr(L,"_score",0.0),reverse=True)
    return out

def compute_for_symbol(symbol, raw_dir, raw_ladders, em_mult, window_pct,
                       em_floor_map, window_pts_map,
                       oi_cover, min_oi, smooth_win, min_spacing, oi_floor, near_thresh,
                       far_thresh, vt_inner, vt_outer, combo_near_pct,
                       far_per_side, far_min_mult, far_min_points, cfg):
    """
    Compute levels for a symbol.
    """
    j,_=fetch_chain_json(symbol,raw_dir); df,meta=json_to_frame(symbol,j); exps=select_today_next(df)
    spot=None
    for k in ("last","last_price","underlying_price","current_price"):
        v=meta.get(k)
        if isinstance(v,(int,float)) and math.isfinite(v): spot=float(v); break
    dfe_today=df[df["expiry"]==exps[0]].copy() if exps else pd.DataFrame()
    atm_k_today,em_today=atm_from_prices(dfe_today)
    if spot is None and atm_k_today is not None: spot=float(atm_k_today)
    if spot is None:
        tmp=df.groupby("strike")["open_interest"].sum(); spot=float((tmp.index*tmp).sum()/max(tmp.sum(),1.0))
    diagnostics={"center":spot,"expiries":[str(e) for e in exps],"windows":{},"smooth_win":smooth_win,"min_spacing":min_spacing}
    run_date=now_ny().date(); levels=[]; plot_levels=[]

    for exp in exps:
        dfe=df[df["expiry"]==exp].copy()
        if raw_ladders:
            os.makedirs(raw_ladders,exist_ok=True); aggregate_by_strike(dfe,symbol).to_csv(os.path.join(raw_ladders,f"ladder_full_{symbol}_{exp}.csv"))
        atm_k,em_raw=atm_from_prices(dfe); center=spot if spot is not None else (atm_k if atm_k is not None else None)

        # window determination
        win_pts = window_pts_map.get(symbol.upper(), 0.0) or 0.0
        if win_pts>0:
            lo,hi = center - 0.5*win_pts, center + 0.5*win_pts
            em_eff = win_pts/2.0  # used only for relative gating/labels
        elif em_raw and em_raw>0:
            em_eff = max(em_raw, em_floor_map.get(symbol.upper(), 0.0))
            lo,hi = center - em_mult*em_eff, center + em_mult*em_eff
        else:
            lo,hi = center - window_pct*center, center + window_pct*center
            em_eff = (hi - lo) / (2.0*em_mult) if em_mult>0 else (hi - lo)/2.0

        dfe_win=dfe[(dfe["strike"]>=lo)&(dfe["strike"]<=hi)].copy(); ladder=aggregate_by_strike(dfe_win,symbol)
        if raw_ladders: ladder.to_csv(os.path.join(raw_ladders,f"ladder_win_{symbol}_{exp}.csv"))
        diagnostics["windows"][str(exp)]={"center":center,"lo":lo,"hi":hi,"rows":int(ladder.shape[0]),"em_raw":em_raw,"em_eff":em_eff,"win_pts":hi-lo}
        if ladder.empty: continue

        xs=ladder.index.values.astype(float); gex=ladder["gex"].values.astype(float); abs_gex=ladder["abs_gex"].values.astype(float)
        tot_oi=ladder["total_oi"].values.astype(float); call_oi=ladder["call_oi"].values.astype(float); put_oi=ladder["put_oi"].values.astype(float)

        # baselines for 'near' (CW/PW) respecting --wall-fallback
        use_fb = True  # default
        try:
            use_fb = (cfg.get("wall_fallback","on") == "on")
        except Exception:
            pass

        cw_ix = np.where(call_oi >= float(min_oi))[0]
        pw_ix = np.where(put_oi  >= float(min_oi))[0]

        call_wall = float(xs[cw_ix[np.argmax(call_oi[cw_ix])]]) if cw_ix.size else (float(xs[np.argmax(call_oi)]) if use_fb else None)
        put_wall  = float(xs[pw_ix[np.argmax(put_oi[pw_ix])]])  if pw_ix.size  else (float(xs[np.argmax(put_oi)]) if use_fb else None)

        if call_wall is not None:
            levels.append(Level(run_date, symbol, exp, call_wall, "Call Wall",
                                "max_call_oi_win" if cw_ix.size else "max_call_oi_any"))

        if put_wall is not None:
            levels.append(Level(run_date, symbol, exp, put_wall,  "Put Wall",
                                "max_put_oi_win"  if pw_ix.size else "max_put_oi_any"))


        # curated/final
        cfg['combo_near_pct']=combo_near_pct
        plot_levels.extend(build_plot_set(symbol,exp,xs,gex,abs_gex,tot_oi,call_oi,put_oi,center,em_eff,em_raw,
                                          smooth_win,min_spacing,oi_floor,near_thresh,far_thresh,vt_inner,vt_outer,
                                          combo_near_pct, far_per_side, far_min_mult, far_min_points,
                                          run_date,cfg))
    return levels, plot_levels, diagnostics

def parse_basis_arg(basis_str: Optional[str]) -> Dict[str, float]:
    """
    Parse basis argument string into dict.
    """
    out={}
    if not basis_str: return out
    for p in [p.strip() for p in basis_str.split(",") if p.strip()]:
        if "=" in p:
            k,v=p.split("=",1)
            try: out[k.strip().upper()]=float(v.strip())
            except Exception: pass
    return out

@dataclass
class ArgsWrap:
    plot_count: int
    plot_count_spx: Optional[int]
    plot_count_ndx: Optional[int]
    min_spacing_map: Dict[str,float]

def write_levels_csv(levels: List[Level], out_dir: str, tag: str = "levels") -> str:
    """
    Write levels to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not levels:
        path = os.path.join(out_dir, f"{tag}_EMPTY.csv")
        pd.DataFrame([]).to_csv(path, index=False)
        return path
    run_date = max(L.run_date for L in levels).strftime("%Y%m%d")
    syms = "-".join(sorted({L.symbol for L in levels}))
    path = os.path.join(out_dir, f"{tag}_{syms}_{run_date}.csv")
    pd.DataFrame([L.__dict__ for L in levels]).to_csv(path, index=False)
    return path

def write_pine_payload(plot_levels: List[Level], diagnostics: Dict[str, Dict], out_dir: str, basis_map: Dict[str, float]) -> List[str]:
    """
    Write Pine payload files.
    """
    os.makedirs(out_dir, exist_ok=True)
    def to_tag(lbl:str)->str:
        if lbl=="Zero Gamma": return "ZG"
        if lbl in ("VT+","VT−","VT-"): return "VT+" if "VT+" in lbl else "VT-"
        if lbl=="Volatility Trigger": return "ZG"
        if lbl=="Call Wall": return "CW"
        if lbl=="Put Wall": return "PW"
        if lbl.startswith("Large Gamma (far)"): return "LGF"
        if lbl.startswith("Combo (far)"): return "CBF"
        if lbl.startswith("Large Gamma"):
            import re as _re; m=_re.search(r"(\d+)", lbl); return f"LG{(m.group(1) if m else '1')}"
        if lbl.startswith("Combo"): return "CB"
        return lbl.replace(" ","")
    by_sym={}
    for L in plot_levels: by_sym.setdefault(L.symbol,[]).append(L)
    files=[]
    for sym, arr in by_sym.items():
        center = diagnostics.get(sym, {}).get("center")
        basis  = basis_map.get(sym.upper(), 0.0)
        center_fut = center + basis if center is not None else None
        date = max(L.run_date for L in arr)
        header = f"SYMBOL={sym};DATE={date};CENTER={center_fut if center_fut is not None else ''};BASIS={basis}"
        lines = [header]; seen_zg=False
        for L in arr:
            tag = to_tag(L.label)
            if tag=="ZG":
                if seen_zg: continue
                seen_zg=True
            side = "+" if (center is not None and L.level>=center) else "-"
            price = L.fut_level if L.fut_level is not None else L.level
            if L.zone_hw and L.zone_hw>0:
                lines.append(f"{tag}{side},{price:.2f},{L.zone_hw:.2f}")
            else:
                lines.append(f"{tag}{side},{price:.2f}")
        fname=os.path.join(out_dir,f"levels_pine_{sym}_{date.strftime('%Y%m%d')}.txt")
        with open(fname,"w",encoding="utf-8") as f: f.write("\n".join(lines))
        files.append(fname)
    return files

def main(argv=None):
    """
    Main entry point.
    """
    ap=argparse.ArgumentParser(description="SPX/NDX options levels (extended-range, far picks, width caps)")
    ap.add_argument("--symbols",nargs="+",default=["SPX","NDX"])
    ap.add_argument("--out",default="out"); ap.add_argument("--raw",default="raw"); ap.add_argument("--raw-ladders",default="raw")
    # window
    ap.add_argument("--em-mult",type=float,default=2.5); ap.add_argument("--window-pct",type=float,default=0.03)
    ap.add_argument("--em-floor-spx",type=float,default=0.0); ap.add_argument("--em-floor-ndx",type=float,default=0.0)
    ap.add_argument("--window-pts-spx",type=float,default=0.0); ap.add_argument("--window-pts-ndx",type=float,default=0.0)
    # selection filters
    ap.add_argument("--oi-cover",type=float,default=0.70); ap.add_argument("--min-oi",type=float,default=150.0)
    ap.add_argument("--smooth-win",type=int,default=5)
    ap.add_argument("--min-spacing-spx",type=float,default=10.0); ap.add_argument("--min-spacing-ndx",type=float,default=25.0)
    ap.add_argument("--near-count",type=int,default=16)  # kept for near.csv writer if needed
    ap.add_argument("--plot-count",type=int,default=12)
    ap.add_argument("--plot-count-spx",type=int); ap.add_argument("--plot-count-ndx",type=int)
    ap.add_argument("--oi-floor-spx",type=float,default=200.0); ap.add_argument("--oi-floor-ndx",type=float,default=400.0)
    ap.add_argument("--lg-near-thresh",type=float,default=0.35); ap.add_argument("--lg-far-thresh",type=float,default=0.55)
    ap.add_argument("--vt-inner",type=float,default=0.30); ap.add_argument("--vt-outer",type=float,default=1.20)
    ap.add_argument("--combo-near-pct",type=float,default=0.90)
    ap.add_argument("--wall-fallback", choices=["on","off"], default="on",
                help="If ON, CW/PW fall back to max OI in window when floor not met; if OFF, CW/PW require floors and may be absent.")
    # far-pick controls
    ap.add_argument("--far-per-side",type=int,default=1)
    ap.add_argument("--far-min-mult",type=float,default=1.6)
    ap.add_argument("--far-min-points",type=float,default=180.0)
    # basis
    ap.add_argument("--basis",default="SPX=20,NDX=110")
    # width caps
    ap.add_argument("--box-hw-cap-spx",type=float,default=15.0)
    ap.add_argument("--box-hw-cap-ndx",type=float,default=35.0)
    ap.add_argument("--box-hw-min-spx",type=float,default=1.0)
    ap.add_argument("--box-hw-min-ndx",type=float,default=6.0)
    ap.add_argument("--box-hw-frac-em-spx",type=float,default=0.18)
    ap.add_argument("--box-hw-frac-em-ndx",type=float,default=0.18)
    ap.add_argument("--lg-fwhm-frac",type=float,default=0.55)
    ap.add_argument("--cb-fwhm-frac",type=float,default=0.55)
    ap.add_argument("--wall-fwhm-frac",type=float,default=0.45)
    args=ap.parse_args(argv)

    min_spacing_map={"SPX":args.min_spacing_spx,"NDX":args.min_spacing_ndx}
    oi_floor_map={"SPX":args.oi_floor_spx,"NDX":args.oi_floor_ndx}
    em_floor_map={"SPX":args.em_floor_spx,"NDX":args.em_floor_ndx}
    window_pts_map={"SPX":args.window_pts_spx,"NDX":args.window_pts_ndx}
    plot_count_map={"SPX": args.plot_count_spx or args.plot_count, "NDX": args.plot_count_ndx or args.plot_count}

    cfg={
        "box_hw_cap":{"SPX":args.box_hw_cap_spx,"NDX":args.box_hw_cap_ndx,"DEFAULT":30.0},
        "box_hw_min":{"SPX":args.box_hw_min_spx,"NDX":args.box_hw_min_ndx,"DEFAULT":2.0},
        "box_hw_frac_em":{"SPX":args.box_hw_frac_em_spx,"NDX":args.box_hw_frac_em_ndx,"DEFAULT":0.20},
        "lg_fwhm_frac":args.lg_fwhm_frac,
        "combo_gate_frac":0.40,
        "combo_keep_frac":0.40,
        "wall_fwhm_frac":args.wall_fwhm_frac,
        "wall_fallback": args.wall_fallback,
    }

    all_levels=[]; plot_levels_all=[]; diags={}
    for sym in args.symbols:
        try:
            lvls,plot_lvls,diag=compute_for_symbol(sym,args.raw or None,args.raw_ladders or None,
                args.em_mult,args.window_pct,em_floor_map,window_pts_map,
                args.oi_cover,args.min_oi,args.smooth_win,min_spacing_map.get(sym.upper(),10.0),
                oi_floor_map.get(sym.upper(),200.0),args.lg_near_thresh,args.lg_far_thresh,
                args.vt_inner,args.vt_outer,args.combo_near_pct,
                args.far_per_side,args.far_min_mult,args.far_min_points,cfg)
            all_levels.extend(lvls); plot_levels_all.extend(plot_lvls); diags[sym]=diag
        except Exception as e:
            print(f"[WARN] {sym}: {e}")

    basis_map={}
    for p in (args.basis or "").split(","):
        if "=" in p:
            k,v=p.split("=",1)
            try: basis_map[k.strip().upper()]=float(v.strip())
            except: pass
    for L in all_levels+plot_levels_all:
        b=basis_map.get(L.symbol.upper())
        if isinstance(b,(int,float)): L.fut_level=float(L.level+b)

    # Per-symbol curation, spacing, and coalescing (fixed scope, no functionality loss)
    final_plot = []
    by_sym = {}
    reserved = {"Zero Gamma", "Volatility Trigger", "Call Wall", "Put Wall", "VT+", "VT−", "VT-"}
    tol_map = {"SPX": 6.0, "NDX": 15.0}  # coalesce tolerance per symbol (pts)

    for L in plot_levels_all:
        by_sym.setdefault(L.symbol, []).append(L)

    for sym, arr in by_sym.items():
        # 1) Split reserved vs the rest
        pri  = [L for L in arr if L.label in reserved]
        rest = [L for L in arr if L.label not in reserved]

        # Prefer higher-scored reserved when collisions happen later
        pri.sort(key=lambda L: getattr(L, "_score", 0.0), reverse=True)

        # 2) Start keep with reserved, add "rest" gated by spacing & score
        keep = pri[:]
        spacing = min_spacing_map.get(sym.upper(), 10.0)
        limit   = int(plot_count_map.get(sym.upper(), args.plot_count))

        def spaced_ok(L):
            return all(abs(L.level - K.level) >= spacing for K in keep)

        for L in sorted(rest, key=lambda L: getattr(L, "_score", 0.0), reverse=True):
            if spaced_ok(L):
                keep.append(L)
            if len(keep) >= max(1, limit):
                break

        # 3) Enforce single CW and single PW (keep the best-scoring instance)
        seen_wall = {"Call Wall": False, "Put Wall": False}
        wall_filtered = []
        for L in sorted(keep, key=lambda x: getattr(x, "_score", 0.0), reverse=True):
            if L.label in seen_wall:
                if seen_wall[L.label]:
                    continue
                seen_wall[L.label] = True
            wall_filtered.append(L)

        # 4) Coalesce near-colliding levels per symbol (merge labels, keep widest zone)
        tol = tol_map.get(sym.upper(), 10.0)
        wall_filtered.sort(key=lambda L: L.level)

        merged = []
        for L in wall_filtered:
            if merged and abs(L.level - merged[-1].level) <= tol:
                A = merged[-1]
                # keep stronger by score; carry composite label; widen zone if needed
                best  = max((A, L), key=lambda x: getattr(x, "_score", 0.0))
                other = L if best is A else A
                if other.label not in best.label:
                    best.label = f"{best.label}/{other.label}"
                best.zone_hw = max(best.zone_hw or 0.0, other.zone_hw or 0.0) or None
                merged[-1] = best
            else:
                merged.append(L)

        # 5) Safety truncate to limit after coalescing
        merged = sorted(merged, key=lambda x: getattr(x, "_score", 0.0), reverse=True)[:limit]
        final_plot.extend(merged)


    os.makedirs(args.out,exist_ok=True)
    def write_levels_csv(levels, tag):
        if not levels:
            path=os.path.join(args.out,f"{tag}_EMPTY.csv"); pd.DataFrame([]).to_csv(path,index=False); return path
        run_date=max(L.run_date for L in levels).strftime("%Y%m%d")
        syms="-".join(sorted({L.symbol for L in levels}))
        path=os.path.join(args.out,f"{tag}_{syms}_{run_date}.csv")
        pd.DataFrame([L.__dict__ for L in levels]).to_csv(path,index=False); return path
    out_levels=write_levels_csv(all_levels,"levels")
    out_plot=write_levels_csv(final_plot,"levels_plot")
    pine_files=write_pine_payload(final_plot,diags,args.out,basis_map)
    with open(os.path.join(args.out,"diagnostics.json"),"w",encoding="utf-8") as f: json.dump(diags,f,indent=2)
    print(f"Wrote {out_levels}")
    print(f"Wrote {out_plot}")
    for p in pine_files: print(f"Wrote {p}")
    print(f"Wrote {os.path.join(args.out,'diagnostics.json')}")

if __name__=="__main__": main()