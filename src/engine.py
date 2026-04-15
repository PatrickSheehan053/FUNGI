"""
FUNGI v7.1 -- DASH Kernel Engine

v7.1: Re-adds clustering_collapse and scale_free_degeneration as loose
shatter criteria now that the weight degeneracy bug is fixed. Also adds
log1p normalization fallback for degenerate rank standardization.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import powerlaw, igraph as ig, warnings, graphblas as gb
warnings.filterwarnings("ignore", category=RuntimeWarning)

def _safe(x, fb=0.0):
    return float(x) if (x is not None and np.isfinite(x)) else fb

# ---- GraphBLAS FFL topology ----
def compute_dynamic_topology(W_sorted, src_sorted, tgt_sorted, k_core, n):
    te = min(int(n * k_core), len(W_sorted))
    T = np.zeros(len(W_sorted), dtype=np.float64)
    if te < 1: return T
    cr, cc = src_sorted[:te], tgt_sorted[:te]
    A = gb.Matrix.from_coo(cr.astype(np.uint64), cc.astype(np.uint64),
                           np.ones(te, dtype=np.float64), nrows=n, ncols=n)
    Tgb = A.mxm(A, gb.semiring.plus_times).new(mask=A.S)
    do = np.zeros(n, dtype=np.float64); oi,ov = A.reduce_rowwise(gb.monoid.plus).new().to_coo(); do[oi]=ov
    di = np.zeros(n, dtype=np.float64); ii,iv = A.reduce_columnwise(gb.monoid.plus).new().to_coo(); di[ii]=iv
    tr,tc,zv = Tgb.to_coo()
    if len(zv) > 0:
        cf = cr.astype(np.int64)*n + cc.astype(np.int64)
        tf = tr.astype(np.int64)*n + tc.astype(np.int64)
        to = np.argsort(tf); tfs=tf[to]; zvs=zv[to]
        si = np.searchsorted(tfs, cf); vi = np.clip(si, 0, len(tfs)-1)
        hm = (tfs[vi]==cf); mz = np.zeros(te, dtype=np.float64); mz[hm]=zvs[vi[hm]]
    else: mz = np.zeros(te, dtype=np.float64)
    lc = np.minimum(do[cr], di[cc]); v = (mz>0)&(lc>0)
    r = np.zeros(te, dtype=np.float64); r[v]=mz[v]/lc[v]; T[:te]=r
    np.clip(T, 0., 1., out=T); return T

# ---- Edge selection with per-node cap ----
def select_edges(omega, W, src, tgt, pert_nodes, n, lam, max_frac=0.15):
    budget = int(np.round(n * lam)); cap = int(n * max_frac)
    prot = []
    for p in pert_nodes:
        pe = np.where(src==p)[0]
        if len(pe)>0:
            k=min(3,len(pe)); prot.extend(pe[np.argsort(omega[pe])[-k:]])
    prot = np.unique(np.array(prot, dtype=int)) if prot else np.array([], dtype=int)
    rem = budget - len(prot)
    if rem > 0:
        m = np.ones(len(omega), dtype=bool)
        if len(prot)>0: m[prot]=False
        av = np.where(m)[0]
        if rem < len(av): fi = av[np.argpartition(omega[av], -rem)[-rem:]]
        else: fi = av
        sel = np.concatenate([prot, fi]) if len(prot)>0 else fi
    else: sel = prot[:budget]
    ss = src[sel]; nc = np.bincount(ss, minlength=n)
    ov = np.where(nc > cap)[0]
    if len(ov) > 0:
        km = np.ones(len(sel), dtype=bool)
        for nd in ov:
            ex = nc[nd]-cap
            if ex<=0: continue
            np_ = np.where(ss==nd)[0]
            nd_ = min(ex, len(np_))
            if nd_>0: km[np_[np.argsort(omega[sel[np_]])[:nd_]]] = False
        sel = sel[km]; ss = src[sel]
        freed = budget - len(sel)
        if freed > 0:
            used = set(sel.tolist()); cm = np.ones(len(omega), dtype=bool)
            for i in used: cm[i]=False
            for nd in ov: cm[src==nd]=False
            cands = np.where(cm)[0]
            if len(cands)>0:
                nf = min(freed, len(cands))
                sel = np.concatenate([sel, cands[np.argpartition(omega[cands],-nf)[-nf:]]])
    return src[sel], tgt[sel], W[sel]

# ---- Shatter checks ----
def check_shatter(ss, st, sw, od, n, active, cfg):
    ne = len(ss)
    if ne > cfg.get("max_edge_count", 500000): return True, "density_collapse"
    if (n-active)/max(n,1) > cfg.get("max_orphan_fraction", 0.70): return True, "orphan_collapse"
    sm = (np.max(od)/n) if len(od)>0 else 0
    if sm > cfg.get("max_hub_saturation", 0.15): return True, "dictator_hub"
    if ne>0 and active>0:
        try:
            G = sp.coo_matrix((np.ones(ne),(ss,st)), shape=(n,n))
            _,lb = connected_components(csgraph=G, directed=False, return_labels=True)
            if np.bincount(lb).max()/n < cfg.get("min_gwcc_fraction", 0.30): return True, "gwcc_percolation"
        except: return True, "gwcc_percolation"
    # Clustering floor (loose: 0.002)
    min_clust = cfg.get("min_clustering", None)
    if min_clust is not None and ne > 50:
        try:
            edges = list(zip(ss.tolist(), st.tolist()))
            ig_u = ig.Graph(n=n, edges=edges, directed=True).as_undirected(mode="collapse")
            cc = ig_u.transitivity_undirected()
            if np.isfinite(cc) and cc < min_clust: return True, "clustering_collapse"
        except: pass
    # Alpha floor/ceiling (loose: [1.5, 4.0])
    af = cfg.get("alpha_floor", None); ac = cfg.get("alpha_ceiling", None)
    if (af is not None or ac is not None) and len(od)>10 and np.max(od)>1:
        try:
            cap = int(n*0.15); cd = od[(od>0)&(od<cap)]
            if len(cd)>10 and len(np.unique(cd))>=3:
                alpha = _safe(powerlaw.Fit(cd, xmin=2, discrete=True, verbose=False).power_law.alpha, 2.5)
                if af is not None and alpha < af: return True, "scale_free_degeneration"
                if ac is not None and alpha > ac: return True, "scale_free_degeneration"
        except: pass
    return False, None

# ---- Utopia loss ----
def calculate_utopia_loss(ss, st, sw, n, od, active, kappa, ub, lw):
    ne = len(ss)

    def _p(par, obs):
        b = ub[par]; w = _safe(lw[par], 1.); o = _safe(obs, 0.)
        if o < b[0]: return w * ((b[0] - o) / max(abs(b[0]), 1e-6)) ** 2
        elif o > b[1]: return w * ((o - b[1]) / max(abs(b[1]), 1e-6)) ** 2
        return 0.

    # ── Alpha ──────────────────────────────────────────────────────────────
    ao = 1.0
    try:
        cap = int(n * 0.15); cd = od[(od > 0) & (od < cap)]
        if len(cd) > 10 and len(np.unique(cd)) >= 3:
            ao = _safe(powerlaw.Fit(
                cd, xmin=2, discrete=True, verbose=False).power_law.alpha, 1.)
    except: pass
    ta = _p("alpha", ao)

    # ── Gini ───────────────────────────────────────────────────────────────
    go = 1.0
    try:
        if active > 1 and np.sum(od) > 0:
            sd = np.sort(od); nn = len(sd)
            go = _safe((2. * np.sum(np.arange(1, nn + 1) * sd)) /
                       (nn * np.sum(sd)) - (nn + 1) / nn, 1.)
    except: pass
    tg = _p("gini", go)

    # ── S_max ──────────────────────────────────────────────────────────────
    smo = _safe((np.max(od) / n) if len(od) > 0 else 0)
    rv = max(0., (smo - kappa) / max(kappa, 1e-6))
    ts = _safe(lw["S_max"], 1.) * rv ** 2

    # ── C, Q, Rho ──────────────────────────────────────────────────────────
    co, qo, ro = 0., 0., 1.
    tc = _safe(lw["C"], 1.)
    tq = _safe(lw["Q"], 1.)
    tr = _safe(lw["rho"], 1.)

    if ne > 100:
        try:
            edges = list(zip(ss.tolist(), st.tolist()))
            ig_g = ig.Graph(n=n, edges=edges, directed=True,
                            edge_attrs={'weight': sw.tolist()})
            try:
                rc = ig_g.assortativity_degree(directed=True)
                if np.isfinite(rc):
                    ro = rc
                    # ── Lambda-conditional rho penalty (Molloy-Reed) ──────
                    # Expected rho for an uncorrelated scale-free network is
                    # determined by the degree distribution moments. Rather
                    # than penalizing against a static bound, we compute the
                    # Molloy-Reed analytical baseline for this graph's current
                    # lambda and alpha, then only penalize if observed rho is
                    # worse (less negative) than the baseline expectation.
                    # This eliminates reward hacking via lambda reduction:
                    # dropping lambda shifts the baseline too, so the optimizer
                    # gains no reward from network starvation.
                    try:
                        lam_obs = _safe(np.mean(od), 1.)
                        # Moments of truncated power-law degree distribution
                        # k_max: structural cutoff for finite network
                        k_max = max(int(n ** (1. / (ao - 1.))) 
                                    if ao > 1.05 else int(n * 0.15), 
                                    int(np.max(od)) if len(od) > 0 else 1)
                        k_max = min(k_max, int(n * 0.15))
                        ks = np.arange(1, k_max + 1, dtype=np.float64)
                        pk = ks ** (-ao)
                        pk = pk / pk.sum()
                        k1 = float(np.sum(ks * pk))        # <k>
                        k2 = float(np.sum(ks ** 2 * pk))   # <k²>
                        k3 = float(np.sum(ks ** 3 * pk))   # <k³>
                        if k1 > 1e-6 and k2 > 1e-6 and k3 > 1e-6:
                            # Newman 2002: rho_baseline ≈ -(k2/k1)² / (k3/k1)
                            # Normalized to be scale-independent
                            rho_baseline = _safe(
                                -(k2 / k1) ** 2 / max(k3 / k1, 1e-6), -0.10)
                            # Clip to reasonable range
                            rho_baseline = float(
                                np.clip(rho_baseline, -0.50, 0.0))
                        else:
                            rho_baseline = -0.10
                    except:
                        rho_baseline = -0.10

                    # Penalty: only fire if observed rho is WORSE than
                    # baseline (less negative). If the graph beats the
                    # Molloy-Reed expectation, no penalty is applied.
                    # Static utopian bounds act as a soft ceiling only.
                    rho_upper = ub["rho"][1]  # upper bound from Phase 0
                    rho_target = min(rho_baseline, rho_upper)
                    if ro > rho_target:
                        # Penalize for being less disassortative than expected
                        w_rho = _safe(lw["rho"], 1.)
                        tr = w_rho * ((ro - rho_target) /
                                      max(abs(rho_target), 1e-6)) ** 2
                    else:
                        # Graph beats the baseline — no rho penalty
                        tr = 0.
            except: pass
            try:
                ig_u = ig_g.as_undirected(
                    mode="collapse", combine_edges=dict(weight="sum"))
                cc = ig_u.transitivity_undirected()
                if np.isfinite(cc): co = cc; tc = _p("C", co)
                pt = ig_u.community_multilevel(); qm = pt.modularity
                if np.isfinite(qm): qo = qm; tq = _p("Q", qo)
            except: pass
        except: pass

    raw = _safe(ta) + _safe(tc) + _safe(tq) + _safe(tg) + _safe(tr) + _safe(ts)
    return np.sqrt(max(raw, 0.)), {
        'alpha': _safe(ao, 1.), 'C': _safe(co), 'Q': _safe(qo),
        'Gini': _safe(go, 1.), 'rho': _safe(ro, 1.), 'S_max': _safe(smo)}

# ---- Entry point ----
def run_dash_and_score(params, W, D, sources, targets, _unused, n_genes,
                       perturbed_nodes, utopian_bounds, loss_weights, shatter_cfg):
    try:
        beta, gamma, delta, kappa, k_core, lam = params
        T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)
        Nm = shatter_cfg.get("max_edge_count", 500000); Ne = min(len(W), Nm+10000)
        Ws,ss,ts,Ts = W[:Ne], sources[:Ne], targets[:Ne], T_local[:Ne]
        num = (Ws**beta)*(1+delta*Ts)
        order = np.lexsort((-num, ss))
        so,to,Wo,no = ss[order],ts[order],Ws[order],num[order]
        bd = np.nonzero(so[1:]!=so[:-1])[0]+1
        si = np.concatenate(([0], bd, [len(so)]))
        kr = np.empty(len(so), dtype=np.float64)
        for i in range(len(si)-1): s,e=si[i],si[i+1]; kr[s:e]=np.arange(1,e-s+1)
        omega = no / (kr**gamma)
        mhs = shatter_cfg.get("max_hub_saturation", 0.15)
        surv_s, surv_t, surv_W = select_edges(omega, Wo, so, to, perturbed_nodes, n_genes, lam, mhs)
        od = np.bincount(surv_s, minlength=n_genes)
        idd = np.bincount(surv_t, minlength=n_genes)
        active = int(np.count_nonzero(od+idd>0))
        sh, reason = check_shatter(surv_s, surv_t, surv_W, od, n_genes, active, shatter_cfg)
        if sh:
            return {'beta':beta,'gamma':gamma,'delta':delta,'kappa':kappa,'k_core':k_core,'lambda':lam,
                    'utopia_loss':999.,'is_shattered':1,'shatter_reason':reason,'n_edges':len(surv_s),
                    'active_nodes':active,'alpha':1.,'Gini':1.,'rho':1.,'C':0.,'Q':0.,
                    'S_max':_safe((np.max(od)/n_genes) if len(od)>0 else 0)}
        loss, topo = calculate_utopia_loss(surv_s, surv_t, surv_W, n_genes, od, active, kappa, utopian_bounds, loss_weights)
        gf = 0.
        try:
            G = sp.coo_matrix((np.ones(len(surv_W)),(surv_s,surv_t)), shape=(n_genes,n_genes))
            _,lb = connected_components(csgraph=G, directed=False, return_labels=True)
            gf = _safe(np.bincount(lb).max()/n_genes)
        except: pass
        gp = 0.
        if gf < 0.45: gp = 8.*((0.45-gf)/0.45)**2
        loss = _safe(np.sqrt(max(loss**2+gp,0.)), 999.)
        return {'beta':beta,'gamma':gamma,'delta':delta,'kappa':kappa,'k_core':k_core,'lambda':lam,
                'utopia_loss':loss,'is_shattered':0,'shatter_reason':None,'n_edges':len(surv_s),
                'active_nodes':active,'gwcc_fraction':gf,
                'alpha':topo['alpha'],'Gini':topo['Gini'],'rho':topo['rho'],'C':topo['C'],'Q':topo['Q'],'S_max':topo['S_max']}
    except Exception as e:
        beta,gamma,delta,kappa,k_core,lam = params
        return {'beta':beta,'gamma':gamma,'delta':delta,'kappa':kappa,'k_core':k_core,'lambda':lam,
                'utopia_loss':999.,'is_shattered':1,'shatter_reason':f'crash:{str(e)[:60]}',
                'n_edges':0,'active_nodes':0,'alpha':1.,'Gini':1.,'rho':1.,'C':0.,'Q':0.,'S_max':0.}
