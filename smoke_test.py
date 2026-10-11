"""smoke_test.py — Validates all pure-Python micro-workers (no model weights needed)."""
import sys
sys.path.insert(0, r"E:\SLM LLM")

print("=== Smoke Test: All Pure-Python Workers ===\n")
errors = []

def check(n, label):
    print(f"[{n:>2}] {label}")

def fail(n, label, err):
    msg = f"[{n:>2}] FAIL  {label}: {err}"
    print(msg)
    errors.append(msg)

# 1. BaseWorker
try:
    from workers.base_worker import BaseWorker, WorkerResult
    check(1, "BaseWorker ................. import OK")
except Exception as e:
    fail(1, "BaseWorker", e)

# 2. HardwareWorker
try:
    from workers.hardware_worker import HardwareWorker
    hw = HardwareWorker()
    res = hw.run()
    assert res.success, res.error
    p = res.data
    check(2, f"HardwareWorker ............. tier={p.tier}  RAM={p.ram_total_mb/1024:.1f}GB  VRAM={p.total_vram_mb/1024:.1f}GB")
except Exception as e:
    fail(2, "HardwareWorker", e)
    p = None

# 3. CatalogWorker - list
try:
    from workers.catalog_worker import CatalogWorker
    cw = CatalogWorker()
    res = cw.run(action="list")
    assert res.success and len(res.data) > 0, res.error
    check(3, f"CatalogWorker list ......... {len(res.data)} models found")
except Exception as e:
    fail(3, "CatalogWorker.list", e)

# 4. CatalogWorker - get
model = None
try:
    res = cw.run(action="get", key="qwen2.5-3b")
    assert res.success, res.error
    model = res.data
    check(4, f"CatalogWorker get .......... {model.name}  ({model.params_billions}B)")
except Exception as e:
    fail(4, "CatalogWorker.get", e)

# 5. QuantWorker
try:
    from workers.quant_worker import QuantWorker
    qw = QuantWorker()
    qres = qw.run(model=model, usable_mem_mb=8192)
    assert qres.success, qres.error
    check(5, f"QuantWorker ................ bits={qres.data['quant_bits']}  label={qres.data['label'][:30]}")
except Exception as e:
    fail(5, "QuantWorker", e)

# 6. PlanWorker
plan = None
try:
    from workers.plan_worker import PlanWorker
    pw = PlanWorker()
    pres = pw.run(model=model, profile=p)
    assert pres.success, pres.error
    plan = pres.data
    check(6, f"PlanWorker ................. strategy={plan.strategy}  shards={len(plan.shards)}  quant={plan.quant_bits}bit")
except Exception as e:
    fail(6, "PlanWorker", e)

# 7. KVBudgetWorker
try:
    from workers.kv_budget_worker import KVBudgetWorker
    kv = KVBudgetWorker()
    kres = kv.run(model=model, profile=p, prompt_tokens=50, requested_max=512)
    assert kres.success, kres.error
    check(7, f"KVBudgetWorker ............. budget={kres.data['safe_context_tokens']}  max_new={kres.data['recommended_max_new']}")
except Exception as e:
    fail(7, "KVBudgetWorker", e)

# 8. TokenWorker
try:
    from workers.token_worker import TokenWorker
    tw = TokenWorker()
    tres = tw.run(prompt="  Hello!!!   world???   \n\n\n  how are you today?  ")
    assert tres.success, tres.error
    check(8, f"TokenWorker ................ saved={tres.data['tokens_saved']} tok  ratio={tres.data['compression_ratio']*100:.1f}%")
except Exception as e:
    fail(8, "TokenWorker", e)

# 9. RepetitionWorker - loop and no-loop
try:
    from workers.repetition_worker import RepetitionWorker
    rw = RepetitionWorker()
    rres  = rw.run(words=["hello","world","hello","world","hello","world"], ngram_size=2, threshold=3)
    rres2 = rw.run(words=["the","quick","brown","fox","jumps"], ngram_size=2, threshold=3)
    assert rres.success and rres.data["is_loop"],  "Should detect loop"
    assert rres2.success and not rres2.data["is_loop"], "Should NOT detect loop"
    check(9, f"RepetitionWorker ........... loop_detected=True  clean_text=no_loop")
except Exception as e:
    fail(9, "RepetitionWorker", e)

# 10. TemplateWorker - all 5 families
try:
    from workers.template_worker import TemplateWorker
    tmplw = TemplateWorker()
    ok_fams = []
    for fam in ["llama", "mistral", "qwen", "phi", "gemma"]:
        r = tmplw.run(family=fam, user_prompt="Hello!", system_prompt="Be helpful.")
        if r.success:
            ok_fams.append(fam)
        else:
            fail(10, f"TemplateWorker [{fam}]", r.error)
    if len(ok_fams) == 5:
        check(10, f"TemplateWorker ............. families={','.join(ok_fams)}")
except Exception as e:
    fail(10, "TemplateWorker", e)

# 11. PrunerWorker
try:
    from workers.pruner_worker import PrunerWorker
    prw = PrunerWorker()
    msgs = [
        {"role": "system",    "content": "You are helpful."},
        {"role": "user",      "content": "Turn 1 text here"},
        {"role": "assistant", "content": "Reply 1 text here"},
        {"role": "user",      "content": "Turn 2 text here"},
        {"role": "assistant", "content": "Reply 2 text here"},
        {"role": "user",      "content": "Latest user turn"},
    ]
    prunres = prw.run(messages=msgs, max_budget_tokens=20)
    assert prunres.success, prunres.error
    check(11, f"PrunerWorker ............... original={prunres.data['original_count']}  pruned={prunres.data['pruned_count']}  dropped={prunres.data['turns_dropped']}")
except Exception as e:
    fail(11, "PrunerWorker", e)

# 12. DownloadWorker - cache check (no network needed for basic test)
try:
    from workers.download_worker import DownloadWorker
    dlw = DownloadWorker()
    dlres = dlw.run(action="check", hf_repo="Qwen/Qwen2.5-0.5B-Instruct")
    assert dlres.success, dlres.error
    check(12, f"DownloadWorker ............. is_cached={dlres.data['is_cached']}  size={dlres.data['size_gb']:.2f}GB")
except Exception as e:
    fail(12, "DownloadWorker", e)

# 13. Orchestrator - full pipeline (no model load)
try:
    from orchestrator import Orchestrator
    orch = Orchestrator()
    r1 = orch.scan_hardware();       assert r1.success, r1.error
    r2 = orch.select_model("phi-3.5-mini"); assert r2.success, r2.error
    r3 = orch.compute_plan();        assert r3.success, r3.error
    r4 = orch.get_kv_budget(prompt_tokens=10, requested_max=256); assert r4.success, r4.error
    ws = orch.worker_summary;        assert len(ws) == 12
    check(13, f"Orchestrator ............... workers={len(ws)}  plan={orch.plan.strategy}  feasible={orch.plan.is_feasible}")
except Exception as e:
    fail(13, "Orchestrator", e)

# 14. main.py import check
try:
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("main", r"E:\SLM LLM\main.py")
    mod  = importlib.util.module_from_spec(spec)
    # just check it imports cleanly (don't call main())
    spec.loader.exec_module(mod)
    assert hasattr(mod, "Orchestrator") or hasattr(mod, "run_wizard")
    check(14, "main.py import ............. OK")
except Exception as e:
    fail(14, "main.py import", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"=== {len(errors)} TEST(S) FAILED ===")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("=== ALL 14 TESTS PASSED ===")
