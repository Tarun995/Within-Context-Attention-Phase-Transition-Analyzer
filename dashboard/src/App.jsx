import React, { useState, useMemo, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import {
  Activity, FlaskConical, Terminal, Upload, AlertTriangle, CheckCircle2,
  XCircle, FileJson, ChevronRight, Beaker, Pin, GitBranch,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Real data, transcribed from README.md / docs/FINDINGS.md
// ---------------------------------------------------------------------------

const HEADLINE_STATS = [
  { metric: "post_plateau_var", label: "Post-plateau variance", nSolved: 17, nFailed: 12, meanSolved: 0.0371, meanFailed: 0.0194, p: 0.042, r: -0.549, sig: true, direction: "Solved > failed" },
  { metric: "plateau_onset_fraction", label: "Plateau onset fraction", nSolved: 17, nFailed: 13, meanSolved: 0.4062, meanFailed: 0.4738, p: 1.0, r: 0.186, sig: false, direction: "No effect" },
  { metric: "entropy_rise_rate", label: "Entropy rise rate", nSolved: 17, nFailed: 13, meanSolved: 0.0096, meanFailed: 0.0106, p: 1.0, r: 0.14, sig: false, direction: "No effect" },
];

// Phase P1 — causal activation-patching pilots (real results, from
// results/patch_manifest.csv and results/patch_manifest_range.csv).
const P1_PILOTS = [
  {
    id: "final-token",
    label: "Pilot 1 — final token only",
    scope: "Patched only the attention-layer output at the last prompt position, GPT-2 small layer -1.",
    pairs: 10,
    shifts: 0,
    detail: "0/10 pairs showed any shift in next-token correctness, in either direction.",
  },
  {
    id: "post-plateau-range",
    label: "Pilot 2 — full post-plateau span",
    scope: "Patched every position from each recipient's own plateau-onset position through the end of the sequence (24\u2013149 positions per pair).",
    pairs: 10,
    shifts: 0,
    detail: "0/10 pairs showed any shift \u2014 even with most of the sequence's post-onset attention output replaced.",
  },
];

const PHASE0_TABLE = [
  { task: "copy_60", tokens: 242, solved: false, inflection: 68, growth: "4.8%" },
  { task: "copy_40", tokens: 162, solved: false, inflection: 68, growth: "Flat" },
  { task: "mod_arith_50", tokens: 504, solved: true, inflection: 137, growth: "29.9%" },
  { task: "transduction_40", tokens: 251, solved: false, inflection: 67, growth: "Flat" },
];

const JOURNEY = [
  {
    phase: "Phase 0", tag: "retracted", date: "Doctoral Symposium, Apr 2026",
    claim: "Solved tasks show a later, more prolonged attention reorganization than failed tasks — evidence of a qualitative internal transition.",
    correction: "Confounded. The one solved task (mod_arith_50) was also the only one at 504 tokens; every failed task was 162–251 tokens. The later inflection point is exactly what raw prompt length predicts, independent of solving. Also n=1 for the solved condition — underpowered on top of confounded.",
  },
  {
    phase: "Phase 0.5", tag: "fix", date: "Deconfounding",
    claim: "Build length-matched task variants so token count stops acting as a proxy for solved/failed status.",
    correction: "copy_matched_504 and transduction_matched_504 were constructed to match modular arithmetic's 504-token length. These became part of the Phase C1 task set.",
  },
  {
    phase: "Bug #1", tag: "bug", date: "Answer-matching scorer",
    claim: "The scorer that judges whether generated output counts as \u201csolved\u201d was assumed correct.",
    correction: "Found and fixed a scoring bug — bare integer answers, single-token answers, and multi-token answers were mishandled. Regression tests added in test_answer_matching.py.",
  },
  {
    phase: "Bug #2", tag: "bug", date: "Changepoint detector",
    claim: "A changepoint detector finds discrete transitions in the entropy curve.",
    correction: "Validated only on synthetic curves that didn't resemble real attention-entropy curves. Abandoned as the primary metric (kept in the codebase, documented as unreliable) in favor of plateau-based metrics: plateau_onset_fraction, post_plateau_var, has_plateau.",
  },
  {
    phase: "Layer sweep", tag: "retracted", date: "12 layers \u00d7 2 metrics",
    claim: "An apparent significant separation at layer 10.",
    correction: "Did not survive Bonferroni correction across the 24 comparisons \u2014 a textbook multiple-comparisons false positive. This is why every Phase C1 run applies Bonferroni correction by default.",
  },
  {
    phase: "Bug #3", tag: "bug", date: "Sign error in stats.py",
    claim: "post_plateau_var separates solved from failed (p_corrected = 0.042) \u2014 direction printed as \u201csolved < failed.\u201d",
    correction: "mean(solved)=0.0371 > mean(failed)=0.0194 \u2014 the printed direction was backwards. Root cause: r = 1 \u2212 2U/(n\u2081n\u2082) produces a negative r when the first group is larger, and the direction branch had the sign swapped. Fixed, verified against the full 55-test suite, re-run reproduced identical statistics with the corrected label.",
  },
  {
    phase: "Final (corrected)", tag: "verified", date: "Phase C1, current",
    claim: "post_plateau_var: solved > failed, p_corrected = 0.042, r = \u22120.549 (medium\u2013large effect).",
    correction: "Solved tasks keep attention more dynamic and oscillatory after the plateau \u2014 not quieter or more \u201clocked in.\u201d plateau_onset_fraction and entropy_rise_rate show no effect, reported rather than dropped.",
  },
  {
    phase: "Phase P1", tag: "tested", date: "Causal activation patching",
    claim: "Does the post_plateau_var correlation reflect a cause of solving, or just a side effect of it?",
    correction: "Two pre-registered patching pilots (final token only, then the full post-plateau span) both found 0/10 pairs showed any output shift when the donor's attention-layer output was spliced in \u2014 even patching nearly the whole post-plateau span. The hook mechanism was independently verified before trusting either result. Honest reading: at GPT-2 small's last layer, this is a correlate of solving, not a cause of it \u2014 scoped to that layer and component; earlier layers, other components, and full-generation effects remain untested.",
  },
];

// Example / synthetic dataset — illustrates the results.json upload format.
// This is NOT real experiment output (the repo's own historical_results JSON
// files are empty placeholders). It's here purely to demo the panel below.
const EXAMPLE_RESULTS = {
  config: { model_name: "gpt2", mode: "phase_c1", base_seed: 42, n_instances: 5 },
  results: [
    { task_id: "easy_mod_arith_seed42000", solved: true, actual_tokens: 118, plateau_onset_fraction: 0.38, post_plateau_var: 0.041, envelope_growth_pct: 22.1 },
    { task_id: "easy_mod_arith_seed42001", solved: true, actual_tokens: 121, plateau_onset_fraction: 0.44, post_plateau_var: 0.035, envelope_growth_pct: 18.7 },
    { task_id: "lookup_seed42000", solved: true, actual_tokens: 96, plateau_onset_fraction: 0.41, post_plateau_var: 0.039, envelope_growth_pct: 20.4 },
    { task_id: "lookup_seed42003", solved: true, actual_tokens: 102, plateau_onset_fraction: 0.36, post_plateau_var: 0.047, envelope_growth_pct: 25.6 },
    { task_id: "mod_arith_m10_d1_inst0", solved: true, actual_tokens: 504, plateau_onset_fraction: 0.27, post_plateau_var: 0.052, envelope_growth_pct: 29.9 },
    { task_id: "mod_arith_m10_d1_inst4", solved: true, actual_tokens: 504, plateau_onset_fraction: 0.31, post_plateau_var: 0.033, envelope_growth_pct: 24.3 },
    { task_id: "mod_arith_m10_d1_inst1", solved: false, actual_tokens: 504, plateau_onset_fraction: 0.49, post_plateau_var: 0.021, envelope_growth_pct: 6.1 },
    { task_id: "sorting_seed42000", solved: false, actual_tokens: 88, plateau_onset_fraction: 0.52, post_plateau_var: 0.017, envelope_growth_pct: 3.2 },
    { task_id: "copy_matched_504_inst2", solved: false, actual_tokens: 504, plateau_onset_fraction: 0.47, post_plateau_var: 0.019, envelope_growth_pct: 4.8 },
    { task_id: "transduction_matched_504_inst1", solved: false, actual_tokens: 504, plateau_onset_fraction: 0.5, post_plateau_var: 0.015, envelope_growth_pct: 2.0 },
  ],
  test_results: HEADLINE_STATS.map((h) => ({
    metric: h.metric, layer: -1, n_solved: h.nSolved, n_failed: h.nFailed,
    solved_mean: h.meanSolved, failed_mean: h.meanFailed, p_corrected: h.p,
    effect_size_r: h.r, significant: h.sig, direction: h.direction,
  })),
};

const CLI_MODES = ["phase_c1", "single", "layer_sweep"];

// ---------------------------------------------------------------------------

function Styles() {
  return (
    <style>{`
      .apta { --bg:#12141a; --panel:#1a1d25; --panel-2:#20242e; --border:#2a2f3b;
        --ink:#e9e6dd; --ink-dim:#8b93a3; --entropy:#5ec3e0; --induction:#f0a94a;
        --verified:#7dd39a; --retract:#e2665b; --bug:#e0b34a;
        --font-display:'Space Grotesk',system-ui,sans-serif;
        --font-body:'Inter',system-ui,sans-serif;
        --font-serif:'Source Serif 4','Georgia',serif;
        --font-mono:'JetBrains Mono','SFMono-Regular',monospace;
        background:var(--bg); color:var(--ink); font-family:var(--font-body);
        min-height:100vh; width:100%; box-sizing:border-box; }
      .apta *{ box-sizing:border-box; }
      .apta ::selection{ background:var(--induction); color:#12141a; }

      .apta-shell{ max-width:1180px; margin:0 auto; padding:36px 24px 80px; }

      .apta-header{ display:flex; flex-direction:column; gap:10px; margin-bottom:28px; border-bottom:1px solid var(--border); padding-bottom:24px; }
      .apta-eyebrow{ font-family:var(--font-mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--induction); display:flex; align-items:center; gap:8px; }
      .apta-eyebrow .dot{ width:6px; height:6px; border-radius:50%; background:var(--verified); box-shadow:0 0 8px var(--verified); animation:blink 2.4s ease-in-out infinite; }
      @keyframes blink{ 0%,100%{opacity:1;} 50%{opacity:.35;} }
      .apta-title{ font-family:var(--font-display); font-weight:600; font-size:clamp(26px,3.4vw,38px); line-height:1.08; margin:0; letter-spacing:-.01em; }
      .apta-sub{ font-family:var(--font-body); color:var(--ink-dim); font-size:14.5px; max-width:640px; line-height:1.55; margin:0; }
      .apta-repo{ font-family:var(--font-mono); font-size:12px; color:var(--ink-dim); }
      .apta-repo a{ color:var(--entropy); text-decoration:none; border-bottom:1px solid transparent; }
      .apta-repo a:hover{ border-bottom-color:var(--entropy); }

      .apta-tabs{ display:flex; gap:6px; margin:24px 0 28px; background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:5px; width:fit-content; }
      .apta-tab{ font-family:var(--font-mono); font-size:12.5px; letter-spacing:.03em; display:flex; align-items:center; gap:7px; padding:9px 15px; border-radius:7px; border:none; background:transparent; color:var(--ink-dim); cursor:pointer; transition:.15s; }
      .apta-tab:hover{ color:var(--ink); }
      .apta-tab.active{ background:var(--panel-2); color:var(--ink); box-shadow:inset 0 0 0 1px var(--border); }
      .apta-tab .chan{ font-size:10px; color:var(--induction); }
      .apta-tab:focus-visible{ outline:2px solid var(--entropy); outline-offset:2px; }

      .apta-card{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:22px; }
      .apta-grid{ display:grid; gap:16px; }

      /* Hero / scope trace */
      .apta-hero{ display:grid; grid-template-columns:1.1fr 1fr; gap:24px; margin-bottom:26px; }
      @media (max-width:860px){ .apta-hero{ grid-template-columns:1fr; } }
      .apta-hero-stat{ font-family:var(--font-mono); font-size:44px; font-weight:600; color:var(--verified); line-height:1; margin:10px 0 6px; }
      .apta-hero h2{ font-family:var(--font-display); font-size:20px; margin:0 0 12px; line-height:1.3; font-weight:600; }
      .apta-hero p{ color:var(--ink-dim); font-size:13.5px; line-height:1.6; margin:0 0 14px; }
      .apta-pill{ display:inline-flex; align-items:center; gap:6px; font-family:var(--font-mono); font-size:11px; padding:4px 10px; border-radius:100px; border:1px solid var(--border); color:var(--ink-dim); }
      .scope{ background:#0d0f14; border:1px solid var(--border); border-radius:10px; padding:14px; }
      .scope svg{ display:block; width:100%; height:auto; }

      table.apta-table{ width:100%; border-collapse:collapse; font-size:13px; }
      table.apta-table th{ text-align:left; font-family:var(--font-mono); font-size:10.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-dim); font-weight:500; padding:8px 10px; border-bottom:1px solid var(--border); }
      table.apta-table td{ padding:9px 10px; border-bottom:1px solid var(--border); font-family:var(--font-mono); font-size:12.5px; }
      table.apta-table tr:last-child td{ border-bottom:none; }
      .num{ text-align:right; font-variant-numeric:tabular-nums; }
      .apta-tag{ font-family:var(--font-mono); font-size:10.5px; padding:2px 7px; border-radius:5px; display:inline-block; }
      .tag-sig{ background:rgba(125,211,154,.14); color:var(--verified); }
      .tag-nosig{ background:rgba(139,147,163,.14); color:var(--ink-dim); }

      .apta-section-title{ font-family:var(--font-display); font-size:16px; font-weight:600; margin:0 0 4px; display:flex; align-items:center; gap:8px;}
      .apta-section-sub{ color:var(--ink-dim); font-size:12.5px; margin:0 0 14px; }

      /* Causal / P1 panel */
      .causal-pilot{ background:var(--panel-2); border:1px solid var(--border); border-radius:9px; padding:14px 16px; margin-bottom:10px; }
      .causal-pilot:last-child{ margin-bottom:0; }
      .causal-pilot-head{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
      .causal-pilot-label{ font-family:var(--font-mono); font-size:12.5px; color:var(--ink); font-weight:600; }
      .causal-pilot-ratio{ font-family:var(--font-mono); font-size:13px; color:var(--ink-dim); }
      .causal-pilot-scope{ font-size:12.5px; color:var(--ink-dim); line-height:1.55; margin-bottom:6px; }
      .causal-pilot-detail{ font-size:12.5px; color:var(--ink); line-height:1.55; border-left:2px solid var(--border); padding-left:10px; }

      /* Notebook / journey */
      .ledger{ display:flex; flex-direction:column; gap:0; }
      .ledger-entry{ display:grid; grid-template-columns:120px 1fr; gap:20px; padding:22px 0; border-bottom:1px dashed var(--border); }
      .ledger-entry:last-child{ border-bottom:none; }
      @media (max-width:700px){ .ledger-entry{ grid-template-columns:1fr; gap:8px; } }
      .ledger-meta{ display:flex; flex-direction:column; gap:8px; }
      .ledger-phase{ font-family:var(--font-mono); font-size:12px; font-weight:600; color:var(--ink); }
      .ledger-date{ font-family:var(--font-mono); font-size:10.5px; color:var(--ink-dim); }
      .stamp{ font-family:var(--font-mono); font-size:9.5px; letter-spacing:.08em; text-transform:uppercase; padding:3px 7px; border-radius:4px; width:fit-content; border:1px solid; }
      .stamp-retracted{ color:var(--retract); border-color:var(--retract); }
      .stamp-bug{ color:var(--bug); border-color:var(--bug); }
      .stamp-fix{ color:var(--entropy); border-color:var(--entropy); }
      .stamp-verified{ color:var(--verified); border-color:var(--verified); }
      .stamp-tested{ color:var(--ink-dim); border-color:var(--ink-dim); }
      .claim{ font-family:var(--font-serif); font-size:15px; line-height:1.6; color:#c9c5b8; }
      .claim.struck{ text-decoration:line-through; text-decoration-color:var(--retract); text-decoration-thickness:1.5px; opacity:.72; }
      .correction{ margin-top:10px; padding-left:14px; border-left:2px solid var(--verified); font-size:13px; line-height:1.6; color:var(--ink-dim); }
      .correction.isbug{ border-left-color:var(--bug); }
      .correction.istested{ border-left-color:var(--ink-dim); }
      .correction-label{ font-family:var(--font-mono); font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:var(--verified); display:block; margin-bottom:4px; }
      .correction.isbug .correction-label{ color:var(--bug); }
      .correction.istested .correction-label{ color:var(--ink-dim); }

      /* Run tab */
      .field{ display:flex; flex-direction:column; gap:6px; }
      .field label{ font-family:var(--font-mono); font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-dim); }
      .field input, .field select{ background:var(--panel-2); border:1px solid var(--border); color:var(--ink); font-family:var(--font-mono); font-size:12.5px; padding:8px 10px; border-radius:7px; }
      .field input:focus, .field select:focus{ outline:2px solid var(--entropy); outline-offset:1px; }
      .form-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:16px; }
      .cmdline{ background:#0d0f14; border:1px solid var(--border); border-radius:8px; padding:14px 16px; font-family:var(--font-mono); font-size:12.5px; color:var(--verified); overflow-x:auto; white-space:pre; }
      .cmdline .flag{ color:var(--induction); }
      .dropzone{ border:1.5px dashed var(--border); border-radius:10px; padding:26px; text-align:center; color:var(--ink-dim); font-size:13px; cursor:pointer; transition:.15s; }
      .dropzone:hover{ border-color:var(--entropy); color:var(--ink); }
      .btn{ font-family:var(--font-mono); font-size:12px; padding:8px 14px; border-radius:7px; border:1px solid var(--border); background:var(--panel-2); color:var(--ink); cursor:pointer; display:inline-flex; align-items:center; gap:6px; transition:.15s; }
      .btn:hover{ border-color:var(--entropy); color:var(--entropy); }
      .btn:focus-visible{ outline:2px solid var(--entropy); outline-offset:2px; }
      .banner{ display:flex; gap:10px; align-items:flex-start; background:rgba(240,169,74,.08); border:1px solid rgba(240,169,74,.3); border-radius:8px; padding:10px 12px; font-size:12px; color:#e0c088; margin-bottom:16px; line-height:1.5; }
      .stat-cards{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:18px; }
      .stat-card{ background:var(--panel-2); border:1px solid var(--border); border-radius:9px; padding:14px; }
      .stat-card .m{ font-family:var(--font-mono); font-size:11px; color:var(--ink-dim); margin-bottom:8px; }
      .stat-card .p{ font-family:var(--font-mono); font-size:22px; font-weight:600; }
      .legend-dot{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }

      .apta-footer{ margin-top:40px; padding-top:20px; border-top:1px solid var(--border); font-family:var(--font-mono); font-size:11px; color:var(--ink-dim); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }
    `}</style>
  );
}

function ScopeHero() {
  return (
    <div className="scope">
      <svg viewBox="0 0 520 160" role="img" aria-label="Stylized entropy and induction envelope traces">
        <defs>
          <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
            <path d="M 26 0 L 0 0 0 26" fill="none" stroke="#232733" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="520" height="160" fill="url(#grid)" />
        <line x1="0" y1="80" x2="520" y2="80" stroke="#2a2f3b" strokeDasharray="4 4" />
        <path
          d="M0,95 C60,92 90,90 130,88 C200,84 230,60 260,50 C300,36 330,30 360,26 C420,20 470,18 520,17"
          fill="none" stroke="var(--entropy)" strokeWidth="2"
          strokeDasharray="600" strokeDashoffset="600">
          <animate attributeName="stroke-dashoffset" from="600" to="0" dur="2.4s" fill="freeze" />
        </path>
        <path
          d="M0,120 C60,119 100,118 130,117 C180,116 220,108 260,90 C300,70 330,60 360,48 C420,30 470,24 520,20"
          fill="none" stroke="var(--induction)" strokeWidth="2"
          strokeDasharray="600" strokeDashoffset="600">
          <animate attributeName="stroke-dashoffset" from="600" to="0" dur="2.4s" begin="0.2s" fill="freeze" />
        </path>
        <line x1="255" y1="10" x2="255" y2="150" stroke="var(--verified)" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
        <text x="260" y="22" fill="var(--verified)" fontSize="9" fontFamily="JetBrains Mono, monospace">plateau onset</text>
        <text x="8" y="14" fill="var(--entropy)" fontSize="9" fontFamily="JetBrains Mono, monospace">entropy</text>
        <text x="8" y="140" fill="var(--induction)" fontSize="9" fontFamily="JetBrains Mono, monospace">induction envelope</text>
      </svg>
    </div>
  );
}

function CausalPanel() {
  return (
    <div className="apta-card">
      <p className="apta-section-title"><GitBranch size={16} color="var(--ink-dim)" /> Is it causal? &mdash; Phase P1</p>
      <p className="apta-section-sub">
        C1 shows post_plateau_var correlates with solving. Phase P1 tested whether it's causal,
        by capturing the attention-layer output from a solved task's forward pass and splicing
        it into a failed task's forward pass (and vice versa), then checking whether the
        model's answer shifted. Two pre-registered pilots, both null.
      </p>
      {P1_PILOTS.map((pilot) => (
        <div className="causal-pilot" key={pilot.id}>
          <div className="causal-pilot-head">
            <span className="causal-pilot-label">{pilot.label}</span>
            <span className="causal-pilot-ratio">{pilot.shifts}/{pilot.pairs} pairs shifted</span>
          </div>
          <div className="causal-pilot-scope">{pilot.scope}</div>
          <div className="causal-pilot-detail">{pilot.detail}</div>
        </div>
      ))}
      <p style={{ fontSize: 12.5, color: "var(--ink-dim)", lineHeight: 1.6, marginTop: 12, marginBottom: 0 }}>
        <strong style={{ color: "var(--ink)" }}>Honest reading:</strong> at GPT-2 small's last
        layer, task correctness is robust to this component being heavily altered &mdash; real
        evidence post_plateau_var is a correlate of solving rather than a cause of it, at this
        layer and via this component. Earlier layers, other components (MLP, individual heads),
        and full-generation effects remain untested &mdash; scope stated explicitly rather than
        implied. Full method in the Notebook tab and <code style={{ fontFamily: "var(--font-mono)" }}>docs/FINDINGS.md</code>.
      </p>
    </div>
  );
}

function Overview() {
  return (
    <div className="apta-grid">
      <div className="apta-hero">
        <div className="apta-card">
          <span className="apta-pill"><Pin size={11} /> Headline result &middot; GPT-2 small &middot; Phase C1</span>
          <div className="apta-hero-stat">p = 0.042</div>
          <h2>Solved tasks keep attention more dynamic after the plateau &mdash; not quieter.</h2>
          <p>
            Across 30 task instances (5 seeds &times; 6 task types), <code style={{ color: "var(--induction)", fontFamily: "var(--font-mono)" }}>post_plateau_var</code> is
            the one of three Bonferroni-corrected metrics that separates solved from failed: solved
            instances show higher post-plateau entropy variance (0.0371 vs 0.0194), a medium&ndash;large
            effect (r = &minus;0.549). Read plainly: the model doesn&rsquo;t settle into a calmer state when
            it succeeds &mdash; it stays more oscillatory, plausibly a signature of sustained pattern-matching.
            Causal patching (below) found this is a correlate, not (yet demonstrated to be) a cause.
          </p>
          <span className="apta-pill">n(solved)=17 &middot; n(failed)=12</span>
        </div>
        <ScopeHero />
      </div>

      <CausalPanel />

      <div className="apta-card">
        <p className="apta-section-title"><Activity size={16} color="var(--entropy)" /> Tested metrics, Bonferroni-corrected (k=3)</p>
        <p className="apta-section-sub">All three metrics were tested; only one survives correction &mdash; the other two are reported honestly, not dropped.</p>
        <table className="apta-table">
          <thead>
            <tr>
              <th>Metric</th><th className="num">n solved</th><th className="num">n failed</th>
              <th className="num">mean solved</th><th className="num">mean failed</th>
              <th className="num">p (corr.)</th><th className="num">effect r</th><th>Result</th>
            </tr>
          </thead>
          <tbody>
            {HEADLINE_STATS.map((row) => (
              <tr key={row.metric}>
                <td style={{ color: "var(--ink)" }}>{row.metric}</td>
                <td className="num">{row.nSolved}</td>
                <td className="num">{row.nFailed}</td>
                <td className="num">{row.meanSolved.toFixed(4)}</td>
                <td className="num">{row.meanFailed.toFixed(4)}</td>
                <td className="num" style={{ color: row.sig ? "var(--verified)" : "var(--ink-dim)" }}>{row.p.toFixed(3)}</td>
                <td className="num">{row.r.toFixed(3)}</td>
                <td><span className={`apta-tag ${row.sig ? "tag-sig" : "tag-nosig"}`}>{row.direction}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="apta-card" style={{ borderColor: "rgba(226,102,91,.35)" }}>
        <p className="apta-section-title"><AlertTriangle size={16} color="var(--retract)" /> Superseded &mdash; Phase 0 (confounded)</p>
        <p className="apta-section-sub">The project's original finding. Kept here for traceability &mdash; see the Notebook tab for why it doesn't hold.</p>
        <table className="apta-table">
          <thead>
            <tr><th>Task</th><th className="num">Tokens</th><th>Solved</th><th className="num">Inflection</th><th className="num">Envelope growth</th></tr>
          </thead>
          <tbody>
            {PHASE0_TABLE.map((r) => (
              <tr key={r.task} style={{ opacity: 0.75 }}>
                <td style={{ color: "var(--ink)" }}>{r.task}</td>
                <td className="num">{r.tokens}</td>
                <td>{r.solved ? <CheckCircle2 size={13} color="var(--verified)" /> : <XCircle size={13} color="var(--ink-dim)" />}</td>
                <td className="num">{r.inflection}</td>
                <td className="num">{r.growth}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="apta-card">
        <p className="apta-section-title"><Beaker size={16} color="var(--induction)" /> Limitations</p>
        <ul style={{ margin: 0, paddingLeft: 18, color: "var(--ink-dim)", fontSize: 13, lineHeight: 1.8 }}>
          <li>Single model tested end-to-end: GPT-2 small (117M). CLI supports any HF causal LM; larger models not yet reported.</li>
          <li>CPU-only run shown; not benchmarked for GPU throughput.</li>
          <li>One base seed (42) with 5 derived instances per task type.</li>
          <li><code style={{ fontFamily: "var(--font-mono)" }}>mod_arith_m10_d1</code> appears in both solved and failed groups across seeds &mdash; the separation is partly at the instance level, not purely between task types.</li>
          <li>Causal patching (Phase P1) tested only the last layer's attention-layer output, scored via next-token prediction rather than full generation &mdash; earlier layers, other components, and full-generation effects remain untested.</li>
        </ul>
      </div>
    </div>
  );
}

function Journey() {
  const stampClass = { retracted: "stamp-retracted", bug: "stamp-bug", fix: "stamp-fix", verified: "stamp-verified", tested: "stamp-tested" };
  const stampLabel = { retracted: "Retracted", bug: "Bug found", fix: "Fix", verified: "Verified", tested: "Tested (null)" };
  return (
    <div className="apta-card">
      <p className="apta-section-title"><FlaskConical size={16} color="var(--entropy)" /> Research notebook</p>
      <p className="apta-section-sub">Every claim in the README traced back to the experiment, bug, or fix that produced it &mdash; including the parts that didn't work. Struck-through lines are retracted claims; the margin note is what was actually found.</p>
      <div className="ledger">
        {JOURNEY.map((entry) => (
          <div className="ledger-entry" key={entry.phase}>
            <div className="ledger-meta">
              <span className="ledger-phase">{entry.phase}</span>
              <span className="ledger-date">{entry.date}</span>
              <span className={`stamp ${stampClass[entry.tag]}`}>{stampLabel[entry.tag]}</span>
            </div>
            <div>
              <div className={`claim ${entry.tag === "retracted" ? "struck" : ""}`}>{entry.claim}</div>
              <div className={`correction ${entry.tag === "bug" ? "isbug" : ""} ${entry.tag === "tested" ? "istested" : ""}`}>
                <span className="correction-label">{entry.tag === "verified" ? "Why it holds" : entry.tag === "tested" ? "What was found" : "What was actually found"}</span>
                {entry.correction}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RunTab() {
  const [cfg, setCfg] = useState({ mode: "phase_c1", model: "gpt2", seed: 42, instances: 5, layer: -1, outputDir: "results" });
  const [parsed, setParsed] = useState(null);
  const [error, setError] = useState("");
  const [rawText, setRawText] = useState("");

  const cmd = useMemo(() => {
    const parts = [
      "attn-phase run",
      `--mode ${cfg.mode}`,
      `--model ${cfg.model}`,
      `--seed ${cfg.seed}`,
    ];
    if (cfg.mode === "phase_c1") parts.push(`--instances ${cfg.instances}`);
    if (cfg.mode === "single") parts.push(`--layer ${cfg.layer}`);
    parts.push(`--output-dir ${cfg.outputDir}`);
    return parts;
  }, [cfg]);

  const handleFile = useCallback((file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        setParsed(json);
        setError("");
      } catch (err) {
        setError("Couldn't parse that as JSON: " + err.message);
      }
    };
    reader.readAsText(file);
  }, []);

  const loadExample = () => { setParsed(EXAMPLE_RESULTS); setError(""); };

  const parsePasted = () => {
    try {
      const json = JSON.parse(rawText);
      setParsed(json);
      setError("");
    } catch (err) {
      setError("Couldn't parse that as JSON: " + err.message);
    }
  };

  const results = parsed?.results || [];
  const testResults = parsed?.test_results || [];
  const isExample = parsed === EXAMPLE_RESULTS;

  return (
    <div className="apta-grid">
      <div className="apta-card">
        <p className="apta-section-title"><Terminal size={16} color="var(--entropy)" /> Configure a run</p>
        <p className="apta-section-sub">
          GPT-2 forward passes need a local Python environment (this dashboard runs in your browser and can't execute them).
          Configure below, run it on your machine, then bring the results back in.
        </p>
        <div className="form-grid">
          <div className="field">
            <label>Mode</label>
            <select value={cfg.mode} onChange={(e) => setCfg({ ...cfg, mode: e.target.value })}>
              {CLI_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Model</label>
            <input value={cfg.model} onChange={(e) => setCfg({ ...cfg, model: e.target.value })} />
          </div>
          <div className="field">
            <label>Seed</label>
            <input type="number" value={cfg.seed} onChange={(e) => setCfg({ ...cfg, seed: e.target.value })} />
          </div>
          {cfg.mode === "phase_c1" && (
            <div className="field">
              <label>Instances / type</label>
              <input type="number" value={cfg.instances} onChange={(e) => setCfg({ ...cfg, instances: e.target.value })} />
            </div>
          )}
          {cfg.mode === "single" && (
            <div className="field">
              <label>Layer (-1 = last)</label>
              <input type="number" value={cfg.layer} onChange={(e) => setCfg({ ...cfg, layer: e.target.value })} />
            </div>
          )}
          <div className="field">
            <label>Output dir</label>
            <input value={cfg.outputDir} onChange={(e) => setCfg({ ...cfg, outputDir: e.target.value })} />
          </div>
        </div>
        <div className="cmdline">
          {cmd.map((p, i) => i === 0 ? p : <span key={i}> <span className="flag">{p.split(" ")[0]}</span> {p.split(" ").slice(1).join(" ")}</span>)}
        </div>
      </div>

      <div className="apta-card">
        <p className="apta-section-title"><Upload size={16} color="var(--induction)" /> Bring in results.json</p>
        <p className="apta-section-sub">Drop the file <code style={{ fontFamily: "var(--font-mono)" }}>{"{mode}_{model}_seed{n}_results.json"}</code> that a run saves to your output dir, or paste its contents.</p>

        <label
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer.files?.[0]); }}
        >
          <input type="file" accept="application/json" style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files?.[0])} />
          <FileJson size={20} style={{ marginBottom: 6 }} />
          <div>Drop a results.json here, or click to choose a file</div>
        </label>

        <div style={{ display: "flex", gap: 10, margin: "14px 0" }}>
          <button className="btn" onClick={loadExample}><ChevronRight size={13} /> Load example data</button>
          {parsed && <button className="btn" onClick={() => { setParsed(null); setRawText(""); }}>Clear</button>}
        </div>

        <details>
          <summary style={{ cursor: "pointer", fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--ink-dim)" }}>or paste raw JSON</summary>
          <textarea
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
            placeholder='{"config": {...}, "results": [...], "test_results": [...]}'
            style={{ width: "100%", minHeight: 90, marginTop: 8, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 7, color: "var(--ink)", fontFamily: "var(--font-mono)", fontSize: 11.5, padding: 10 }}
          />
          <button className="btn" style={{ marginTop: 8 }} onClick={parsePasted}>Parse pasted JSON</button>
        </details>

        {error && <div className="banner"><AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} />{error}</div>}
      </div>

      {parsed && (
        <div className="apta-card">
          {isExample && (
            <div className="banner">
              <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
              This is a synthetic example (10 illustrative task instances), not real experiment output &mdash;
              the repo doesn't publish raw per-instance JSON. It's here to show what the panel below does
              with a real <code style={{ fontFamily: "var(--font-mono)" }}>results.json</code> from your own run.
            </div>
          )}

          {testResults.length > 0 && (
            <>
              <p className="apta-section-title">Statistical tests in this file</p>
              <div className="stat-cards">
                {testResults.map((t) => (
                  <div className="stat-card" key={t.metric}>
                    <div className="m">{t.metric}</div>
                    <div className="p" style={{ color: t.significant ? "var(--verified)" : "var(--ink-dim)" }}>
                      p={Number(t.p_corrected).toFixed(3)}
                    </div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink-dim)", marginTop: 4 }}>
                      {t.direction} &middot; r={Number(t.effect_size_r).toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>

              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={testResults} margin={{ top: 6, right: 12, left: 0, bottom: 6 }}>
                  <CartesianGrid stroke="#232733" vertical={false} />
                  <XAxis dataKey="metric" tick={{ fill: "#8b93a3", fontSize: 10, fontFamily: "JetBrains Mono, monospace" }} axisLine={{ stroke: "#2a2f3b" }} tickLine={false} />
                  <YAxis tick={{ fill: "#8b93a3", fontSize: 10, fontFamily: "JetBrains Mono, monospace" }} axisLine={{ stroke: "#2a2f3b" }} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#0d0f14", border: "1px solid #2a2f3b", fontFamily: "JetBrains Mono, monospace", fontSize: 11 }} />
                  <Bar dataKey="solved_mean" name="mean (solved)" fill="var(--entropy)" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="failed_mean" name="mean (failed)" fill="var(--induction)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}

          {results.length > 0 && (
            <>
              <p className="apta-section-title" style={{ marginTop: 22 }}>Per-task: onset fraction vs. post-plateau variance</p>
              <ResponsiveContainer width="100%" height={240}>
                <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 6 }}>
                  <CartesianGrid stroke="#232733" />
                  <XAxis type="number" dataKey="plateau_onset_fraction" name="onset fraction" tick={{ fill: "#8b93a3", fontSize: 10, fontFamily: "JetBrains Mono, monospace" }} axisLine={{ stroke: "#2a2f3b" }} />
                  <YAxis type="number" dataKey="post_plateau_var" name="post-plateau var" tick={{ fill: "#8b93a3", fontSize: 10, fontFamily: "JetBrains Mono, monospace" }} axisLine={{ stroke: "#2a2f3b" }} />
                  <ZAxis range={[70, 70]} />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#0d0f14", border: "1px solid #2a2f3b", fontFamily: "JetBrains Mono, monospace", fontSize: 11 }}
                    formatter={(v, n) => [Number(v).toFixed(4), n]}
                    labelFormatter={() => ""} />
                  <Scatter data={results.filter((r) => r.solved)} fill="var(--verified)" name="solved" />
                  <Scatter data={results.filter((r) => !r.solved)} fill="var(--retract)" name="failed" />
                </ScatterChart>
              </ResponsiveContainer>

              <p className="apta-section-title" style={{ marginTop: 22 }}>Task table</p>
              <table className="apta-table">
                <thead>
                  <tr><th>Task</th><th>Solved</th><th className="num">Tokens</th><th className="num">Onset frac.</th><th className="num">Post-plateau var</th><th className="num">Envelope growth</th></tr>
                </thead>
                <tbody>
                  {results.map((r) => (
                    <tr key={r.task_id}>
                      <td style={{ color: "var(--ink)" }}>{r.task_id}</td>
                      <td>{r.solved ? <CheckCircle2 size={13} color="var(--verified)" /> : <XCircle size={13} color="var(--retract)" />}</td>
                      <td className="num">{r.actual_tokens ?? "\u2014"}</td>
                      <td className="num">{r.plateau_onset_fraction?.toFixed(3) ?? "\u2014"}</td>
                      <td className="num">{r.post_plateau_var?.toFixed(4) ?? "\u2014"}</td>
                      <td className="num">{r.envelope_growth_pct != null ? `${r.envelope_growth_pct.toFixed(1)}%` : "\u2014"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function AttentionPhaseDashboard() {
  const [tab, setTab] = useState("overview");
  const tabs = [
    { id: "overview", label: "Finding", chan: "CH1", icon: Activity },
    { id: "journey", label: "Notebook", chan: "CH2", icon: FlaskConical },
    { id: "run", label: "Run", chan: "CH3", icon: Terminal },
  ];
  return (
    <div className="apta">
      <Styles />
      <div className="apta-shell">
        <div className="apta-header">
          <span className="apta-eyebrow"><span className="dot" /> Mechanistic interpretability &middot; GPT-2 small</span>
          <h1 className="apta-title">Within-Context Attention Phase Transition Analyzer</h1>
          <p className="apta-sub">
            A hypothesis-testing pipeline, not a visualization tool: synthetic tasks run through GPT-2, attention-derived
            metrics extracted per token position, solved vs. failed compared with a properly powered, multiple-comparisons-corrected test,
            then causally tested via activation patching.
          </p>
          <span className="apta-repo">
            <a href="https://github.com/Tarun995/Within-Context-Attention-Phase-Transition-Analyzer" target="_blank" rel="noreferrer">
              github.com/Tarun995/Within-Context-Attention-Phase-Transition-Analyzer
            </a>
          </span>
        </div>

        <div className="apta-tabs" role="tablist">
          {tabs.map((t) => (
            <button key={t.id} className={`apta-tab ${tab === t.id ? "active" : ""}`} role="tab"
              aria-selected={tab === t.id} onClick={() => setTab(t.id)}>
              <span className="chan">{t.chan}</span><t.icon size={13} />{t.label}
            </button>
          ))}
        </div>

        {tab === "overview" && <Overview />}
        {tab === "journey" && <Journey />}
        {tab === "run" && <RunTab />}

        <div className="apta-footer">
          <span>MIT licensed &middot; GPT-2 small (117M) &middot; single base seed (42)</span>
          <span>tests/ &middot; 60+ regression + patch-mechanism tests</span>
        </div>
      </div>
    </div>
  );
}