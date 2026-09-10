const $ = (id) => document.getElementById(id);
const DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
let lastStatus = {};
let selectedJob = "";
let windowsState = [];
let templates = [];
let lastInfra = {};
let selectedUser = "";

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

function fmtDur(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function fmtGb(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1024) return `${(v / 1024).toFixed(1)} TB`;
  return `${v.toFixed(v >= 10 ? 1 : 3)} GB`;
}

function fmtTokens(n) {
  const v = Number(n) || 0;
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
}

function owner(gpu) {
  const procs = gpu.compute_processes || [];
  if (!procs.length) return "—";
  return procs.map((p) => `${p.username || "?"} (${p.troiani_owned ? "troiani" : "other"})`).join(", ");
}

function gpuLabel(gpu) {
  return gpu.ref || `${String(gpu.node || "node").toLowerCase()}/gpu${gpu.index}`;
}

function smOf(gpu) {
  if (gpu.sm != null) return Number(gpu.sm);
  if (gpu.sm_util != null) return Number(gpu.sm_util);
  return Number(gpu.utilization) || 0;
}

function vramPct(gpu) {
  if (gpu.vram_pct != null) return Number(gpu.vram_pct);
  return 100 * (gpu.vram_used_gb || gpu.memory_used_gb || 0) / Math.max(gpu.vram_gb || gpu.memory_gb || 1, 1);
}

function bar(pct, cls) {
  const width = Math.max(0, Math.min(100, Number(pct) || 0));
  return `<div class="bar ${cls || ""}" title="${width.toFixed(0)}%"><span style="width:${width.toFixed(1)}%"></span><em>${width.toFixed(0)}%</em></div>`;
}

function pill(text, cls) {
  return `<span class="pill ${cls || ""}">${esc(text)}</span>`;
}

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((btn) => btn.classList.toggle("on", btn.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((panel) => panel.classList.toggle("on", panel.dataset.panel === name));
  if (name) window.location.hash = name;
  if (name === "infra") refreshInfra();
  if (name === "policy") renderPolicyCompare();
}

async function act(path, confirmFirst, message, method = "POST", body) {
  if (confirmFirst && !window.confirm(message || "Are you sure?")) return;
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch("/v1/" + path, opts);
  if (!res.ok) {
    const detail = await res.text();
    $("toast").textContent = `${res.status}: ${detail.slice(0, 220)}`;
    return null;
  }
  $("toast").textContent = "";
  await refresh();
  return true;
}

function normalizeWindow(raw) {
  if (typeof raw === "string") {
    const [start, end] = raw.split("-");
    return { name: "", start: (start || "").trim(), end: (end || "").trim(), days: [], enabled: true };
  }
  return {
    name: raw.name || "",
    start: raw.start || "20:00",
    end: raw.end || "08:00",
    days: Array.isArray(raw.days) ? raw.days : [],
    enabled: raw.enabled !== false,
  };
}

function renderWindowRows() {
  $("window-rows").innerHTML = windowsState.map((w, idx) => `
    <div class="window-row">
      <label>Name <input data-w="${idx}" data-k="name" value="${esc(w.name)}"></label>
      <label>Start <input data-w="${idx}" data-k="start" value="${esc(w.start)}" placeholder="20:00"></label>
      <label>End <input data-w="${idx}" data-k="end" value="${esc(w.end)}" placeholder="08:00"></label>
      <div class="days">${DAYS.map((d) => `
        <label><input type="checkbox" data-w="${idx}" data-day="${d}" ${w.days.includes(d) ? "checked" : ""}>${d}</label>
      `).join("")}</div>
      <button type="button" data-toggle="${idx}" class="${w.enabled ? "on" : ""}">${w.enabled ? "On" : "Off"}</button>
      <button type="button" data-del="${idx}">×</button>
    </div>
  `).join("") || `<p class="note">No windows — opportunistic at any hour, still yielding to researchers.</p>`;
}

function readWindowForm() {
  windowsState = windowsState.map((w, idx) => {
    const name = document.querySelector(`[data-w="${idx}"][data-k="name"]`);
    if (!name) return w;
    const days = DAYS.filter((d) => {
      const box = document.querySelector(`[data-w="${idx}"][data-day="${d}"]`);
      return box && box.checked;
    });
    return {
      name: name.value,
      start: document.querySelector(`[data-w="${idx}"][data-k="start"]`).value,
      end: document.querySelector(`[data-w="${idx}"][data-k="end"]`).value,
      days,
      enabled: Boolean(w.enabled),
    };
  });
}

function renderGpus(status) {
  const draining = new Set(status.draining || []);
  const gpus = status.gpus || [];
  $("gpu-strip").innerHTML = gpus.map((g) => `
    <div class="gpu-card">
      <div class="mono">${esc(gpuLabel(g))}</div>
      <div>${esc((g.name || g.model || "").slice(0, 28))} · ${(g.vram_gb || g.memory_gb || 0).toFixed(0)}G</div>
      <div>${pill(g.occupancy, "occ-" + String(g.occupancy || "").toLowerCase())}</div>
      <div class="metric"><span>SM</span>${bar(smOf(g))}</div>
      <div class="metric"><span>VRAM</span>${bar(vramPct(g), "mem")}</div>
      <div class="who">${esc(owner(g))}</div>
    </div>
  `).join("");
  const rows = gpus.map((g) => {
    const researcher = g.occupancy === "RESEARCHER" || (g.compute_processes || []).some((p) => !p.troiani_owned);
    const isDraining = draining.has(g.uuid) || g.occupancy === "DRAINING";
    let buttons = "";
    if (researcher) {
      buttons = `<button disabled title="Researcher present — we will not touch this GPU">Blocked</button>`;
    } else if (isDraining) {
      buttons = `<button onclick="act('gpus/${esc(g.uuid)}/undrain')">Undrain</button>`;
    } else {
      buttons = `<button class="danger" onclick="act('gpus/${esc(g.uuid)}/drain', true, 'Drain ${esc(gpuLabel(g))}? Troiani will checkpoint and leave. Researcher processes are never killed.')">Drain</button>`;
    }
    return `<tr>
      <td class="mono">${esc(gpuLabel(g))}</td>
      <td>${esc((g.name || g.model || "").slice(0, 28))}</td>
      <td>${esc((g.vram_gb || g.memory_gb || 0).toFixed(0))}G</td>
      <td>${pill(g.occupancy, "occ-" + String(g.occupancy || "").toLowerCase())}</td>
      <td>${bar(smOf(g))}</td>
      <td>${bar(vramPct(g), "mem")}</td>
      <td>${esc(owner(g))}</td>
      <td class="mono">${esc(g.job_id || "—")}</td>
      <td>${buttons}</td>
    </tr>`;
  }).join("");
  $("gpu-body").innerHTML = rows || `<tr><td colspan="9" class="note">No GPUs reported yet.</td></tr>`;
}

function renderJobs(status) {
  const live = new Set(["SCHEDULED", "STARTING", "RUNNING", "CHECKPOINTING", "RESUMING"]);
  const paused = new Set(["PAUSED", "PREEMPTED", "QUEUED", "PENDING", "FAILED"]);
  const rows = (status.jobs || []).map((j) => {
    const id = j.id;
    const state = j.state;
    let actions = "";
    if (live.has(state)) {
      actions += `<button onclick="act('jobs/${esc(id)}/preempt', true, 'Preempt ${esc(id)}? Checkpoint then leave.')">Preempt</button>`;
      actions += `<button onclick="act('jobs/${esc(id)}/pause', true, 'Pause ${esc(id)}?')">Pause</button>`;
      actions += `<button class="danger" onclick="act('jobs/${esc(id)}/kill', true, 'Force-kill Troiani job ${esc(id)}? Never used on researcher PIDs.')">Kill</button>`;
      actions += `<button class="danger" onclick="act('jobs/${esc(id)}/cancel', true, 'Cancel ${esc(id)}?')">Cancel</button>`;
    } else if (paused.has(state)) {
      actions += `<button onclick="act('jobs/${esc(id)}/resume')">Resume</button>`;
      if (state !== "CANCELLED") actions += `<button class="danger" onclick="act('jobs/${esc(id)}/cancel', true, 'Cancel ${esc(id)}?')">Cancel</button>`;
    }
    const tps = j.tokens_per_sec != null ? `${fmtTokens(j.tokens_per_sec)}/s` : "—";
    return `<tr class="clickable" onclick="selectJob('${esc(id)}')">
      <td class="mono">${esc(id)}</td>
      <td>${pill(state, "occ-" + String(state).toLowerCase())}</td>
      <td>${esc((j.spec || {}).priority)}</td>
      <td>${esc((j.spec || {}).gpus)} · ${esc((j.assigned_refs || []).join(" ") || j.assigned_node || (j.spec || {}).preferred_node || "—")}</td>
      <td>${fmtDur(j.runtime_s)}</td>
      <td>${esc(j.last_step)}</td>
      <td>${j.last_loss == null ? "—" : Number(j.last_loss).toFixed(4)}${j.val_loss == null ? "" : ` <span class="sub">val ${Number(j.val_loss).toFixed(4)}</span>`}</td>
      <td>${fmtTokens(j.tokens_ingested)} <span class="sub">${esc(tps)}</span></td>
      <td class="mono">${esc(j.last_checkpoint_id || "—")}</td>
      <td>${actions || "—"}</td>
    </tr>`;
  }).join("");
  $("job-body").innerHTML = rows || `<tr><td colspan="10" class="note">No jobs yet.</td></tr>`;
}

async function selectJob(id) {
  selectedJob = id;
  const job = (lastStatus.jobs || []).find((j) => j.id === id);
  $("job-detail").textContent = job ? JSON.stringify(job, null, 2) : id;
  const res = await fetch("/v1/jobs/" + encodeURIComponent(id) + "/log");
  if (res.ok) {
    const data = await res.json();
    $("job-log").textContent = (data.text || "(empty)") + (data.source ? `\n\n— ${data.source}` : "");
  }
}

function datasetLabel(run) {
  const ds = run.dataset || {};
  const name = ds.name || ds.dataset_name || "";
  const rev = ds.revision || ds.dataset_hash || ds.version || "";
  if (!name && !rev) return "—";
  return rev ? `${name || "dataset"}@${String(rev).slice(0, 8)}` : name;
}

function renderRuns(status) {
  const rows = (status.runs || []).map((r) => `<tr>
    <td class="mono">${esc(r.id)}</td>
    <td>${esc(r.experiment)}</td>
    <td>${esc(r.status)}</td>
    <td>${fmtDur(r.runtime_s)}</td>
    <td>${fmtTokens(r.tokens_ingested)}</td>
    <td>${(r.metrics && r.metrics.val_loss != null) ? Number(r.metrics.val_loss).toFixed(4) : "—"}</td>
    <td>${r.tokens_per_sec == null ? "—" : fmtTokens(r.tokens_per_sec) + "/s"}</td>
    <td>${esc((r.hardware || {}).node || "—")}</td>
    <td class="mono">${esc(datasetLabel(r))}</td>
    <td class="mono">${esc(r.git_sha || "—").slice(0, 8)}</td>
    <td><button onclick="showRun('${esc(r.id)}')">Inspect</button></td>
  </tr>`).join("");
  $("run-body").innerHTML = rows || `<tr><td colspan="11" class="note">No runs yet.</td></tr>`;
}

async function showRun(id) {
  showTab("runs");
  const [lineage, repro] = await Promise.all([
    fetch("/v1/runs/" + encodeURIComponent(id) + "/lineage"),
    fetch("/v1/runs/" + encodeURIComponent(id) + "/reproduce"),
  ]);
  const body = {
    lineage: lineage.ok ? await lineage.json() : { error: lineage.status },
    reproduce: repro.ok ? await repro.json() : { error: repro.status },
  };
  $("run-detail").textContent = JSON.stringify(body, null, 2);
}

function renderWorkers(status) {
  const rows = (status.workers || []).map((w) => `<tr>
    <td>${esc(w.node)}</td>
    <td>${w.stale ? pill("STALE", "occ-researcher") : pill("LIVE", "occ-available")}</td>
    <td>${fmtDur(w.heartbeat_age_s)}</td>
    <td>${esc((w.gpus || []).length)}</td>
  </tr>`).join("");
  $("worker-body").innerHTML = rows || `<tr><td colspan="4" class="note">No workers.</td></tr>`;
}

function renderUserOccupancy(status) {
  const rows = (status.user_occupancy || []).map((u) => {
    const other = Number(u.other_s) || 0;
    const kind = other > 0 ? "other" : "troiani";
    return `<tr>
      <td class="mono">${esc(u.username)}</td>
      <td>${esc(u.node)}</td>
      <td>${fmtDur(u.seconds)}</td>
      <td>${pill(kind, other > 0 ? "occ-researcher" : "occ-troiani")}</td>
    </tr>`;
  }).join("");
  $("user-occ-body").innerHTML = rows || `<tr><td colspan="4" class="note">No process time yet.</td></tr>`;
}

function renderSharing(status) {
  const rows = (status.annoyance || []).map((a) => {
    const score = Number(a.score) || 0;
    const cls = score >= 50 ? "occ-researcher" : score > 0 ? "occ-draining" : "occ-available";
    const why = (a.reasons || []).join(" · ") || "quiet";
    return `<tr>
      <td>${esc(a.node)}</td>
      <td>${pill(String(score), cls)}</td>
      <td>${esc(why)}</td>
    </tr>`;
  }).join("");
  $("share-body").innerHTML = rows || `<tr><td colspan="3" class="note">No nodes yet.</td></tr>`;
}

function renderIntents(status) {
  const rows = (status.intents || []).slice(0, 12).map((i) => `<tr>
    <td>${esc(String(i.ts || "").slice(0, 19))}</td>
    <td>${esc(i.kind)}</td>
    <td>${esc(i.node)}</td>
    <td>${esc(i.username || "—")}</td>
    <td>${esc(String(i.detail || "").slice(0, 90))}</td>
  </tr>`).join("");
  $("intent-body").innerHTML = rows || `<tr><td colspan="5" class="note">Quiet.</td></tr>`;
}

function renderEvents(status) {
  const rows = (status.events || []).slice(0, 16).map((e) => `<tr>
    <td>${esc(String(e.ts || "").slice(0, 19))}</td>
    <td>${esc(e.type)}</td>
    <td class="mono">${esc(e.job_id || "—")}</td>
    <td>${esc(JSON.stringify(e.payload || {}).slice(0, 90))}</td>
  </tr>`).join("");
  $("event-body").innerHTML = rows || `<tr><td colspan="4" class="note">No events.</td></tr>`;
}

function renderActivityUsers(status) {
  const body = $("activity-user-body");
  if (!body) return;
  const rows = (status.activity_users || []).map((u) => {
    const kinds = Object.entries(u.kinds || {}).map(([k, n]) => `${k}:${n}`).join(" ");
    return `<tr class="clickable" onclick="selectUser('${esc(u.username)}')">
      <td class="mono">${esc(u.username)}</td>
      <td>${esc(u.signals)}</td>
      <td>${esc((u.nodes || []).join(", ") || "—")}</td>
      <td>${esc(kinds || "—")}</td>
      <td>${fmtDur(u.seconds)}</td>
    </tr>`;
  }).join("");
  body.innerHTML = rows || `<tr><td colspan="5" class="note">No other-user activity yet.</td></tr>`;
}

function selectUser(name) {
  selectedUser = name;
  const found = (lastStatus.activity_users || []).find((u) => u.username === name);
  $("activity-user-detail").textContent = found ? JSON.stringify(found, null, 2) : name;
}

async function refreshInfra() {
  const panel = document.querySelector('.tab[data-panel="infra"]');
  if (!panel || !panel.classList.contains("on")) return;
  const res = await fetch("/v1/infra");
  if (!res.ok) return;
  lastInfra = await res.json();
  const control = lastInfra.control || {};
  const disk = control.disk || {};
  const folders = control.folders || [];
  const ckpts = folders.find((f) => f.name === "checkpoints") || {};
  $("infra-cards").innerHTML = [
    ["Disk used", fmtGb(disk.used_gb), `${fmtGb(disk.free_gb)} free`],
    ["Checkpoints", fmtGb(ckpts.gb), `${ckpts.files ?? 0} files`],
    ["Runs", String((lastInfra.runs || []).length), "logical experiments/…"],
    ["Nodes", String(Object.keys(lastInfra.nodes || {}).length), "heartbeat snapshots"],
  ].map(([k, v, s]) => `<div class="card"><div class="label">${esc(k)}</div><div class="value">${esc(v)}</div><div class="sub">${esc(s)}</div></div>`).join("");
  $("infra-folder-body").innerHTML = folders.map((f) => `<tr>
    <td>${esc(f.name)}</td><td class="mono">${esc(f.path)}</td><td>${esc(fmtGb(f.gb))}</td><td>${esc(f.files)}</td>
  </tr>`).join("") || `<tr><td colspan="4" class="note">No folders.</td></tr>`;
  const runBytes = Object.fromEntries((control.checkpoint_runs || []).map((r) => [r.run_id, r]));
  $("infra-ckpt-body").innerHTML = (lastInfra.runs || []).map((r) => {
    const diskRow = runBytes[r.run_id] || {};
    return `<tr>
      <td class="mono">${esc(r.run_id)}</td>
      <td>${esc(r.experiment || "—")}</td>
      <td>${esc(r.checkpoints)}</td>
      <td>${esc(diskRow.gb != null ? fmtGb(diskRow.gb) : "—")}</td>
      <td><button class="danger" onclick="gcCheckpoints('${esc(r.run_id)}')">GC</button></td>
    </tr>`;
  }).join("") || `<tr><td colspan="5" class="note">No checkpoint runs.</td></tr>`;
  $("infra-node-body").innerHTML = Object.entries(lastInfra.nodes || {}).map(([node, info]) => {
    const d = (info && info.disk) || {};
    const c = (info && info.checkpoints) || {};
    return `<tr><td>${esc(node)}</td><td>${esc(fmtGb(d.used_gb))}</td><td>${esc(fmtGb(d.free_gb))}</td><td>${esc(fmtGb(c.gb))}</td></tr>`;
  }).join("") || `<tr><td colspan="4" class="note">Workers have not reported disk yet.</td></tr>`;
  const tops = [];
  Object.entries(lastInfra.nodes || {}).forEach(([node, info]) => {
    (info.top || []).forEach((p) => tops.push({ ...p, node }));
  });
  tops.sort((a, b) => (b.cpu || 0) - (a.cpu || 0));
  $("infra-top-body").innerHTML = tops.slice(0, 18).map((p) => `<tr>
    <td class="mono">${esc(p.pid)}</td><td>${esc(p.user || "—")}</td>
    <td>${esc(p.node)} · ${esc(p.name)}</td><td>${esc(p.cpu)}</td><td>${esc(p.rss_mb)}M</td>
  </tr>`).join("") || `<tr><td colspan="5" class="note">No process sample yet.</td></tr>`;
}

async function gcCheckpoints(runId) {
  const msg = runId ? `Garbage-collect old checkpoints for ${runId}? Latest / best / preemption are kept.` : "Garbage-collect old checkpoints on this control node?";
  await act("infra/gc", true, msg, "POST", runId ? { run_id: runId } : {});
  await refreshInfra();
}

function renderCards(status) {
  const m = status.metrics || {};
  const jobs = status.jobs || [];
  const live = jobs.filter((j) => ["RUNNING", "STARTING", "RESUMING", "CHECKPOINTING"].includes(j.state)).length;
  const tokens = jobs.reduce((s, j) => s + (Number(j.tokens_ingested) || 0), 0);
  $("cards").innerHTML = [
    ["Troiani GPU-h", ((m.troiani_gpu_seconds || 0) / 3600).toFixed(2), `${fmtDur(m.troiani_gpu_seconds || 0)} useful`],
    ["Researcher GPU-h", ((m.researcher_gpu_seconds || 0) / 3600).toFixed(2), "never touched"],
    ["Tokens ingested", fmtTokens(m.tokens_ingested || tokens), `${live} live job(s)`],
    ["Preemptions", String(Math.round(m.preemptions || 0)), "cooperative yields"],
    ["Available GPU-h", ((m.available_gpu_seconds || 0) / 3600).toFixed(2), "idle capacity"],
  ].map(([k, v, s]) => `<div class="card"><div class="label">${esc(k)}</div><div class="value">${esc(v)}</div><div class="sub">${esc(s)}</div></div>`).join("");
  const stopped = status.stop_all || (status.policy && status.policy.stop_all);
  const open = !stopped && status.policy_open !== false;
  $("window-pill").textContent = stopped ? "STOP ALL" : (open ? "window open" : "window closed");
  $("window-pill").className = "pill " + (stopped ? "occ-researcher" : open ? "occ-available" : "occ-draining");
}

const RULE_PRESETS = {
  weekends: { name: "weekends", start: "00:00", end: "23:59", days: ["sat", "sun"], enabled: false },
  weeknights: { name: "weeknights", start: "20:00", end: "08:00", days: ["mon", "tue", "wed", "thu", "fri"], enabled: false },
  "weekend-nights": { name: "weekend-nights", start: "20:00", end: "08:00", days: ["sat", "sun"], enabled: false },
  lunch: { name: "lunch", start: "12:00", end: "14:00", days: ["mon", "tue", "wed", "thu", "fri"], enabled: false },
  special: { name: "special", start: "09:00", end: "11:00", days: [], enabled: false },
  festive: { name: "festive", start: "00:00", end: "23:59", days: [], enabled: false },
};

const AGG_HINT = {
  low: "Low: leave a card unless leftover VRAM is under 4 GB. Longer cooldown, 2 GPU cap.",
  mid: "Mid: take a card if leftover VRAM < 2 GB. Default lab sharing.",
  extreme: "Extreme: claim almost-idle cards (0.5 GB leftover). Short cooldown, 6 GPU cap.",
};

function proposedPolicy() {
  if ($("timezone")) readWindowForm();
  return {
    timezone: $("timezone") ? $("timezone").value : "",
    max_troiani_gpus: Number($("max_troiani_gpus").value),
    max_jobs: Number($("max_jobs").value),
    max_gpus_per_job: Number($("max_gpus_per_job").value),
    preempt_grace_s: Number($("preempt_grace_s").value),
    cooldown_s: Number($("cooldown_s").value),
    max_runtime_s: Number($("max_runtime_s").value),
    aggressiveness: $("aggressiveness") ? $("aggressiveness").value : "mid",
    stop_all: Boolean(lastStatus.stop_all || (lastStatus.policy && lastStatus.policy.stop_all)),
    windows: windowsState.filter((w) => w.start && w.end),
  };
}

function policySummary(p) {
  const rules = (p.windows || []).map((w) => {
    const days = (w.days || []).join(",") || "any day";
    return `${w.enabled ? "ON " : "off"} ${w.name || "rule"} ${w.start}-${w.end} ${days}`;
  });
  return [
    p.stop_all ? "STOP ALL latched" : "Normal",
    `Claim ${p.aggressiveness || "mid"} · ${p.max_troiani_gpus} GPUs · ${p.max_jobs} jobs · cooldown ${p.cooldown_s}s`,
    rules.length ? rules.join("\n") : "No rules — opportunistic any hour",
  ].join("\n");
}

function renderPolicyCompare() {
  const current = lastStatus.policy || {};
  if ($("policy-current")) $("policy-current").textContent = policySummary(current);
  if ($("policy-proposed")) $("policy-proposed").textContent = policySummary(proposedPolicy());
}

function setAggressiveness(level, applyKnobs) {
  const key = ["low", "mid", "extreme"].includes(level) ? level : "mid";
  if ($("aggressiveness")) $("aggressiveness").value = key;
  document.querySelectorAll("#agg-buttons button").forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.agg === key);
  });
  if ($("agg-hint")) $("agg-hint").textContent = AGG_HINT[key];
  if (applyKnobs) {
    const knobs = { low: [4, 60, 2, 2], mid: [2, 30, 4, 4], extreme: [0.5, 5, 6, 6] }[key];
    $("cooldown_s").value = knobs[1];
    $("max_troiani_gpus").value = knobs[2];
    $("max_jobs").value = knobs[3];
    $("policy-form").dataset.dirty = "1";
  }
}

function renderEmergency(status) {
  const stopped = status.stop_all || (status.policy && status.policy.stop_all);
  if ($("stop-state")) $("stop-state").textContent = stopped ? "STOP ALL latched" : "Normal";
  if ($("emergency")) $("emergency").classList.toggle("latched", Boolean(stopped));
}

function renderPolicy(status) {
  const p = status.policy || {};
  if ($("policy-form").dataset.dirty !== "1") {
    $("timezone").value = p.timezone || "Europe/Madrid";
    $("max_troiani_gpus").value = p.max_troiani_gpus ?? 4;
    $("max_jobs").value = p.max_jobs ?? 4;
    $("max_gpus_per_job").value = p.max_gpus_per_job ?? 4;
    $("preempt_grace_s").value = p.preempt_grace_s ?? 120;
    $("cooldown_s").value = p.cooldown_s ?? 30;
    $("max_runtime_s").value = p.max_runtime_s ?? 0;
    windowsState = (p.windows || []).map(normalizeWindow);
    renderWindowRows();
    setAggressiveness(p.aggressiveness || "mid", false);
  }
  renderEmergency(status);
  renderPolicyCompare();
}

function renderLaunchNodes(status) {
  const select = $("launch-node");
  const current = select.value;
  const nodes = status.nodes || [];
  select.innerHTML = `<option value="">any idle node</option>` + nodes.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if ([...select.options].some((o) => o.value === current)) select.value = current;
}

function drawChart(status) {
  const canvas = $("chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const hist = (status.metrics && status.metrics.history) || [];
  if (hist.length < 2) {
    ctx.fillStyle = "#6a655e";
    ctx.font = "12px IBM Plex Mono";
    ctx.fillText("SM (solid) and VRAM (dashed) appear after a few heartbeats.", 12, h / 2);
    return;
  }
  const keys = Object.keys(hist[0]).filter((k) => k !== "ts");
  const colors = ["#9a3412", "#1d4ed8", "#3f6212", "#a16207", "#7c3aed", "#0f766e"];
  const draw = (metric, dashed) => {
    keys.forEach((key, idx) => {
      ctx.beginPath();
      ctx.strokeStyle = colors[idx % colors.length];
      ctx.setLineDash(dashed ? [4, 3] : []);
      hist.forEach((sample, i) => {
        const point = sample[key] || {};
        const value = metric === "sm"
          ? Number(point.sm != null ? point.sm : point.util || 0)
          : Number(point.vram != null ? point.vram : (100 * (point.mem || 0) / 80));
        const x = (i / (hist.length - 1)) * (w - 8) + 4;
        const y = h - 6 - (Math.max(0, Math.min(100, value)) / 100) * (h - 16);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  };
  draw("sm", false);
  draw("vram", true);
  ctx.setLineDash([]);
}

function applyTemplate(spec) {
  $("launch-name").value = spec.name || "adhoc";
  $("launch-exp").value = spec.experiment || "adhoc";
  $("launch-gpus").value = spec.gpus || 1;
  $("launch-mem").value = spec.min_gpu_memory_gb || 0;
  $("launch-type").value = spec.preferred_gpu_type || "";
  $("launch-node").value = spec.preferred_node || "";
  $("launch-ckpt").value = spec.checkpoint_interval || 60;
  $("launch-max").value = spec.max_runtime || 0;
  $("launch-cmd").value = (spec.command || []).join("\n");
}

function renderTemplates() {
  $("templates").innerHTML = templates.map((t) =>
    `<button type="button" data-tpl="${esc(t.id)}">${esc(t.id)}</button>`
  ).join("");
}

async function refresh() {
  const res = await fetch("/v1/status");
  if (!res.ok) return;
  const status = await res.json();
  lastStatus = status;
  $("clock").textContent = status.now || "";
  renderCards(status);
  renderGpus(status);
  renderJobs(status);
  renderRuns(status);
  renderWorkers(status);
  renderSharing(status);
  renderUserOccupancy(status);
  renderIntents(status);
  renderEvents(status);
  renderActivityUsers(status);
  renderPolicy(status);
  renderLaunchNodes(status);
  drawChart(status);
  refreshInfra();
  if (selectedJob) {
    const job = (status.jobs || []).find((j) => j.id === selectedJob);
    if (job) $("job-detail").textContent = JSON.stringify(job, null, 2);
  }
}

async function savePolicy(event) {
  event.preventDefault();
  const next = proposedPolicy();
  const current = lastStatus.policy || {};
  const summary = `Save this policy?\n\nCURRENT windows: ${JSON.stringify(current.windows || [])}\n\nPROPOSED windows: ${JSON.stringify(next.windows)}\n\nCaps ${current.max_troiani_gpus}→${next.max_troiani_gpus} GPUs, ${current.max_jobs}→${next.max_jobs} jobs.`;
  if (!window.confirm(summary)) return;
  await act("policy", false, "", "PUT", next);
  $("policy-form").dataset.dirty = "0";
}

async function submitLaunch(event) {
  event.preventDefault();
  const command = $("launch-cmd").value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const ok = await act("jobs", false, "", "POST", {
    name: $("launch-name").value,
    experiment: $("launch-exp").value,
    gpus: Number($("launch-gpus").value),
    min_gpu_memory_gb: Number($("launch-mem").value),
    preferred_gpu_type: $("launch-type").value || null,
    preferred_node: $("launch-node").value || "",
    checkpoint_interval: Number($("launch-ckpt").value),
    max_runtime: Number($("launch-max").value),
    priority: "opportunistic",
    preemptible: true,
    command,
  });
  if (ok) showTab("jobs");
}

async function submitYaml(event) {
  event.preventDefault();
  const text = $("launch-yaml").value.trim();
  if (!text) {
    $("toast").textContent = "Paste a YAML task or choose a file.";
    return;
  }
  const ok = await act("jobs/from-yaml", false, "", "POST", { yaml: text });
  if (ok) showTab("jobs");
}

function applyTheme(theme) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("tp-theme", next); } catch (err) { /* ignore */ }
  const btn = $("theme-toggle");
  if (btn) btn.textContent = next === "dark" ? "Light" : "Dark";
}

window.act = act;
window.selectJob = selectJob;
window.showRun = showRun;
window.selectUser = selectUser;
window.gcCheckpoints = gcCheckpoints;
window.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll(".tabs button").forEach((btn) => {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  });
  const initial = (window.location.hash || "#cluster").slice(1);
  showTab(initial || "cluster");
  $("policy-form").addEventListener("submit", savePolicy);
  $("policy-form").addEventListener("input", () => { $("policy-form").dataset.dirty = "1"; renderPolicyCompare(); });
  $("add-window").addEventListener("click", () => {
    readWindowForm();
    windowsState.push({ name: "slot", start: "20:00", end: "08:00", days: ["mon", "tue", "wed", "thu", "fri"], enabled: false });
    $("policy-form").dataset.dirty = "1";
    renderWindowRows();
    renderPolicyCompare();
  });
  $("rule-chips").addEventListener("click", (event) => {
    const name = event.target.dataset.rule;
    if (!name || !RULE_PRESETS[name]) return;
    readWindowForm();
    if (windowsState.some((w) => w.name === name)) return;
    windowsState.push({ ...RULE_PRESETS[name] });
    $("policy-form").dataset.dirty = "1";
    renderWindowRows();
    renderPolicyCompare();
  });
  $("preset-clear").addEventListener("click", () => {
    windowsState = [];
    $("policy-form").dataset.dirty = "1";
    renderWindowRows();
    renderPolicyCompare();
  });
  $("agg-buttons").addEventListener("click", (event) => {
    const level = event.target.dataset.agg;
    if (!level) return;
    setAggressiveness(level, true);
    renderPolicyCompare();
  });
  $("stop-all").addEventListener("click", () => {
    act("policy/stop-all", true, "STOP ALL? Every opportunistic Troiani job will checkpoint and leave. Researcher processes are never killed.");
  });
  $("resume-all").addEventListener("click", () => {
    act("policy/resume", false);
  });
  $("window-rows").addEventListener("click", (event) => {
    const del = event.target.dataset.del;
    const toggle = event.target.dataset.toggle;
    if (toggle != null) {
      readWindowForm();
      const idx = Number(toggle);
      windowsState[idx].enabled = !windowsState[idx].enabled;
      $("policy-form").dataset.dirty = "1";
      renderWindowRows();
      renderPolicyCompare();
      return;
    }
    if (del == null) return;
    readWindowForm();
    windowsState.splice(Number(del), 1);
    $("policy-form").dataset.dirty = "1";
    renderWindowRows();
    renderPolicyCompare();
  });
  $("launch-form").addEventListener("submit", submitLaunch);
  $("yaml-form").addEventListener("submit", submitYaml);
  $("yaml-file").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    $("launch-yaml").value = await file.text();
  });
  $("templates").addEventListener("click", (event) => {
    const id = event.target.dataset.tpl;
    const found = templates.find((t) => t.id === id);
    if (found) applyTemplate(found.spec);
  });
  try {
    const res = await fetch("/v1/templates");
    if (res.ok) templates = await res.json();
  } catch (err) {
    templates = [];
  }
  renderTemplates();
  let theme = "light";
  try { theme = localStorage.getItem("tp-theme") || "light"; } catch (err) { theme = "light"; }
  applyTheme(theme);
  $("theme-toggle").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  refresh();
  setInterval(refresh, 2000);
});
