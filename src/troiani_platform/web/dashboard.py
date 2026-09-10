from __future__ import annotations

from typing import Any


def render_dashboard(_status: dict[str, Any] | None = None) -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Troiani Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="/static/app.css"/>
</head>
<body>
  <header class="top">
    <div class="brand">
      <div class="mark">Tp</div>
      <div>
        <h1>Troiani Platform</h1>
        <p class="lede">Atlas · Uranus · opportunistic</p>
      </div>
    </div>
    <div class="meta">
      <span id="window-pill" class="pill">window</span>
      <span id="clock" class="clock"></span>
      <nav class="ext">
        <button type="button" id="theme-toggle">Dark</button>
        <a href="http://127.0.0.1:5000" target="_blank" rel="noreferrer">MLflow</a>
        <a href="http://127.0.0.1:6006" target="_blank" rel="noreferrer">TensorBoard</a>
        <a href="/v1/status">status</a>
      </nav>
    </div>
  </header>
  <nav class="tabs" id="tabs">
    <button data-tab="cluster" class="on">Cluster</button>
    <button data-tab="jobs">Jobs</button>
    <button data-tab="launch">Launch</button>
    <button data-tab="policy">Policy</button>
    <button data-tab="runs">Runs</button>
    <button data-tab="activity">Activity</button>
    <button data-tab="infra">Infra</button>
  </nav>
  <p class="toast" id="toast"></p>

  <section class="tab on" data-panel="cluster">
    <div class="cards" id="cards"></div>
    <div class="split">
      <div>
        <h2>GPUs</h2>
        <div class="gpu-strip" id="gpu-strip"></div>
        <table>
          <thead><tr><th>GPU</th><th>Type</th><th>VRAM</th><th>State</th><th>SM</th><th>VRAM used</th><th>Who</th><th>Job</th><th></th></tr></thead>
          <tbody id="gpu-body"></tbody>
        </table>
      </div>
      <aside>
        <h2>Workers</h2>
        <table>
          <thead><tr><th>Node</th><th>Pulse</th><th>Age</th><th>GPUs</th></tr></thead>
          <tbody id="worker-body"></tbody>
        </table>
        <h2>Sharing</h2>
        <table>
          <thead><tr><th>Node</th><th>Score</th><th>Why</th></tr></thead>
          <tbody id="share-body"></tbody>
        </table>
        <h2>Who has the GPUs</h2>
        <table>
          <thead><tr><th>User</th><th>Node</th><th>Time</th><th>Kind</th></tr></thead>
          <tbody id="user-occ-body"></tbody>
        </table>
        <p class="note">Courtesy only. High means other people are on the box. Researcher processes are never killed.</p>
      </aside>
    </div>
    <h2>SM vs VRAM</h2>
    <p id="chart-legend"><b>Solid SM</b> = compute busy. <b>Dashed VRAM</b> = memory occupied. High VRAM + low SM is a parked model.</p>
    <canvas id="chart"></canvas>
  </section>

  <section class="tab" data-panel="jobs">
    <div class="row-head">
      <h2>Jobs</h2>
      <button class="danger" onclick="act('release-all', true, 'Release every opportunistic Troiani job? They will checkpoint and leave.')">Release all Troiani</button>
    </div>
    <table>
      <thead><tr><th>Job</th><th>State</th><th>Pri</th><th>GPUs / node</th><th>Runtime</th><th>Step</th><th>Loss</th><th>Tokens</th><th>Ckpt</th><th>Actions</th></tr></thead>
      <tbody id="job-body"></tbody>
    </table>
    <div class="split">
      <div>
        <h2>Selected job</h2>
        <pre class="log" id="job-detail">Click a job id to inspect.</pre>
      </div>
      <div>
        <h2>Log tail</h2>
        <pre class="log" id="job-log">Worker heartbeats stream the last 12k of each job log into the control plane.</pre>
      </div>
    </div>
  </section>

  <section class="tab" data-panel="launch">
    <h2>Launch</h2>
    <p class="note">Same path as <code>troiani-platform job submit</code>. Paste a Troiani or SkyPilot-shaped YAML, or type argv. Never a shell string. Idle cards only.</p>
    <div class="templates" id="templates"></div>
    <form class="launch" id="yaml-form">
      <label class="wide">Task YAML (file or paste)
        <input type="file" id="yaml-file" accept=".yaml,.yml">
        <textarea id="launch-yaml" rows="8" placeholder="name: exp&#10;resources:&#10;  accelerators: A100:1&#10;run: python -m troiani_platform.training.dummy_train --steps 80"></textarea>
      </label>
      <div class="wide"><button type="submit">Submit YAML</button></div>
    </form>
    <form class="launch" id="launch-form">
      <label>Name <input id="launch-name" required value="adhoc"></label>
      <label>Experiment <input id="launch-exp" value="adhoc"></label>
      <label>GPUs <input id="launch-gpus" type="number" min="1" value="1" required></label>
      <label>Min VRAM (GB) <input id="launch-mem" type="number" min="0" value="40"></label>
      <label>Preferred type <input id="launch-type" placeholder="A100 / 40GB / 80GB"></label>
      <label>Preferred node
        <select id="launch-node"><option value="">any idle node</option></select>
      </label>
      <label>Checkpoint every (s) <input id="launch-ckpt" type="number" min="0" value="60"></label>
      <label>Max runtime (s, 0=off) <input id="launch-max" type="number" min="0" value="180"></label>
      <label class="wide">Command (one argv token per line)
        <textarea id="launch-cmd" rows="6" required>python
-m
troiani_platform.training.dummy_train
--steps
80
--sleep
0.1</textarea>
      </label>
      <div class="wide"><button type="submit">Submit job</button></div>
    </form>
  </section>

  <section class="tab" data-panel="policy">
    <h2>Policy</h2>
    <p class="note">Named rules compose the window. All-disabled = any hour. STOP ALL latches a yield: Troiani checkpoints and leaves. Researchers are never killed.</p>
    <div class="emergency" id="emergency">
      <div>
        <strong id="stop-state">Normal</strong>
        <span class="note">Partial stop = disable one rule. STOP ALL preempts Troiani jobs only.</span>
      </div>
      <div>
        <button type="button" class="danger" id="stop-all">STOP ALL</button>
        <button type="button" id="resume-all">Resume</button>
      </div>
    </div>
    <div class="row-head">
      <h3>Claim aggressiveness</h3>
      <div class="agg" id="agg-buttons">
        <button type="button" data-agg="low">Low</button>
        <button type="button" data-agg="mid">Mid</button>
        <button type="button" data-agg="extreme">Extreme</button>
      </div>
    </div>
    <p class="note" id="agg-hint">Mid: take a card if leftover VRAM &lt; 2 GB. Low is shy; extreme takes almost-idle cards.</p>
    <div class="split policy-compare">
      <div>
        <h3>Current</h3>
        <pre class="log short" id="policy-current">Loading saved policy…</pre>
      </div>
      <div>
        <h3>Proposed</h3>
        <pre class="log short" id="policy-proposed">Edit a rule to preview the diff.</pre>
      </div>
    </div>
    <form class="policy" id="policy-form">
      <label>Timezone <input id="timezone" value="Europe/Madrid"></label>
      <label>Max Troiani GPUs <input id="max_troiani_gpus" type="number" min="0" required></label>
      <label>Max jobs <input id="max_jobs" type="number" min="0" required></label>
      <label>Max GPUs / job <input id="max_gpus_per_job" type="number" min="1" required></label>
      <label>Preempt grace (s) <input id="preempt_grace_s" type="number" min="0" required></label>
      <label>Cooldown (s) <input id="cooldown_s" type="number" min="0" required></label>
      <label>Max runtime (s, 0=off) <input id="max_runtime_s" type="number" min="0" required></label>
      <input type="hidden" id="aggressiveness" value="mid">
      <div class="wide">
        <div class="row-head">
          <h3>Rules</h3>
          <div class="chips" id="rule-chips">
            <button type="button" data-rule="weekends">Weekend all</button>
            <button type="button" data-rule="weeknights">Weeknights</button>
            <button type="button" data-rule="weekend-nights">Weekend nights</button>
            <button type="button" data-rule="lunch">Lunch</button>
            <button type="button" data-rule="special">Special slot</button>
            <button type="button" data-rule="festive">Festive</button>
            <button type="button" id="add-window">Blank rule</button>
            <button type="button" id="preset-clear">Clear rules</button>
          </div>
        </div>
        <div id="window-rows"></div>
      </div>
      <div class="wide"><button type="submit">Save proposed policy</button></div>
    </form>
  </section>

  <section class="tab" data-panel="runs">
    <h2>Runs</h2>
    <table>
      <thead><tr><th>Run</th><th>Exp</th><th>Status</th><th>Runtime</th><th>Tokens</th><th>val</th><th>tok/s</th><th>Node</th><th>Dataset</th><th>Git</th><th></th></tr></thead>
      <tbody id="run-body"></tbody>
    </table>
    <pre class="log" id="run-detail">Lineage / reproduce appear here.</pre>
  </section>

  <section class="tab" data-panel="activity">
    <h2>Per user</h2>
    <table>
      <thead><tr><th>User</th><th>Signals</th><th>Nodes</th><th>Kinds</th><th>Time</th></tr></thead>
      <tbody id="activity-user-body"></tbody>
    </table>
    <pre class="log" id="activity-user-detail">Click a user to see their recent signals.</pre>
    <div class="split">
      <div>
        <h2>Other people</h2>
        <table>
          <thead><tr><th>When</th><th>Signal</th><th>Node</th><th>Who</th><th>Detail</th></tr></thead>
          <tbody id="intent-body"></tbody>
        </table>
      </div>
      <div>
        <h2>Events</h2>
        <table>
          <thead><tr><th>When</th><th>Type</th><th>Job</th><th>Payload</th></tr></thead>
          <tbody id="event-body"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="tab" data-panel="infra">
    <div class="row-head">
      <h2>Infra</h2>
      <button class="danger" onclick="gcCheckpoints()">GC checkpoints</button>
    </div>
    <div class="cards" id="infra-cards"></div>
    <div class="split">
      <div>
        <h2>Folders</h2>
        <table>
          <thead><tr><th>Name</th><th>Path</th><th>GB</th><th>Files</th></tr></thead>
          <tbody id="infra-folder-body"></tbody>
        </table>
        <h2>Checkpoint runs</h2>
        <table>
          <thead><tr><th>Run</th><th>Exp</th><th>Ckpts</th><th>GB</th><th></th></tr></thead>
          <tbody id="infra-ckpt-body"></tbody>
        </table>
      </div>
      <div>
        <h2>Nodes</h2>
        <table>
          <thead><tr><th>Node</th><th>Disk used</th><th>Free</th><th>Ckpts</th></tr></thead>
          <tbody id="infra-node-body"></tbody>
        </table>
        <h2>Top</h2>
        <table>
          <thead><tr><th>PID</th><th>User</th><th>Name</th><th>CPU</th><th>RSS</th></tr></thead>
          <tbody id="infra-top-body"></tbody>
        </table>
      </div>
    </div>
  </section>

  <script src="/static/app.js"></script>
</body>
</html>
"""
