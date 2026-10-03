"use strict";
// RetroEnv explorer: overview, task table, task pages with reference routes and trajectories, model runs.

const $ = (selector, root = document) => root.querySelector(selector);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
const pct = (x) => (x == null ? "–" : `${(100 * x).toFixed(1)}%`);
const num = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const store = {
  get(key, fallback) {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* storage may be blocked */
    }
  },
};
const state = { benchmark: null, dataset: "" };
let META = null;
const overviewCache = new Map();

async function api(path, params = {}) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null)).toString();
  const response = await fetch(`/api/${path}${query ? `?${query}` : ""}`);
  if (!response.ok) {
    const text = await response.text();
    // FastAPI's bare "Not Found" means the route itself is missing: the server predates this page.
    if (response.status === 404 && text === '{"detail":"Not Found"}') {
      throw new Error("The explorer server is older than this page. Stop it (Ctrl+C) and start it again.");
    }
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

function go(path, params = {}) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null)).toString();
  location.hash = `${path}${query ? `?${query}` : ""}`;
}

// --- molecules: drawn by RDKit on the server, in currentColor so they follow the theme ---
const svgCache = new Map();
function mol(smiles, w = 200, h = 110, cls = "") {
  return `<span class="mol ${cls}" data-smi="${esc(smiles)}" data-w="${w}" data-h="${h}" title="${esc(smiles)}" style="width:${w}px;height:${h}px"></span>`;
}
async function hydrate(root = document) {
  const pending = [...root.querySelectorAll(".mol[data-smi]:not([data-done])")];
  await Promise.all(
    pending.map(async (el) => {
      el.dataset.done = "1";
      const key = `${el.dataset.smi}|${el.dataset.w}|${el.dataset.h}`;
      if (!svgCache.has(key)) {
        const query = new URLSearchParams({ smi: el.dataset.smi, w: el.dataset.w, h: el.dataset.h });
        svgCache.set(
          key,
          fetch(`/api/mol.svg?${query}`).then((r) => r.text()),
        );
      }
      el.innerHTML = await svgCache.get(key);
    }),
  );
}

function bars(rows, attrs = () => "") {
  const max = Math.max(1, ...rows.map((r) => r[1]));
  return `<div class="bars">${rows
    .map((r) => {
      const [label, value, shown] = r;
      return `<div class="row" ${attrs(r)}><span class="lbl">${esc(label)}</span><div class="track"><div class="fill" style="width:${((100 * value) / max).toFixed(1)}%"></div></div><span class="n">${shown ?? fmt(value)}</span></div>`;
    })
    .join("")}</div>`;
}
const STEP_ORDER = (a, b) => Number(a[0]) - Number(b[0]);
const TIER_ORDER = { easy: 0, medium: 1, hard: 2 };
const SIZE_ORDER = { "<=20": 0, "21-30": 1, "31-40": 2, "41+": 3 };

async function overviewData() {
  const key = `${state.benchmark}|${state.dataset}`;
  if (!overviewCache.has(key))
    overviewCache.set(key, api("overview", { benchmark: state.benchmark, dataset: state.dataset }));
  return overviewCache.get(key);
}

// --- environment (the RL side) ---
async function overviewView(view, params) {
  const data = await overviewData();
  const splits = Object.keys(data.splits);
  const focus =
    params.split && data.splits[params.split] ? params.split : splits.includes("train") ? "train" : splits[0];
  const s = data.splits[focus];
  const stepKeys = [...new Set(splits.flatMap((k) => Object.keys(data.splits[k].steps)))].sort((a, b) => a - b);
  const filterLink =
    (key) =>
    ([label]) =>
      `data-go='${esc(JSON.stringify({ split: focus, [key]: label }))}'`;
  view.innerHTML = `
    <div class="grid" style="grid-template-columns:minmax(0,3fr) minmax(0,2fr)">
      <div class="panel"><div class="panel-h"><h2>${esc(state.benchmark)}: RL tasks by split</h2></div>
        <div class="tbl" style="border:0"><table>
          <tr><th>Split</th><th>Used for</th><th class="num">Tasks</th><th class="num">Step budget ${stepKeys.join(" / ")}</th><th class="num">Easy / medium / hard</th><th class="num">Two-route</th></tr>
          ${splits
            .map((k) => {
              const x = data.splits[k];
              return `<tr class="click${k === focus ? " on" : ""}" data-split="${k}"><td>${k}</td>
            <td class="sm muted">${{ train: "RL rollouts and SFT", dev: "checkpoint selection", eval: "the board", stress: "sealed second test" }[k] || ""}</td>
            <td class="num">${fmt(x.tasks)}</td><td class="num">${stepKeys.map((d) => fmt(x.steps[d] || 0)).join(" / ")}</td>
            <td class="num">${["easy", "medium", "hard"].map((t) => fmt(x.tier[t] || 0)).join(" / ")}</td><td class="num">${fmt(x.two_route)}</td></tr>`;
            })
            .join("")}
        </table></div></div>
      <div class="panel"><div class="panel-h"><h2>One episode</h2></div><div class="panel-b"><dl class="kv">
        <dt>Observation</dt><dd>The target SMILES, a step budget and how many routes to return; no stock list and no answer</dd>
        <dt>Actions</dt><dd>Calls to 9 tools, up to 32 calls over 16 model turns</dd>
        <dt>End</dt><dd>One <span class="mono">emit_routes</span> call with the route trees</dd>
        <dt>Reward</dt><dd>0 to 1 from 9 verifier components, scored against the hidden patent route</dd>
        <dt>Pass</dt><dd>Every required route verified</dd>
      </dl></div></div>
    </div>
    <div class="section">
      <h2>${esc(focus)} tasks <span class="faint sm">· click a bar to list them</span></h2>
      <div class="grid">
        <div class="panel"><div class="panel-h"><h3>Step budget</h3></div><div class="panel-b">${bars(
          Object.entries(s.steps)
            .sort(STEP_ORDER)
            .map(([k, v]) => [k, v]),
          filterLink("steps"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>First reaction of the reference route</h3></div><div class="panel-b">${bars(
          Object.entries(s.family).sort((a, b) => b[1] - a[1]),
          filterLink("family"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>Difficulty tier</h3></div><div class="panel-b">${bars(
          Object.entries(s.tier).sort((a, b) => (TIER_ORDER[a[0]] ?? 9) - (TIER_ORDER[b[0]] ?? 9)),
          filterLink("tier"),
        )}</div></div>
        <div class="panel"><div class="panel-h"><h3>Heavy atoms in the target</h3></div><div class="panel-b">${bars(
          Object.entries(s.size).sort((a, b) => SIZE_ORDER[a[0]] - SIZE_ORDER[b[0]]),
          filterLink("size"),
        )}</div></div>
      </div>
    </div>
    <div class="section">
      <h2>Tools and reward</h2>
      <div class="grid">
        <div class="panel"><div class="panel-h"><h3>Tools the policy can call</h3></div><div class="tbl" style="border:0"><table>
          ${META.tools.map((t) => `<tr><td class="mono">${esc(t.name)}</td><td class="sm">${esc(t.description)}${t.oracle ? ` <span class="warn xs"><i class="dot"></i>reads the hidden route</span>` : ""}</td></tr>`).join("")}
        </table></div></div>
        <div class="panel"><div class="panel-h"><h3>Reward components</h3></div><div class="panel-b">${bars(
          Object.entries(META.weights)
            .sort((a, b) => b[1] - a[1])
            .map(([k, w]) => [k, w, w.toFixed(2)]),
        )}</div></div>
      </div>
    </div>`;
  view
    .querySelectorAll("tr[data-split]")
    .forEach((row) => row.addEventListener("click", () => go("/", { split: row.dataset.split })));
  view
    .querySelectorAll("[data-go]")
    .forEach((row) => row.addEventListener("click", () => go("/tasks", JSON.parse(row.dataset.go))));
}

// --- SFT data ---
async function sftView(view, params) {
  const data = await overviewData();
  const datasets = Object.entries(data.datasets);
  if (!datasets.length) {
    view.innerHTML = `<p class="note">No SFT exports found under .local/sft/.</p>`;
    return;
  }
  const ds = data.dataset;
  const p = { steps: "", recovered: "", tool: "", q: "", offset: "0", ...params };
  const rows = state.dataset
    ? await api("sft_rows", { dataset: state.dataset, ...p, limit: 50 })
    : { total: 0, rows: [] };
  const offset = Number(p.offset) || 0;
  view.innerHTML = `
    <div class="tbl"><table>
      <tr><th>SFT dataset</th><th class="num">Rows</th><th class="num">Recover from a rejected cut</th><th class="num">Exact patent route</th><th class="num">Tool calls / row</th><th class="num">Distinct assistant texts</th><th class="num">Tokens p50 / max</th><th class="num">4–5 step rows</th></tr>
      ${datasets
        .map(([name, x]) => {
          const steps = x.steps || {};
          const total = Object.values(steps).reduce((a, b) => a + b, 0) || 1;
          return `<tr class="click${name === state.dataset ? " on" : ""}" data-dataset="${esc(name)}"><td>${esc(name)}</td><td class="num">${fmt(x.rows)}</td><td class="num">${pct(x.recovered)}</td><td class="num">${pct(x.exact)}</td>
        <td class="num">${num(x.calls, 1)}</td><td class="num">${fmt(x.behaviour?.distinct_assistant_texts)}</td><td class="num">${x.tokens ? `${fmt(x.tokens.p50)} / ${fmt(x.tokens.max)}` : "–"}</td>
        <td class="num">${pct(((steps["4"] || 0) + (steps["5"] || 0)) / total)}</td></tr>`;
        })
        .join("")}
    </table></div>
    ${
      ds
        ? `<div class="grid" style="margin-top:16px">
      <div class="panel"><div class="panel-h"><h3>Rows by step budget</h3></div><div class="panel-b">${bars(Object.entries(ds.steps).sort(STEP_ORDER))}</div></div>
      <div class="panel"><div class="panel-h"><h3>Rows that recover from a rejected cut</h3></div><div class="panel-b">${bars(
        Object.entries(ds.recovered_by_steps)
          .sort(STEP_ORDER)
          .map(([k, v]) => [`${k} steps`, v, pct(v)]),
      )}</div></div>
      <div class="panel"><div class="panel-h"><h3>Assistant turns per row</h3></div><div class="panel-b">${bars(
        Object.entries(ds.turns)
          .sort(STEP_ORDER)
          .map(([k, v]) => [k === "12" ? "12+" : k, v]),
      )}</div></div>
      <div class="panel"><div class="panel-h"><h3>Rows that use each tool</h3></div><div class="panel-b">${bars(Object.entries(ds.tools).sort((a, b) => b[1] - a[1]))}</div></div>
    </div>
    <div class="section"><h2>Rows of ${esc(ds.name)}</h2>
      <div class="toolbar">
        <input class="input" id="q" placeholder="Row or task ID" value="${esc(p.q)}">
        <select data-f="steps" aria-label="Steps"><option value="">Steps: any</option>${Object.keys(ds.steps)
          .sort()
          .map((k) => `<option${p.steps === k ? " selected" : ""}>${k}</option>`)
          .join("")}</select>
        <select data-f="recovered" aria-label="Recovery"><option value="">Recovery: any</option><option value="yes"${p.recovered === "yes" ? " selected" : ""}>recovers from a rejected cut</option><option value="no"${p.recovered === "no" ? " selected" : ""}>no rejected cut</option></select>
        <select data-f="tool" aria-label="Tool"><option value="">Uses tool: any</option>${Object.keys(ds.tools)
          .sort()
          .map((k) => `<option${p.tool === k ? " selected" : ""}>${k}</option>`)
          .join("")}</select>
      </div>
      <div class="tbl"><table><tr><th>Row</th><th>Part</th><th class="num">Steps</th><th class="num">Reward</th><th>Outcome</th><th class="num">Turns</th><th class="num">Tool calls</th></tr>
        ${rows.rows
          .map(
            (
              r,
            ) => `<tr class="click" data-task="${esc(r.task_id)}" data-row="${esc(r.id)}"><td class="mono" style="font-size:11.5px">${esc(r.id)}</td><td>${r.part}</td><td class="num">${r.steps}</td><td class="num">${num(r.reward)}</td>
          <td class="sm">${r.exact ? `<span class="ok"><i class="dot"></i>exact patent route</span>` : `<span class="ok"><i class="dot"></i>passed</span>`}${r.recovered ? ` <span class="warn"><i class="dot"></i>recovers</span>` : ""}</td>
          <td class="num">${r.turns}</td><td class="num">${r.calls}</td></tr>`,
          )
          .join("")}
      </table></div>
      <div class="pager">${rows.total ? `${fmt(offset + 1)}–${fmt(Math.min(offset + 50, rows.total))} of ${fmt(rows.total)}` : "No rows match"}
        <button class="btn" id="prev" type="button">Previous</button><button class="btn" id="next" type="button">Next</button></div>
    </div>`
        : `<p class="note" style="margin-top:12px">Pick a dataset to browse its rows.</p>`
    }`;
  view.querySelectorAll("tr[data-dataset]").forEach((row) =>
    row.addEventListener("click", () => {
      setDataset(row.dataset.dataset, false);
      go("/sft");
    }),
  );
  const update = (patch) => go("/sft", { ...p, ...patch, offset: patch.offset ?? 0 });
  view
    .querySelectorAll("select[data-f]")
    .forEach((el) => el.addEventListener("change", () => update({ [el.dataset.f]: el.value })));
  $("#q")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") update({ q: e.target.value.trim() });
  });
  $("#prev")?.addEventListener("click", () => update({ offset: Math.max(0, offset - 50) }));
  $("#next")?.addEventListener("click", () => {
    if (offset + 50 < rows.total) update({ offset: offset + 50 });
  });
  view
    .querySelectorAll("tr[data-row]")
    .forEach((row) =>
      row.addEventListener("click", () =>
        go(`/task/${encodeURIComponent(row.dataset.task)}`, {
          b: (ds.manifest?.guard?.dir || "").split("/").pop() || state.benchmark,
          traj: `sft:${row.dataset.row}`,
        }),
      ),
    );
}

// --- task table ---
async function tasksView(view, params) {
  const options = await overviewData();
  const p = {
    split: "",
    steps: "",
    family: "",
    tier: "",
    size: "",
    routes: "",
    sft: "",
    q: "",
    offset: "0",
    ...params,
  };
  const data = await api("tasks", { benchmark: state.benchmark, dataset: state.dataset, ...p, limit: 50 });
  const all = Object.values(options.splits);
  const keys = (field) => [...new Set(all.flatMap((x) => Object.keys(x[field])))];
  const select = (name, label, values) =>
    `<select data-f="${name}" aria-label="${label}"><option value="">${label}: any</option>${values.map((v) => `<option value="${esc(v)}"${String(p[name]) === String(v) ? " selected" : ""}>${esc(v)}</option>`).join("")}</select>`;
  const offset = Number(p.offset) || 0;
  view.innerHTML = `
    <div class="toolbar">
      <input class="input" id="q" placeholder="Task ID, SMILES or substring" value="${esc(p.q)}">
      ${select("split", "Split", Object.keys(options.splits))}
      ${select(
        "steps",
        "Steps",
        keys("steps").sort((a, b) => a - b),
      )}
      ${select("family", "First reaction", keys("family").sort())}
      ${select("tier", "Tier", ["easy", "medium", "hard"])}
      ${select("size", "Heavy atoms", ["<=20", "21-30", "31-40", "41+"])}
      ${select("routes", "Routes", ["1", "2"])}
      ${
        state.dataset
          ? `<select data-f="sft" aria-label="SFT rows"><option value="">SFT rows: any</option>${[
              ["has", "has rows"],
              ["recovered", "recovers from a rejected cut"],
              ["none", "no rows"],
            ]
              .map(([v, l]) => `<option value="${v}"${p.sft === v ? " selected" : ""}>${l}</option>`)
              .join("")}</select>`
          : ""
      }
    </div>
    <div class="tbl"><table>
      <tr><th></th><th>Task</th><th>Split</th><th class="num">Steps</th><th>First reaction</th><th>Tier</th><th class="num">Heavy atoms</th><th class="num">Routes</th>${data.has_models ? `<th class="num">Models passed</th>` : ""}${state.dataset ? `<th>SFT rows</th>` : ""}</tr>
      ${data.rows
        .map(
          (
            r,
          ) => `<tr class="click" data-id="${esc(r.id)}"><td>${mol(r.smiles, 96, 52, "thumb")}</td><td class="mono">${esc(r.id)}</td><td>${r.split}</td>
        <td class="num">${r.steps}</td><td class="sm">${esc(r.family)}</td><td class="sm">${esc(r.tier || "–")}</td><td class="num">${r.heavy}</td><td class="num">${r.routes}</td>
        ${data.has_models ? `<td class="num">${r.models ? `${r.models[0]} / ${r.models[1]}` : "–"}</td>` : ""}
        ${state.dataset ? `<td class="sm">${r.sft ? `${r.sft}${r.recovered ? ` <span class="warn"><i class="dot"></i>recovers</span>` : ""}` : `<span class="faint">none</span>`}</td>` : ""}</tr>`,
        )
        .join("")}
    </table></div>
    <div class="pager">${data.total ? `${fmt(offset + 1)}–${fmt(Math.min(offset + 50, data.total))} of ${fmt(data.total)}` : "No tasks match"}
      <button class="btn" id="prev" type="button">Previous</button><button class="btn" id="next" type="button">Next</button></div>`;
  const update = (patch) => go("/tasks", { ...p, ...patch, offset: patch.offset ?? 0 });
  view
    .querySelectorAll("select[data-f]")
    .forEach((el) => el.addEventListener("change", () => update({ [el.dataset.f]: el.value })));
  $("#q").addEventListener("keydown", (e) => {
    if (e.key === "Enter") update({ q: e.target.value.trim() });
  });
  $("#prev").addEventListener("click", () => update({ offset: Math.max(0, offset - 50) }));
  $("#next").addEventListener("click", () => {
    if (offset + 50 < data.total) update({ offset: offset + 50 });
  });
  view
    .querySelectorAll("tr[data-id]")
    .forEach((row) => row.addEventListener("click", () => go(`/task/${encodeURIComponent(row.dataset.id)}`)));
}

// --- route trees ---
function referenceTree(route, target) {
  const byProduct = Object.fromEntries(route.steps.map((s) => [s.product, s]));
  const node = (smiles, seen = new Set()) => {
    const step = byProduct[smiles];
    const stock = route.in_stock[smiles];
    const status =
      smiles === target
        ? `<span class="faint">target</span>`
        : step
          ? `<span class="${stock ? "ok" : "warn"}"><i class="dot"></i>${stock ? "in stock, but made in this route" : "made in this route"}</span>`
          : `<span class="${stock ? "ok" : "err"}"><i class="dot"></i>${stock ? "in stock" : "not in stock"}</span>`;
    const head = `<div class="node-head">${mol(smiles, 150, 80)}<div class="meta"><span class="mono" style="font-size:11px">${esc(smiles)}</span>${status}</div></div>`;
    if (!step || seen.has(smiles)) return `<div class="node">${head}</div>`;
    seen.add(smiles);
    return `<div class="node">${head}<div class="rxn">↳ ${esc(step.text)} <span class="faint">· ${esc(step.family)}</span></div><div class="kids">${step.reactants.map((r) => node(r, seen)).join("")}</div></div>`;
  };
  return `<div class="tree">${node(target)}</div>`;
}

function submittedTree(root) {
  const node = (n) => {
    const head = `<div class="node-head">${mol(n.smiles || "", 130, 70)}<div class="meta"><span class="mono" style="font-size:11px">${esc(n.smiles)}</span>
      ${n.children?.length ? `<span class="faint">expanded</span>` : `<span class="${n.in_stock ? "ok" : "warn"}"><i class="dot"></i>${n.in_stock ? "claimed in stock" : "claimed not in stock"}</span>`}</div></div>`;
    const reaction = n.children?.[0];
    if (!reaction) return `<div class="node">${head}</div>`;
    const md = reaction.metadata || {};
    return `<div class="node">${head}<div class="rxn">↳ ${esc(md.explanation || "")} <span class="faint">· ${esc(md.reaction_class || "")}${md.confidence != null ? ` · confidence ${md.confidence}` : ""}</span></div>
      <div class="kids">${(reaction.children || []).map(node).join("")}</div></div>`;
  };
  return `<div class="tree">${node(root)}</div>`;
}

// --- trajectories ---
function callLabel(name, a) {
  const cut = (p, rs) => `${esc(p || "")} → ${(rs || []).map(esc).join(" + ")}`;
  switch (name) {
    case "validate_disconnection":
    case "reaction_class_lookup":
    case "reaction_conditions_search":
      return cut(a.product_smiles, a.reactants);
    case "stock_retrieve":
      return `${esc(a.query)} <span class="faint">(${esc(a.mode || "auto")})</span>`;
    case "reaction_precedent_search":
    case "search_literature":
      return `${esc(a.product_smiles || "")}${a.reaction_class ? ` · class ${esc(a.reaction_class)}` : ""}`;
    case "inspect_molecule":
      return esc(a.smiles);
    case "pubchem_lookup":
      return esc(a.query);
    case "emit_routes": {
      const routes = a.submission?.routes;
      return Array.isArray(routes) ? `${routes.length} route tree${routes.length === 1 ? "" : "s"}` : "submission";
    }
    default:
      return esc(JSON.stringify(a).slice(0, 160));
  }
}

function resultLine(name, out) {
  if (out.error) return `<li class="err"><i class="dot"></i>${esc(out.error)}</li>`;
  const left =
    out.model_turns_remaining != null ? ` <span class="faint">· ${out.model_turns_remaining} turns left</span>` : "";
  switch (name) {
    case "validate_disconnection":
      return out.valid
        ? `<li class="ok"><i class="dot"></i>Cut supported <span class="faint mono">${esc(out.support)}</span>${left}</li>`
        : `<li class="warn"><i class="dot"></i>Cut not supported <span class="faint mono">${esc(out.support)}</span> <span class="faint">${esc((out.errors || [])[0] || "")}</span>${left}</li>`;
    case "stock_retrieve":
      if ((out.mode || "exact") !== "exact")
        return `<li><i class="dot"></i>${fmt(out.returned)} results for ${esc(out.query)} (${esc(out.mode)})${left}</li>`;
      return out.results?.length
        ? `<li class="ok"><i class="dot"></i>In stock: <span class="mono">${esc(out.query)}</span>${left}</li>`
        : `<li class="warn"><i class="dot"></i>Not in stock: <span class="mono">${esc(out.query)}</span>${left}</li>`;
    case "reaction_precedent_search": {
      const best = out.results?.[0];
      return `<li><i class="dot"></i>${fmt(out.returned)} precedents${best ? `, closest Tanimoto ${num(best.similarity, 2)}${best.literature?.[0] ? ` (${esc(best.literature[0].identifier)})` : ""}` : ""}${left}</li>`;
    }
    case "inspect_molecule":
      return `<li><i class="dot"></i>${esc(out.formula)} · ${out.rings} rings · ${out.chiral_centres} stereocentres · ${num(out.molecular_weight, 1)} g/mol${left}</li>`;
    case "emit_routes":
      return `<li class="${out.score?.valid ? "ok" : "warn"}"><i class="dot"></i>Scored ${num(out.score?.reward)} · ${out.score?.valid ? "passed" : "did not pass"}</li>`;
    case "reaction_class_lookup":
      return `<li><i class="dot"></i>Class: ${esc(out.reaction_class)}${left}</li>`;
    case "reaction_conditions_search":
      return `<li><i class="dot"></i>${fmt((out.conditions || []).length)} conditions (${esc(out.source || "")})${left}</li>`;
    case "search_literature":
      return `<li><i class="dot"></i>${fmt(out.returned)} citations${left}</li>`;
    default:
      return `<li><i class="dot"></i>${esc(JSON.stringify(out).slice(0, 160))}${left}</li>`;
  }
}

function parse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return { result: text };
  }
}

function trajectoryHtml(messages, { trained }) {
  const calls = {};
  messages.forEach((m) =>
    (m.tool_calls || []).forEach((c) => {
      calls[c.id] = {
        name: c.function.name,
        args: parse(
          typeof c.function.arguments === "string" ? c.function.arguments : JSON.stringify(c.function.arguments),
        ),
      };
    }),
  );
  const out = [];
  let firstUser = true;
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m.role === "system") {
      out.push(`<details class="msg ctx"><summary>System prompt</summary><pre>${esc(m.content)}</pre></details>`);
      continue;
    }
    if (m.role === "user") {
      out.push(
        firstUser
          ? `<details class="msg ctx"><summary>Task prompt</summary><pre>${esc(m.content)}</pre></details>`
          : `<div class="msg ctx"><div class="role">user <span>· harness message</span></div><p>${esc(m.content)}</p></div>`,
      );
      firstUser = false;
      continue;
    }
    if (m.role === "assistant") {
      const items = (m.tool_calls || [])
        .map(
          (c) =>
            `<li><span class="fn">${esc(c.function.name)}</span>${callLabel(c.function.name, calls[c.id].args)}</li>`,
        )
        .join("");
      const emit = (m.tool_calls || []).find((c) => c.function.name === "emit_routes");
      const routes = emit ? calls[emit.id].args.submission?.routes : null;
      out.push(`<div class="msg asst"><div class="role">assistant${trained ? " <span>· trained on</span>" : ""}</div>
        ${m.content ? `<p>${esc(m.content)}</p>` : `<p class="faint">(no text)</p>`}<ol class="calls">${items}</ol>
        ${Array.isArray(routes) && routes.length ? `<details><summary>Submitted route${routes.length > 1 ? "s" : ""}</summary><div class="stack" style="margin-top:8px">${routes.map(submittedTree).join("")}</div></details>` : ""}
        <details><summary>Raw</summary><pre>${esc(JSON.stringify(m, (k, v) => (k === "arguments" && typeof v === "string" ? parse(v) : v), 2))}</pre></details></div>`);
      const results = [];
      while (messages[i + 1]?.role === "tool") results.push(messages[++i]);
      if (results.length) {
        out.push(`<div class="msg ctx"><div class="role">tool results${trained ? " <span>· context only</span>" : ""}</div><ul class="calls">${results.map((r) => resultLine(calls[r.tool_call_id]?.name || r.name, parse(r.content))).join("")}</ul>
          <details><summary>Raw</summary><pre>${esc(
            JSON.stringify(
              results.map((r) => ({ ...r, content: parse(r.content) })),
              null,
              2,
            ),
          )}</pre></details></div>`);
      }
      continue;
    }
    if (m.role === "tool")
      out.push(`<div class="msg ctx"><ul class="calls">${resultLine(m.name, parse(m.content))}</ul></div>`);
  }
  return `<div class="turns">${out.join("")}</div>`;
}

// --- task page: observation, hidden answer key, a live session, then trajectories ---
const TEMPLATES = {
  inspect_molecule: (t) => ({ smiles: t }),
  pubchem_lookup: (t) => ({ query: t }),
  stock_retrieve: (t) => ({ query: t, mode: "exact", limit: 5 }),
  reaction_precedent_search: (t) => ({ product_smiles: t, limit: 5 }),
  validate_disconnection: (t) => ({ product_smiles: t, reactants: ["", ""] }),
  reaction_class_lookup: (t) => ({ product_smiles: t, reactants: ["", ""] }),
  reaction_conditions_search: (t) => ({ product_smiles: t, reactants: ["", ""], limit: 3 }),
  search_literature: (t) => ({ product_smiles: t, limit: 5 }),
  emit_routes: () => ({ submission: { routes: [] } }),
};

async function taskView(view, id, params) {
  if (params.b && params.b !== state.benchmark) setBenchmark(params.b, false);
  const t = await api("task", { benchmark: state.benchmark, id, reveal: params.reveal || 0, dataset: state.dataset });
  const r = t.row,
    d = t.difficulty || {};
  const trajectories = [
    ...t.runs.map((x) => ({
      key: `run:${x.run}:${x.file}`,
      kind: "model run",
      source: x.label,
      label: `attempt ${x.attempt}`,
      reward: x.reward,
      exact: x.exact,
      valid: x.valid,
      turns: x.turns,
      calls: x.calls,
      refused: x.refused,
    })),
    ...t.sft.map((x) => ({
      key: `sft:${x.id}`,
      kind: "SFT row",
      source: state.dataset,
      label: x.id.split(":").slice(1).join(":"),
      reward: x.reward,
      exact: x.exact,
      recovered: x.recovered,
      turns: x.turns,
      calls: x.calls,
    })),
  ];
  const selected = params.traj || "";
  const canReveal = !t.hidden;
  view.innerHTML = `
    <div class="crumbs"><a href="#/tasks">RL tasks</a><span>/</span><span class="mono">${esc(id)}</span></div>
    <div class="split">
      <div class="panel"><div class="panel-h"><h2>What the policy sees</h2><span class="grow"></span><span class="faint sm mono">${esc(id)}</span></div>
        <div class="panel-b">${mol(r.smiles, 420, 210)}
          <dl class="kv" style="margin-top:12px">
            <dt>Target</dt><dd class="mono">${esc(r.smiles)}</dd>
            <dt>Split</dt><dd>${esc(r.split)}</dd>
            <dt>Step budget</dt><dd>${t.task.max_steps} reactions per route</dd>
            <dt>Routes asked for</dt><dd>${t.task.min_routes === t.task.max_routes ? t.task.max_routes : `${t.task.min_routes} to ${t.task.max_routes}`}</dd>
            <dt>Budget</dt><dd>16 model turns, 32 tool calls</dd>
            <dt>Stock</dt><dd>reachable only through stock_retrieve${t.target_in_stock ? " (the target itself is in stock)" : ""}</dd>
          </dl>
          <details style="margin-top:10px"><summary>Exact prompt</summary><pre style="margin-top:6px">${esc(t.prompt)}</pre></details>
          <h3 style="margin:14px 0 6px">Difficulty <span class="faint sm">(not shown to the policy)</span></h3>
          <dl class="kv">
            <dt>Tier</dt><dd>${esc(d.tier || "–")}${
              d.points
                ? ` <span class="faint">(${
                    Object.entries(d.points)
                      .filter(([, v]) => v)
                      .map(([k]) => k.replaceAll("_", " "))
                      .join(", ") || "no points"
                  })</span>`
                : ""
            }</dd>
            <dt>Heavy atoms</dt><dd>${r.heavy}${r.stereo ? " · has stereocentres" : ""}</dd>
            <dt>First reaction</dt><dd>${esc(r.family)}</dd>
            <dt>Closest train target</dt><dd>${d.nn_train_similarity != null ? `Tanimoto ${num(d.nn_train_similarity, 2)}` : "–"}</dd>
            <dt>Rarest leaf</dt><dd>${d.min_leaf_archive_routes != null ? `in ${fmt(d.min_leaf_archive_routes)} archive routes` : "–"}</dd>
          </dl>
        </div></div>
      <div class="panel"><div class="panel-h"><h2>Answer key, hidden from the policy</h2><span class="grow"></span>
          ${t.routes?.[0]?.source?.[0]?.group_id ? `<span class="faint sm">patent ${esc(t.routes[0].source[0].group_id)}</span>` : ""}</div>
        <div class="panel-b">${
          t.hidden
            ? `<div class="note">This is ${/^[aeiou]/.test(r.split) ? "an" : "a"} ${esc(r.split)} task, so its answer stays hidden by default. <button class="link" id="reveal" type="button">Show it</button></div>`
            : `<p class="faint sm" style="margin-bottom:10px">The verifier scores a submission against these reactions. <span class="mono">validate_disconnection</span> and <span class="mono">reaction_class_lookup</span> answer from them too.</p>` +
              t.routes
                .map(
                  (route, i) =>
                    `${t.routes.length > 1 ? `<h3 style="margin:${i ? "16px" : "0"} 0 8px">Route ${i + 1}</h3>` : ""}${referenceTree(route, r.smiles)}`,
                )
                .join("")
        }</div></div>
    </div>

    <div class="section">
      <h2>Try it <span class="faint sm">· play this task through the same core session the server runs</span></h2>
      <div class="panel" id="play"><div class="panel-b"><div class="toolbar" style="margin:0">
        <select id="toolset" aria-label="Toolset"><option value="full">full toolset</option><option value="unaided">unaided (no answer-reading tools)</option></select>
        <button class="btn" id="start" type="button">Start an episode</button>
        <span class="faint sm">The first start on a benchmark builds its precedent index, about 30 seconds for v3.</span></div></div></div>
    </div>

    <div class="section">
      <h2>How it went for models and SFT rows</h2>
      ${
        trajectories.length
          ? `<div class="tbl"><table>
        <tr><th>Kind</th><th>Source</th><th>Episode</th><th class="num">Reward</th><th>Outcome</th><th class="num">Turns</th><th class="num">Tool calls</th></tr>
        ${trajectories
          .map(
            (
              x,
            ) => `<tr class="click${x.key === selected ? " on" : ""}" data-traj="${esc(x.key)}"><td class="sm muted">${x.kind}</td><td>${esc(x.source)}</td><td class="mono" style="font-size:11.5px">${esc(x.label)}</td><td class="num">${num(x.reward)}</td>
          <td class="sm">${x.refused ? `<span class="err"><i class="dot"></i>refused</span>` : x.exact ? `<span class="ok"><i class="dot"></i>exact patent route</span>` : x.valid === false ? `<span class="warn"><i class="dot"></i>did not pass</span>` : `<span class="ok"><i class="dot"></i>passed</span>`}${x.recovered ? ` <span class="warn"><i class="dot"></i>recovers from a rejected cut</span>` : ""}</td>
          <td class="num">${fmt(x.turns)}</td><td class="num">${fmt(x.calls)}</td></tr>`,
          )
          .join("")}
      </table></div><div id="traj" style="margin-top:14px"></div>`
          : `<p class="note">No model runs or SFT rows cover this task yet.</p>`
      }
    </div>`;
  $("#reveal")?.addEventListener("click", () => go(`/task/${encodeURIComponent(id)}`, { ...params, reveal: 1 }));
  view
    .querySelectorAll("tr[data-traj]")
    .forEach((row) =>
      row.addEventListener("click", () => go(`/task/${encodeURIComponent(id)}`, { ...params, traj: row.dataset.traj })),
    );
  $("#start").addEventListener("click", () =>
    startPlay($("#play"), id, r.smiles, $("#toolset").value, canReveal, params.reveal || 0),
  );
  if (selected) await showTrajectory($("#traj"), selected);
}

function rewardHtml(result) {
  const score = result?.score;
  if (!score) return "";
  const parts = Object.entries(score.components || {});
  return `<div class="panel" style="margin-top:10px"><div class="panel-h"><h3>Reward ${num(score.reward)}</h3><span class="${score.valid ? "ok" : "warn"} sm"><i class="dot"></i>${score.valid ? "passed" : "did not pass"}</span>
      <span class="faint sm">${esc(score.verification_tier || "")}</span></div>
    <div class="panel-b">${bars(parts.map(([k, v]) => [`${k} × ${num(META.weights[k], 2)}`, v, num(v, 2)]))}
      ${(score.hard_failures || []).length ? `<ul class="calls" style="margin-top:10px">${score.hard_failures.map((f) => `<li class="warn"><i class="dot"></i>${esc(f)}</li>`).join("")}</ul>` : ""}</div></div>`;
}

async function startPlay(panel, id, target, toolset, canReveal, reveal) {
  panel.innerHTML = `<div class="panel-b"><div class="loading-row"><span class="spinner"></span>Starting the episode</div></div>`;
  let session;
  try {
    session = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ benchmark: state.benchmark, task: id, toolset }),
    }).then((r) => r.json());
  } catch (error) {
    panel.innerHTML = `<div class="panel-b note err">${esc(error.message)}</div>`;
    return;
  }
  const tools = session.tools.map((tool) => tool.function.name);
  const cuts = await api("disconnections", { smi: target });
  panel.innerHTML = `
    <div class="panel-h"><h3>Episode on ${esc(id)}</h3><span class="grow"></span><span class="faint sm" id="budget">32 tool calls left</span></div>
    <div class="panel-b">
      <div class="split" style="grid-template-columns:minmax(0,1fr) minmax(0,1fr)">
        <div>
          <div class="toolbar" style="margin-bottom:8px"><select id="tool" aria-label="Tool">${tools.map((name) => `<option>${name}</option>`).join("")}</select>
            <button class="btn" id="call" type="button">Call</button>
            ${canReveal ? `<button class="btn" id="answer" type="button">Submit the answer key</button>` : ""}
            <button class="btn" id="empty" type="button">Submit an empty route</button></div>
          <textarea id="args" class="input" spellcheck="false" style="width:100%;height:120px;padding:8px 10px;font-family:var(--mono);font-size:12px"></textarea>
          <p class="faint xs" style="margin-top:4px">Arguments as JSON. Pick a tool to load a template.</p>
        </div>
        <div>
          <h3 style="margin-bottom:6px">Candidate cuts of the target <span class="faint sm">(rule library)</span></h3>
          ${cuts.length ? `<ul class="calls">${cuts.map((c, i) => `<li><button class="link" data-cut="${i}" type="button">${tools.includes("validate_disconnection") ? "validate" : "use"}</button> ${esc(c.text)} <span class="faint">· ${esc(c.bond)}</span></li>`).join("")}</ul>` : `<p class="faint sm">No rule matches this target.</p>`}
        </div>
      </div>
      <div class="turns" id="log" style="margin-top:12px"></div>
    </div>`;
  const args = $("#args", panel),
    tool = $("#tool", panel),
    log = $("#log", panel);
  const fill = () => {
    args.value = JSON.stringify((TEMPLATES[tool.value] || (() => ({})))(target), null, 2);
  };
  tool.addEventListener("change", fill);
  fill();
  const show = (call, outcome) => {
    const result = outcome.result || {};
    log.insertAdjacentHTML(
      "beforeend",
      `<div class="msg asst"><div class="role">you</div><ol class="calls"><li><span class="fn">${esc(call.tool)}</span>${callLabel(call.tool, call.arguments)}</li></ol>
      ${call.tool === "emit_routes" && call.arguments.submission?.routes?.length ? `<details><summary>Submitted route</summary><div class="stack" style="margin-top:8px">${call.arguments.submission.routes.map(submittedTree).join("")}</div></details>` : ""}</div>
      <div class="msg ctx"><div class="role">environment</div><ul class="calls">${resultLine(call.tool, result)}</ul>
      <details><summary>Raw</summary><pre>${esc(JSON.stringify(result, null, 2))}</pre></details>${call.tool === "emit_routes" ? rewardHtml(result) : ""}</div>`,
    );
    if (outcome.tool_calls_remaining != null)
      $("#budget", panel).textContent =
        `${outcome.tool_calls_remaining} tool calls left${outcome.done ? " · episode over" : ""}`;
    hydrate(log);
  };
  const post = async (path, body) =>
    (
      await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })
    ).json();
  $("#call", panel).addEventListener("click", async () => {
    let parsed;
    try {
      parsed = JSON.parse(args.value || "{}");
    } catch (error) {
      log.insertAdjacentHTML(
        "beforeend",
        `<p class="note err">Arguments are not valid JSON: ${esc(error.message)}</p>`,
      );
      return;
    }
    show(
      { tool: tool.value, arguments: parsed },
      await post(`/api/session/${session.session}/call`, { tool: tool.value, arguments: parsed }),
    );
  });
  $("#empty", panel).addEventListener("click", async () => {
    const call = { tool: "emit_routes", arguments: { submission: { routes: [] } } };
    show(call, await post(`/api/session/${session.session}/call`, call));
  });
  $("#answer", panel)?.addEventListener("click", async () => {
    const outcome = await post(`/api/session/${session.session}/reference?reveal=${reveal}`, {});
    show({ tool: "emit_routes", arguments: outcome.arguments || {} }, outcome);
  });
  panel.querySelectorAll("[data-cut]").forEach((button) =>
    button.addEventListener("click", async () => {
      const cut = cuts[Number(button.dataset.cut)];
      const name = tools.includes("validate_disconnection") ? "validate_disconnection" : "stock_retrieve";
      if (name === "stock_retrieve") {
        for (const piece of cut.reactants) {
          const call = { tool: name, arguments: { query: piece, mode: "exact", limit: 1 } };
          show(call, await post(`/api/session/${session.session}/call`, call));
        }
        return;
      }
      const call = { tool: name, arguments: { product_smiles: target, reactants: cut.reactants } };
      show(call, await post(`/api/session/${session.session}/call`, call));
    }),
  );
}

async function showTrajectory(el, key) {
  el.innerHTML = `<div class="loading-row"><span class="spinner"></span>Loading trajectory</div>`;
  if (key.startsWith("sft:")) {
    const data = await api("sft_row", { dataset: state.dataset, id: key.slice(4) });
    const m = data.meta;
    el.innerHTML = `<div class="chips" style="margin-bottom:10px"><span>Row <span class="mono">${esc(m.id)}</span></span><span>${m.part} split of the SFT data</span>
      <span>${data.messages.length} messages</span><span>${m.turns} assistant turns</span><span>tools: ${m.tools.map(esc).join(", ")}</span></div>${trajectoryHtml(data.messages, { trained: true })}`;
  } else {
    const [, run, file] = key.match(/^run:(.+):([^:]+)$/);
    const data = await api("episode", { run, file });
    const s = data.score;
    el.innerHTML = `<div class="chips" style="margin-bottom:10px"><span>${esc(run)}</span><span>reward ${num(s.reward)}</span><span>${s.valid ? "passed" : "did not pass"}</span>
      ${s.exact_match ? "<span>exact patent route</span>" : ""}${s.usage?.cost_usd != null ? `<span>$${num(s.usage.cost_usd, 4)}</span>` : ""}
      ${(s.hard_failures || []).length ? `<span class="warn">${esc(s.hard_failures[0])}</span>` : ""}${(s.errors || []).length ? `<span class="err">${esc(String(s.errors[0]).slice(0, 140))}</span>` : ""}</div>
      ${trajectoryHtml(data.messages, { trained: false })}`;
  }
  hydrate(el);
}

// --- runs ---
async function runsView(view) {
  const runs = META.runs;
  view.innerHTML = `<div class="section" style="margin-top:0"><h2>Model runs</h2><p class="faint sm" style="margin-bottom:10px">Evaluation runs found under runs/. Pick one to list its episodes.</p>
    <div class="tbl"><table><tr><th>Model</th><th>Run</th><th>Benchmark</th><th>Split</th><th>Toolset</th><th class="num">Tasks</th><th class="num">Pass@1</th><th class="num">Exact</th><th class="num">Reward</th><th class="num">Cost</th></tr>
    ${runs
      .map(
        (
          r,
        ) => `<tr class="click" data-run="${esc(r.name)}"><td>${esc(r.label)}</td><td class="mono" style="font-size:11.5px">${esc(r.name)}</td><td>${esc(r.benchmark || "–")}</td><td>${esc(r.split)}</td><td>${esc(r.toolset || "–")}</td>
      <td class="num">${fmt(r.tasks)}</td><td class="num">${num(r.pass)}</td><td class="num">${num(r.exact)}</td><td class="num">${num(r.reward)}</td><td class="num">${r.cost != null ? `$${num(r.cost, 2)}` : "–"}</td></tr>`,
      )
      .join("")}
    </table></div></div>`;
  view
    .querySelectorAll("tr[data-run]")
    .forEach((row) => row.addEventListener("click", () => go(`/run/${encodeURIComponent(row.dataset.run)}`)));
}

async function runView(view, name, params) {
  const data = await api("run", { run: name });
  const filter = params.show || "";
  const rows = data.episodes.filter(
    (e) => !filter || (filter === "passed" ? e.valid : filter === "failed" ? !e.valid : e.refused),
  );
  view.innerHTML = `<div class="crumbs"><a href="#/runs">Runs</a><span>/</span><span class="mono">${esc(name)}</span></div>
    <div class="toolbar"><h2 style="margin-right:12px">${esc(data.summary.label)}</h2>
      <select id="show" aria-label="Episodes"><option value="">All episodes</option>${[
        ["passed", "Passed"],
        ["failed", "Did not pass"],
        ["refused", "Refused"],
      ]
        .map(([v, l]) => `<option value="${v}"${filter === v ? " selected" : ""}>${l}</option>`)
        .join("")}</select>
      <span class="faint sm">${fmt(rows.length)} of ${fmt(data.episodes.length)} episodes · pass@1 ${num(data.summary.pass)} · on ${esc(data.summary.benchmark || "unknown benchmark")}</span></div>
    <div class="tbl"><table><tr><th>Task</th><th class="num">Attempt</th><th class="num">Reward</th><th>Outcome</th><th class="num">Turns</th><th class="num">Tool calls</th></tr>
      ${rows
        .map(
          (
            e,
          ) => `<tr class="click" data-task="${esc(e.task_id)}" data-file="${esc(e.file)}"><td class="mono">${esc(e.task_id)}</td><td class="num">${e.attempt}</td><td class="num">${num(e.reward)}</td>
        <td class="sm">${e.refused ? `<span class="err"><i class="dot"></i>refused</span>` : e.exact ? `<span class="ok"><i class="dot"></i>exact patent route</span>` : e.valid ? `<span class="ok"><i class="dot"></i>passed</span>` : `<span class="warn"><i class="dot"></i>did not pass</span>`}</td>
        <td class="num">${fmt(e.turns)}</td><td class="num">${fmt(e.calls)}</td></tr>`,
        )
        .join("")}
    </table></div>`;
  $("#show").addEventListener("change", (e) => go(`/run/${encodeURIComponent(name)}`, { show: e.target.value }));
  view
    .querySelectorAll("tr[data-task]")
    .forEach((row) =>
      row.addEventListener("click", () =>
        go(`/task/${encodeURIComponent(row.dataset.task)}`, {
          b: data.summary.benchmark,
          traj: `run:${name}:${row.dataset.file}`,
        }),
      ),
    );
}

// --- shell ---
function setBenchmark(name, reroute = true) {
  state.benchmark = name;
  store.set("benchmark", name);
  $("#benchmark").value = name;
  if (reroute) route();
}
function setDataset(name, reroute = true) {
  state.dataset = name;
  store.set("dataset", name);
  $("#dataset").value = name;
  if (reroute) route();
}

async function route() {
  const [path, query] = location.hash.replace(/^#/, "").split("?");
  const parts = (path || "/").split("/").filter(Boolean);
  const params = Object.fromEntries(new URLSearchParams(query || ""));
  const view = $("#view");
  const active = parts[0] === "task" ? "tasks" : parts[0] === "run" ? "runs" : parts[0] || "overview";
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === active));
  view.innerHTML = `<div class="loading-row"><span class="spinner"></span>Loading</div>`;
  try {
    if (!parts.length) await overviewView(view, params);
    else if (parts[0] === "tasks") await tasksView(view, params);
    else if (parts[0] === "sft") await sftView(view, params);
    else if (parts[0] === "task") await taskView(view, decodeURIComponent(parts[1]), params);
    else if (parts[0] === "runs") await runsView(view, params);
    else if (parts[0] === "run") await runView(view, decodeURIComponent(parts.slice(1).join("/")), params);
    else view.innerHTML = `<p class="note">Nothing here. <a class="link" href="#/">Back to the overview</a></p>`;
  } catch (error) {
    view.innerHTML = `<p class="note err">${esc(error.message)}</p>`;
  }
  hydrate(view);
  window.scrollTo(0, 0);
}

async function boot() {
  META = await api("meta");
  const benchmarks = META.benchmarks.map((b) => b.name);
  const datasets = META.datasets.map((d) => d.name);
  $("#benchmark").innerHTML = benchmarks.map((b) => `<option>${esc(b)}</option>`).join("");
  $("#dataset").innerHTML =
    `<option value="">none</option>` + datasets.map((d) => `<option>${esc(d)}</option>`).join("");
  const savedB = store.get("benchmark", "");
  state.benchmark = benchmarks.includes(savedB)
    ? savedB
    : benchmarks.includes("retroeval-v3")
      ? "retroeval-v3"
      : benchmarks[0];
  const savedD = store.get("dataset", null);
  state.dataset =
    savedD !== null && (savedD === "" || datasets.includes(savedD))
      ? savedD
      : datasets.includes("chemist")
        ? "chemist"
        : datasets[0] || "";
  $("#benchmark").value = state.benchmark;
  $("#dataset").value = state.dataset;
  $("#benchmark").addEventListener("change", (e) => setBenchmark(e.target.value));
  $("#dataset").addEventListener("change", (e) => setDataset(e.target.value));
  window.addEventListener("hashchange", route);
  route();
}
boot();
