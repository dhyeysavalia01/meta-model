const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = v => String(v ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const st = { task: null, sid: null, charts: [], step: 1 };

const BLUE = "#3848E0", TEAL = "#0F9D8A", AMBER = "#F6B51E", ROSE = "#E0476B";
const SERIES = [BLUE, TEAL, AMBER, ROSE, "#7B5CE6", "#0EA5E9", "#84CC16", "#F97316", "#64748B", "#DB2777"];

if (window.Chart) {
  Chart.defaults.font.family = '"Bricolage Grotesque", system-ui, sans-serif';
  Chart.defaults.color = "#62668A";
  Chart.defaults.borderColor = "#ECEEF6";
  Object.assign(Chart.defaults.plugins.tooltip, { backgroundColor: "#1A1D3A", cornerRadius: 8, padding: 10 });
}

const NICE = {
  n_samples: "Number of rows", log_n_samples: "Rows (log scale)", n_features: "Number of features",
  n_num: "Numerical features", pct_cat: "Share of categorical features", pct_missing: "Share of missing values",
  dim_ratio: "Features per row", mean_skew: "Average skewness", mean_kurt: "Average kurtosis",
  mean_abs_corr: "Average feature correlation", n_classes: "Number of classes", class_entropy: "Class entropy",
  imbalance_ratio: "Class imbalance ratio", target_skew: "Target skewness", target_kurt: "Target kurtosis",
  target_cv: "Target variation", lm_stump: "Decision stump score", lm_1nn: "1-nearest-neighbour score",
  lm_linear: "Linear model score",
};
const fmt = v => v === null || v === undefined ? "n/a" : Number.isInteger(v) ? v.toLocaleString() : String(v);

/* ---------- helpers ---------- */
const busy = msg => { const s = $("#status"); s.textContent = msg || ""; s.classList.toggle("on", !!msg); };
const fail = (n, m) => { $("#e" + n).innerHTML = m ? `<div class="alert" role="alert">${esc(m)}</div>` : ""; };
const go = n => $("#s" + n).scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
const sum = (i, t) => { $(`#steps li[data-step="${i}"] small`).textContent = t; };

async function api(path, opts) {
  const r = await fetch(path, opts);
  let data = {};
  try { data = await r.json(); } catch (e) {}
  if (!r.ok) {
    const d = data.detail;
    if (d && d.errors) throw new Error(d.errors.join(" "));
    if (Array.isArray(d)) throw new Error(d.map(e => e.msg).join("; "));
    throw new Error(typeof d === "string" ? d : `Server error (${r.status}). Check the uvicorn terminal for details.`);
  }
  return data;
}
const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

async function run(btn, msg, fn) {
  btn.classList.add("loading"); btn.disabled = true; busy(msg);
  try { await fn(); } finally { btn.classList.remove("loading"); btn.disabled = false; busy(); }
}

function setStep(n, scroll = true) {
  st.step = n;
  $$(".step").forEach(s => s.classList.toggle("hidden", +s.dataset.step > n));
  $$("#steps li").forEach(li => {
    const i = +li.dataset.step;
    li.classList.toggle("done", i < n);
    li.classList.toggle("active", i === n);
    li.classList.toggle("locked", i > n);
    li.querySelector(".dot").textContent = i < n ? "✓" : i;
    li.querySelector("button").disabled = i > n;
    if (i > n) li.querySelector("small").textContent = "";
  });
  if (scroll) go(n);
}
$$("#steps button").forEach(b => b.onclick = () => go(+b.closest("li").dataset.step));

/* ---------- html builders ---------- */
const stats = items => `<div class="stats">${items.map(([v, c, k]) =>
  `<div class="${k || ""}"><b>${esc(v)}</b><span>${esc(c)}</span></div>`).join("")}</div>`;
const tags = (arr, cls) => arr.length
  ? `<div class="tags">${arr.map(c => `<span class="tag ${cls}">${esc(c)}</span>`).join("")}</div>`
  : `<p class="note">None</p>`;
const table = (cols, rows, cls = "") => `<div class="tablewrap"><table class="tbl ${cls}"><thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>
  <tbody>${rows.map(r => `<tr>${r.map(v => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
const barRow = (label, frac, val, tone = "") => {
  const w = frac > 0 ? Math.max(2, Math.min(1, frac) * 100) : 0;
  return `<div class="brow"><span class="bl" title="${esc(label)}">${esc(label)}</span><span class="bar ${tone}"><i style="width:${w}%"></i></span><span class="bv">${esc(val)}</span></div>`;
};
const featBars = fi => {
  const mx = Math.max(...fi.map(f => Math.abs(f.importance)), 1e-9);
  return `<div class="bars">${fi.map(f => barRow(f.feature, Math.max(0, f.importance) / mx, f.importance.toFixed(3))).join("")}</div>`;
};
const panel = (id, title, cap, span) =>
  `<div class="panel ${span}"><h3>${title}</h3><p class="cap">${cap}</p><div class="cv"><canvas id="${id}"></canvas></div></div>`;

/* ---------- chart helpers ---------- */
const opts = (x, y) => ({
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: { x: { title: { display: true, text: x } }, y: { title: { display: true, text: y } } },
});
const chart = (id, cfg) => { if (window.Chart) st.charts.push(new Chart($("#" + id), cfg)); };

/* ---------- step 1: task ---------- */
$$(".choice").forEach(b => b.onclick = () => {
  if (st.task === b.dataset.task && st.step >= 2) return;
  st.task = b.dataset.task;
  $$(".choice").forEach(x => x.setAttribute("aria-checked", x === b));
  sum(1, st.task === "regression" ? "Regression" : "Classification");
  setStep(2);
});

/* ---------- step 2: upload ---------- */
const fileEl = $("#file");
function picked() {
  const f = fileEl.files[0];
  $("#drop").classList.toggle("has", !!f);
  $("#dropTitle").textContent = f ? f.name : "Drop a file here or browse";
  $("#dropSub").textContent = f ? `${Math.max(1, Math.round(f.size / 1024))} KB. Click to replace.` : "CSV, XLSX or XLS";
  $("#btnUpload").disabled = !f;
}
fileEl.onchange = picked;
const drop = $("#drop");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files.length) { fileEl.files = e.dataTransfer.files; picked(); } };

function renderInfo(d) {
  const n = d.n_samples, miss = Object.entries(d.missing_by_column);
  $("#info").innerHTML =
    stats([[n.toLocaleString(), "rows"], [d.n_features, "columns"],
           [d.missing_total.toLocaleString(), "missing values", d.missing_total ? "bad" : "good"]]) +
    `<h3>Column types</h3>
     <p class="sub">Numerical (${d.numerical.length})</p>${tags(d.numerical, "num")}
     <p class="sub">Categorical (${d.categorical.length})</p>${tags(d.categorical, "cat")}
     <h3>Missing values</h3>` +
    (miss.length
      ? `<div class="bars">${miss.map(([c, v]) => barRow(c, v / n, `${v} (${(v / n * 100).toFixed(1)}%)`, "rose")).join("")}</div>`
      : `<p class="ok">No gaps found.</p>`) +
    `<h3>Preview</h3>` + table(d.columns, d.preview.map(r => d.columns.map(c => r[c])));
}

$("#btnUpload").onclick = e => run(e.currentTarget, "Uploading dataset…", async () => {
  fail(2);
  const f = fileEl.files[0];
  const fd = new FormData();
  fd.append("file", f); fd.append("task", st.task);
  try {
    const d = await api("/upload", { method: "POST", body: fd });
    st.sid = d.session_id;
    renderInfo(d);
    $("#preInfo").innerHTML = "";
    sum(2, `${f.name}, ${d.n_samples.toLocaleString()} rows`);
    setStep(3, false);
    go(3);
  } catch (err) { fail(2, err.message); }
});

/* ---------- step 3: missing values ---------- */
$$("input[name=cat]").forEach(r => r.onchange = () => { $("#constant").hidden = $("input[name=cat]:checked").value !== "constant"; });

$("#btnPre").onclick = e => run(e.currentTarget, "Filling missing values…", async () => {
  fail(3);
  try {
    const d = await post("/preprocess", {
      session_id: st.sid,
      numeric_strategy: $("input[name=num]:checked").value,
      categorical_strategy: $("input[name=cat]:checked").value,
      constant: $("#constant").value.trim() || "Unknown",
    });
    $("#preInfo").innerHTML = `<p class="ok">Done. ${d.missing_total} missing values remain across ${d.n_samples.toLocaleString()} rows and ${d.n_features} columns.</p>`;
    $("#target").innerHTML = d.columns.map(c => `<option>${esc(c)}</option>`).join("");
    $("#target").value = d.columns[d.columns.length - 1];
    sum(3, `${$("input[name=num]:checked + span").textContent} / ${$("input[name=cat]:checked + span").textContent}`);
    fail(4);
    setStep(4);
  } catch (err) { fail(3, err.message); }
});

/* ---------- step 4: recommend ---------- */
$("#btnRec").onclick = e => run(e.currentTarget, "Profiling your data and ranking models…", async () => {
  fail(4); fail(5);
  setStep(4, false);
  try {
    const d = await post("/recommend", { session_id: st.sid, target: $("#target").value });
    sum(4, $("#target").value);
    sum(5, d.recommended);
    renderRec(d);
    setStep(5);
    await validate(d.recommended, false);
  } catch (err) { fail(4, err.message); }
});

function renderRec(d) {
  const pct = Math.round(d.confidence * 100);
  const maxP = Math.max(...d.ranking.map(r => r.confidence), 1e-9);
  const maxI = Math.max(...d.drivers.map(x => x.importance), 1e-9);
  $("#rec").innerHTML = `
    <div class="hero">
      <div>
        <div class="hero-name">${esc(d.recommended)}</div>
        <p>The meta-model is ${pct}% sure this is the best fit for your data. That is its confidence in the pick, not the accuracy you will get.</p>
      </div>
      <div class="ring" role="img" aria-label="Confidence ${pct} percent">
        <svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="52" pathLength="100"/><circle class="arc" cx="60" cy="60" r="52" pathLength="100" style="--p:${pct}"/></svg>
        <span>${pct}%</span>
      </div>
    </div>
    <h3>Ranked models</h3>
    <div class="rank">${d.ranking.map((r, i) => `
      <div class="rrow ${i > 2 ? "low" : ""}">
        <span class="rk">${i + 1}</span><span class="rn">${esc(r.model)}</span>
        <span class="bar ${i === 0 ? "amber" : ""}"><i style="width:${Math.max(2, r.confidence / maxP * 100)}%"></i></span>
        <span class="bv">${(r.confidence * 100).toFixed(1)}%</span>
        <button type="button" class="btn ghost val" data-m="${esc(r.model)}" aria-pressed="false">Validate</button>
      </div>`).join("")}</div>
    <h3>What shaped this recommendation</h3>
    <div class="bhead brow"><span>Dataset characteristic</span><span>How much it matters</span><span>Your data</span></div>
    ${d.drivers.map(x => barRow(NICE[x.feature] || x.feature, x.importance / maxI, fmt(x.value))).join("")}`;
  $$(".val").forEach(b => b.onclick = () => validate(b.dataset.m));
}

/* ---------- step 5/6: validate ---------- */
async function validate(model, scroll = true) {
  fail(5);
  $$(".val").forEach(b => { b.disabled = true; b.setAttribute("aria-pressed", b.dataset.m === model); });
  busy(`Cross-validating ${model}…`);
  try {
    const d = await post("/evaluate", { session_id: st.sid, model });
    setStep(6, false);
    st.charts.forEach(c => c.destroy()); st.charts = [];
    st.task === "regression" ? renderReg(d) : renderClf(d);
    sum(6, model);
    if (scroll) go(6);
  } catch (err) { fail(5, err.message); }
  finally { busy(); $$(".val").forEach(b => { b.disabled = false; }); }
}

function renderReg(d) {
  const m = d.metrics, a = d.actual_vs_predicted, h = d.residual_hist, r = d.residuals_vs_predicted;
  const share = Math.max(0, Math.round(m.R2 * 100));
  $("#results").innerHTML =
    `<p class="lede"><b>${esc(d.model)}</b> with ${esc(d.validation)}.</p>` +
    stats([[m.R2, `R², about ${share}% of the variation is explained`, "lead"],
           [m.MAE, "MAE, average error"], [m.RMSE, "RMSE, typical error size"], [m.MSE, "MSE, squared error"]]) +
    `<div class="bento">
      ${panel("c1", "Actual vs predicted", "Points close to the dashed line are accurate predictions.", "span7")}
      ${panel("c2", "Residual distribution", "Errors should cluster around zero in a bell shape.", "span5")}
      ${panel("c3", "Residuals vs predicted", "An even band around zero means the errors have no hidden pattern.", "span7")}
      <div class="panel span5"><h3>Feature importance</h3><p class="cap">How much the score drops when a feature is shuffled.</p>${featBars(d.feature_importance)}</div>
    </div>`;

  const lo = Math.min(...a.actual, ...a.predicted), hi = Math.max(...a.actual, ...a.predicted);
  chart("c1", { type: "scatter", options: opts("Actual", "Predicted"), data: { datasets: [
    { data: a.actual.map((x, i) => ({ x, y: a.predicted[i] })), backgroundColor: BLUE + "88", pointRadius: 3 },
    { type: "line", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: "#9A9EBF", borderDash: [6, 5], borderWidth: 2, pointRadius: 0 },
  ] } });

  const o2 = opts("Residual", "Count"); o2.scales.x.ticks = { maxTicksLimit: 7 };
  chart("c2", { type: "bar", options: o2, data: {
    labels: h.counts.map((_, i) => ((h.edges[i] + h.edges[i + 1]) / 2).toFixed(1)),
    datasets: [{ data: h.counts, backgroundColor: ROSE, borderRadius: 4, barPercentage: 1, categoryPercentage: .95 }] } });

  chart("c3", { type: "scatter", options: opts("Predicted", "Residual"), data: { datasets: [
    { data: r.predicted.map((x, i) => ({ x, y: r.residual[i] })), backgroundColor: ROSE + "88", pointRadius: 3 },
    { type: "line", data: [{ x: Math.min(...r.predicted), y: 0 }, { x: Math.max(...r.predicted), y: 0 }], borderColor: "#9A9EBF", borderDash: [6, 5], borderWidth: 2, pointRadius: 0 },
  ] } });
}

function renderClf(d) {
  const rep = d.report, cm = d.confusion_matrix;
  const cmHtml = `<div class="tablewrap"><table class="tbl cm"><thead><tr><th></th>${d.classes.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>
    ${cm.map((row, i) => {
      const tot = row.reduce((p, c) => p + c, 0) || 1;
      return `<tr><th>${esc(d.classes[i])}</th>${row.map((v, j) => {
        const a = v ? (0.06 + 0.9 * v / tot).toFixed(2) : 0;
        return `<td style="background:rgba(${i === j ? "56,72,224" : "224,71,107"},${a});color:${v / tot > 0.5 ? "#fff" : "inherit"}">${v}</td>`;
      }).join("")}</tr>`;
    }).join("")}</tbody></table></div>`;
  const reportHtml = table(["Class", "Precision", "Recall", "F1", "Support"], d.classes.map(c => {
    const r = rep[c]; return [c, r.precision.toFixed(2), r.recall.toFixed(2), r["f1-score"].toFixed(2), r.support];
  }));

  $("#results").innerHTML =
    `<p class="lede"><b>${esc(d.model)}</b> with ${esc(d.validation)}.</p>` +
    stats([[rep["macro avg"]["f1-score"].toFixed(3), "Macro F1, balanced across classes", "lead"],
           [rep.accuracy.toFixed(3), "Accuracy"], [d.auc, `AUC, ${d.auc_type}`]]) +
    `<div class="bento">
      <div class="panel span7"><h3>Confusion matrix</h3><p class="cap">Rows are the true class and columns the predicted class. Blue cells are correct, red cells are mistakes.</p>${cmHtml}</div>
      <div class="panel span5"><h3>Classification report</h3><p class="cap">Precision, recall and F1 for each class.</p>${reportHtml}</div>
      ${panel("c1", "ROC curve", "Curves that hug the top-left corner beat the dashed chance line.", "span7")}
      ${panel("c2", "Class distribution", "How many samples each class has.", "span5")}
      <div class="panel span12"><h3>Feature importance</h3><p class="cap">How much the score drops when a feature is shuffled.</p>${featBars(d.feature_importance)}</div>
    </div>`;

  const o1 = opts("False positive rate", "True positive rate");
  o1.plugins.legend = { display: d.roc.length > 1, position: "bottom" };
  chart("c1", { type: "scatter", options: o1, data: { datasets: [
    ...d.roc.map((r, i) => ({ label: r.class, data: r.fpr.map((x, j) => ({ x, y: r.tpr[j] })), showLine: true, pointRadius: 0, borderWidth: 2.5, borderColor: SERIES[i % SERIES.length] })),
    { label: "Chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderWidth: 2, borderDash: [6, 5], borderColor: "#9A9EBF" },
  ] } });
  chart("c2", { type: "bar", options: opts("Class", "Samples"), data: {
    labels: Object.keys(d.class_distribution),
    datasets: [{ data: Object.values(d.class_distribution), backgroundColor: BLUE, borderRadius: 6 }] } });
}

setStep(1, false);