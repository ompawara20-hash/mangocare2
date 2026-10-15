import { useEffect, useState, DragEvent } from "react";
import { Leaf, Upload, History, BookOpen, Cpu, Download, AlertTriangle, Trash2, Loader2 } from "lucide-react";
import { api, ApiError } from "./api";

type Tab = "analyze" | "history" | "diseases" | "model";
/* eslint-disable @typescript-eslint/no-explicit-any */
const STATUS: Record<string, { label: string; cls: string }> = {
  healthy_appearance: { label: "Healthy appearance (not a guarantee)", cls: "bg-green-50 border-green-400 text-green-900" },
  possible_disease: { label: "Possible disease detected", cls: "bg-red-50 border-red-400 text-red-900" },
  uncertain: { label: "Uncertain: please retake the photo", cls: "bg-amber-50 border-amber-400 text-amber-900" },
};
const Card = ({ children, cls = "" }: { children: React.ReactNode; cls?: string }) => <div className={`rounded-xl border bg-white p-5 shadow-sm ${cls}`}>{children}</div>;
const List = ({ title, items }: { title: string; items?: string[] }) => items?.length ? <div className="mt-3"><h4 className="font-semibold text-forest">{title}</h4><ul className="list-disc ml-5 text-sm">{items.map((x) => <li key={x}>{x}</li>)}</ul></div> : null;

function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const [res, setRes] = useState<any>(null);
  const pick = (f?: File) => { if (f) { setFile(f); setErr(null); setRes(null); } };
  const drop = (e: DragEvent) => { e.preventDefault(); pick(e.dataTransfer.files[0]); };
  const submit = async () => {
    if (!file || busy) return;
    setBusy(true); setErr(null); setRes(null);
    const fd = new FormData(); fd.append("file", file);
    try { setRes(await api("/analyze", { method: "POST", body: fd })); } catch (e) { setErr(e as ApiError); } finally { setBusy(false); }
  };
  const st = res && STATUS[res.status];
  return (
    <div className="space-y-4">
      <Card>
        <label onDragOver={(e) => e.preventDefault()} onDrop={drop} className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed border-leaf p-8 text-center">
          <Upload className="text-forest" />
          <span>{file ? `${file.name} (${(file.size / 1024).toFixed(0)} KB)` : "Drag & drop a mango LEAF photo, or click to browse"}</span>
          <input type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
        </label>
        {file && <img src={URL.createObjectURL(file)} alt="preview" className="mx-auto mt-4 max-h-64 rounded-lg" />}
        <button disabled={!file || busy} onClick={submit} className="mt-4 inline-flex items-center gap-2 rounded-lg bg-forest px-5 py-2 text-white disabled:opacity-50">
          {busy && <Loader2 className="animate-spin" size={16} />} Analyze image
        </button>
        <p className="mt-3 text-xs text-gray-600">Tips: good light, in focus, close view of the affected area, no filters. A photo cannot confirm internal or lab-confirmed conditions.</p>
      </Card>
      {err && <Card cls="border-red-300 bg-red-50"><div className="flex gap-2 text-red-800"><AlertTriangle /><div><b>Unable to analyze this image</b><p className="text-sm">{err.message}</p>{err.suggestion && <p className="text-sm">{err.suggestion}</p>}</div></div></Card>}
      {res && st && (
        <Card cls={st.cls}>
          <h3 className="text-xl font-bold">{st.label}</h3>
          <p className="mt-1">Top class: <b>{res.prediction.class_name}</b> — model score {(res.prediction.confidence * 100).toFixed(1)}%</p>
          <p className="text-xs">{res.prediction.interpretation}. Mango check score: {res.image_validation.score}</p>
          {res.alternatives?.length > 0 && <p className="mt-2 text-sm">Other scores: {res.alternatives.map((a: any) => `${a.class_name} ${(a.score * 100).toFixed(1)}%`).join(", ")}</p>}
          {res.condition_information && <div className="mt-3 rounded bg-white/70 p-3 text-gray-900"><p className="text-sm">{res.condition_information.description}</p><List title="Visible symptoms" items={res.condition_information.symptoms} /><List title="Precautions" items={res.condition_information.precautions} /><List title="Next steps" items={res.condition_information.next_steps} /></div>}
          <List title="Limitations" items={res.limitations} />
          <p className="mt-3 text-xs">ID {res.analysis_id} · {res.created_at}</p>
          <a href={`/api/reports/${res.analysis_id}`} className="mt-3 inline-flex items-center gap-2 rounded-lg bg-forest px-4 py-2 text-white"><Download size={16} /> Download PDF report</a>
        </Card>
      )}
    </div>
  );
}

function HistoryTab() {
  const [d, setD] = useState<any>(null); const [err, setErr] = useState<ApiError | null>(null);
  const load = () => api("/history").then(setD).catch((e) => setErr(e));
  useEffect(() => { load(); }, []);
  if (err) return <Card cls="border-red-300">{err.message}</Card>;
  if (!d) return <p>Loading…</p>;
  const s = d.stats;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{[["Total", s.total], ["Healthy-looking", s.healthy_appearance ?? 0], ["Possible disease", s.possible_disease ?? 0], ["Uncertain", s.uncertain ?? 0]].map(([k, v]) => <Card key={k as string}><div className="text-2xl font-bold text-forest">{v}</div><div className="text-xs">{k}</div></Card>)}</div>
      {d.items.length === 0 ? <Card>No analyses yet. Analyze a photo to get started.</Card> :
        <Card><table className="w-full text-sm"><thead><tr className="text-left"><th>Time (UTC)</th><th>Class</th><th>Status</th><th>Score</th><th /></tr></thead><tbody>
          {d.items.map((r: any) => <tr key={r.id} className="border-t"><td>{r.created_at}</td><td>{r.class_name}</td><td>{STATUS[r.status]?.label ?? r.status}</td><td>{(r.confidence * 100).toFixed(1)}%</td>
            <td className="whitespace-nowrap"><a className="text-forest underline" href={`/api/reports/${r.id}`}>PDF</a> <button aria-label="delete" onClick={() => api(`/history/${r.id}`, { method: "DELETE" }).then(load)}><Trash2 size={14} /></button></td></tr>)}
        </tbody></table></Card>}
    </div>
  );
}

function Diseases() {
  const [q, setQ] = useState(""); const [d, setD] = useState<any>(null);
  useEffect(() => { api(`/diseases?q=${encodeURIComponent(q)}`).then(setD).catch(() => setD({ items: [] })); }, [q]);
  return (
    <div className="space-y-3">
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search diseases…" className="w-full rounded-lg border p-2" />
      <p className="text-xs text-gray-600">Educational information only. Not a substitute for expert advice. References not yet verified.</p>
      {d?.items.map((x: any) => <Card key={x.id}><h3 className="font-bold text-forest">{x.name}{x.scientific_name && <i className="ml-2 text-sm font-normal">{x.scientific_name}</i>}</h3><p className="text-sm">{x.description}</p><List title="Symptoms" items={x.symptoms} /><List title="Precautions" items={x.precautions} /></Card>)}
    </div>
  );
}

function ModelTab() {
  const [m, setM] = useState<any>(null); const [h, setH] = useState<any>(null);
  useEffect(() => { api("/model-info").then(setM).catch(() => {}); api("/health").then(setH).catch(() => {}); }, []);
  if (!m || !h) return <Card cls="border-red-300">Backend unreachable.</Card>;
  return (
    <Card>
      <h3 className="font-bold text-forest">{m.name} v{m.version}</h3>
      <p className="text-sm">{m.architecture} · scope: {m.scope} · input {m.input_size.join("×")}</p>
      <p className="mt-2 text-sm">Status: <b>{h.status}</b> · disease model {h.disease_model_ready ? "ready" : "NOT ready"} · validator {h.validator_ready ? "ready" : "NOT ready"} · device {h.device}</p>
      <p className="mt-2 text-sm">Classes: {m.classes.join(", ")}</p>
      <p className="mt-2 text-sm">{m.metrics_note}</p>
      <List title="Limitations" items={m.limitations} />
    </Card>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("analyze");
  const tabs: [Tab, string, JSX.Element][] = [["analyze", "Analyze", <Upload size={16} />], ["history", "History", <History size={16} />], ["diseases", "Disease Library", <BookOpen size={16} />], ["model", "About the Model", <Cpu size={16} />]];
  return (
    <div className="mx-auto max-w-3xl p-4">
      <header className="mb-4 flex items-center gap-2 text-forest"><Leaf /><h1 className="text-2xl font-bold">MangoCare AI</h1></header>
      <nav className="mb-4 flex flex-wrap gap-2">{tabs.map(([k, l, i]) => <button key={k} onClick={() => setTab(k)} className={`inline-flex items-center gap-1 rounded-full border px-3 py-1 text-sm ${tab === k ? "bg-forest text-white" : "bg-white"}`}>{i}{l}</button>)}</nav>
      {tab === "analyze" && <Analyze />}{tab === "history" && <HistoryTab />}{tab === "diseases" && <Diseases />}{tab === "model" && <ModelTab />}
      <footer className="mt-6 text-xs text-gray-600">Preliminary, image-based screening for educational use only. Not a laboratory diagnosis.</footer>
    </div>
  );
}
