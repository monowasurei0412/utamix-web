// AI 推論の専用ワーカー。AI MIX で使うときだけ作られ、使い終わったら終了する(かんたん・詳細モードでは読み込まない)。
// ・推論エンジン: ONNX Runtime Web(WebAssembly・CPU・1スレッド)。GPU は使わない(モデルが小さく、転送の方が重いため。README 参照)
// ・SwiftF0(音程・有声の信頼度, 16kHz)  ・DeepFilterNet3(ノイズ除去, 48kHz)  ・Basic Pitch(採譜: 音ごとの確率, 22.05kHz。v4.6 AI原曲再現MIX)
// ・multilingual-e5-small(文の意味を 384 次元の数にする, v4.8「言葉で直す」)
importScripts('ort.wasm.min.js', 'aicore.js', 'spmtok.js');
ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.wasmPaths = self.location.href.replace(/[^/]*$/, '');
ort.env.logLevel = 'error';
const A = AICORE, DF = A.DF, SESS = {}, MODELS = { swift: ['swiftf0'], dfn: ['dfn3_enc', 'dfn3_erb_dec', 'dfn3_df_dec'], bpitch: ['basicpitch_nmp'], e5: ['e5small_q'] };
let TOK = null;
let cancel = false;
const post = (o, tr) => self.postMessage(o, tr || []);
async function load(kind) {
  const out = [];
  for (const nm of MODELS[kind]) {
    if (!SESS[nm]) { const t0 = performance.now(); const buf = await (await fetch('models/' + nm + '.onnx')).arrayBuffer();
      SESS[nm] = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }); post({ type: 'log', msg: 'load ' + nm + ' ' + Math.round(performance.now() - t0) + 'ms' }); }
    out.push(SESS[nm]);
  }
  return out;
}
const T32 = (a, dims) => new ort.Tensor('float32', a, dims);
const yieldNow = () => new Promise(r => setTimeout(r, 0));

// SwiftF0: 44.1kHz の声 → 16ms ごとの音程(Hz)・信頼度(0〜1)。Python 版(swift_f0/core.py の detect)と同じ区切り方
async function swift(x, sr, id) {
  const [s] = await load('swift'), y = A.resample(x, sr, 16000), HOP = 256, n = Math.max(1, Math.floor(y.length / HOP)), WIN = 1875, LEFT = 11, LA = 10;
  const pitch = new Float32Array(n), conf = new Float32Array(n), fmin = T32(new Float32Array([46.875]), []), fmax = T32(new Float32Array([2093.75]), []);
  for (let start = 0; start < n; start += WIN) {
    if (cancel) throw new Error('cancel');
    const end = Math.min(start + WIN, n), left = Math.max(0, start - LEFT), a = left * HOP, b = end < n ? Math.min(y.length, (end + LA) * HOP) : y.length, w = y.slice(a, b);
    const r = await s.run({ audio: T32(w, [1, w.length]), fmin, fmax }), p = r.pitch.data, c = r.confidence.data;
    for (let k = start; k < end; k++) { const j = k - left; if (j >= c.length) break; pitch[k] = p[j]; let pk = 0; for (let i = k * HOP; i < (k + 1) * HOP && i < y.length; i++) pk = Math.max(pk, Math.abs(y[i])); conf[k] = pk < 1e-3 ? 0 : c[j]; }
    post({ type: 'prog', id, p: end / n });
  }
  return { pitch, conf, hopSec: HOP / 16000 };
}

// Basic Pitch: 22.05kHz にして、2秒(43844サンプル)の窓を、前後15フレームずつ重ねて推論し、重なりを捨ててつなぐ(Python 版 basic_pitch/inference.py と同じ)
//  返すのは「音ごとの確率」(フレーム×88鍵, 約86.1フレーム/秒)
async function bpitch(x, sr, id) {
  const [s] = await load('bpitch'), y0 = A.resample(x, sr, 22050), N = 43844, OV = 30 * 256, HOPW = N - OV, y = new Float32Array(y0.length + OV / 2); y.set(y0, OV / 2);
  const nWin = Math.ceil(y.length / HOPW), FR = 172 - 30, outN = Math.floor(y0.length / HOPW * FR), pr = new Float32Array(Math.max(1, nWin * FR) * 88);
  let o = 0;
  for (let w = 0; w < nWin; w++) {
    if (cancel) throw new Error('cancel');
    const win = new Float32Array(N); win.set(y.subarray(w * HOPW, Math.min(y.length, w * HOPW + N)));
    const r = await s.run({ 'serving_default_input_2:0': T32(win, [1, N, 1]) }), nt = r['StatefulPartitionedCall:1'].data;
    for (let f = 15; f < 172 - 15; f++) { pr.set(nt.subarray(f * 88, (f + 1) * 88), o); o += 88; }
    if (w % 4 === 3) { post({ type: 'prog', id, p: (w + 1) / nWin }); await yieldNow(); }
  }
  return { post: pr.slice(0, Math.min(o, outN * 88)), fps: 22050 / 256 };
}

// DeepFilterNet3: 1000 フレーム(10秒)ずつ、前の 600 フレーム(6秒)を慣らしに使って推論し、つなぎ目は混ぜる(Python で、全体を一度に推論したときと同じ品質になることを確認)
async function dfn(x, sr, limDb, id, Cq) {
  // メモリを抑えるため、推論したところから順に「マスク・ディープフィルタ → 音に戻す」まで済ませる(曲全体の係数やスペクトルを持たない)
  const [enc, ed, dd] = await load('dfn'), x48 = A.resample(x, sr, DF.SR), S = DF.analysis(x48), T = S.T, ft = DF.features(S), NE = DF.NE, NDF = DF.NDF, NC = NDF * 10;
  const lsnr = new Float32Array(T), C = Cq || 1000, O = 600, F = DF.N / 2 + 1, syn = DF.synthStream(x48.length);
  let tail = null, tailA = 0, done = 0; // tail: 前のかたまりの、つなぎ目で混ぜる所(マスク・係数)
  const m = new Float32Array((C + O + 10) * NE), co = new Float32Array((C + O + 10) * NC);
  for (let s = 0; s < T; s += C) {
    if (cancel) throw new Error('cancel');
    const a = Math.max(0, s - O), b = Math.min(T, s + C + 10), e = Math.min(T, s + C), L = b - a;
    const erb = ft.erb.slice(a * NE, b * NE), sp = new Float32Array(2 * L * NDF); sp.set(ft.spec.subarray(a * NDF, b * NDF), 0); sp.set(ft.spec.subarray(T * NDF + a * NDF, T * NDF + b * NDF), L * NDF);
    const r = await enc.run({ feat_erb: T32(erb, [1, 1, L, NE]), feat_spec: T32(sp, [1, 2, L, NDF]) });
    const r2 = await ed.run({ emb: r.emb, e3: r.e3, e2: r.e2, e1: r.e1, e0: r.e0 }), r3 = await dd.run({ emb: r.emb, c0: r.c0 });
    const mm = r2.m.data, cc = r3.coefs.data, ls = r.lsnr.data, x0 = a === 0 ? 0 : a + (O >> 1);
    // このかたまりで決まる範囲 [x0, e) のマスク・係数(先頭の [x0, s) は前のかたまりと混ぜる)
    for (let t = x0; t < e; t++) { const j = t - a, i = t - x0, w = a === 0 || t >= s ? 1 : (t - x0) / Math.max(1, s - x0), tl = tail && t < s ? t - tailA : -1;
      for (let q = 0; q < NE; q++) m[i * NE + q] = (tl >= 0 ? tail.m[tl * NE + q] * (1 - w) : 0) + mm[j * NE + q] * (tl >= 0 ? w : 1);
      for (let q = 0; q < NC; q++) co[i * NC + q] = (tl >= 0 ? tail.co[tl * NC + q] * (1 - w) : 0) + cc[j * NC + q] * (tl >= 0 ? w : 1);
      lsnr[t] = (tl >= 0 ? tail.ls[tl] * (1 - w) : 0) + ls[j] * (tl >= 0 ? w : 1); }
    // 次のかたまりと混ぜる所は、まだ確定しない
    const nx0 = e < T ? Math.max(0, e - O) + (O >> 1) : e, fin = Math.min(e, nx0);
    for (let t = done; t < fin; t++) syn.push(DF.applyFrame(S, t, m, co, x0), t);
    done = fin;
    if (e < T) { const n = e - nx0; tail = { m: m.slice((nx0 - x0) * NE, (e - x0) * NE), co: co.slice((nx0 - x0) * NC, (e - x0) * NC), ls: lsnr.slice(nx0, e) }; tailA = nx0;
      // 確定していない [nx0, e) は、次のかたまりで混ぜてから音にする → m, co の先頭へ移す必要はない(次のかたまりで tail から作り直す)
    }
    for (const k in r) r[k].dispose && r[k].dispose(); for (const k in r2) r2[k].dispose && r2[k].dispose(); for (const k in r3) r3[k].dispose && r3[k].dispose();
    post({ type: 'prog', id, p: e / T }); await yieldNow();
  }
  let y48 = syn.end();
  if (limDb > 0) { const g = Math.pow(10, -limDb / 20); for (let i = 0; i < y48.length; i++) y48[i] = x48[i] * g + y48[i] * (1 - g); } // 減らす量の上限(声を壊さないため)
  const y = A.resample(y48, DF.SR, sr, x.length);
  return { y, lsnr, lsnrHop: DF.H / DF.SR }; // lsnr: 10ms ごとの「声とノイズの比」の推定(dB)
}

// 文 → 意味の数(384次元・長さ1)。E5 の決まりどおり先頭に "query: " を付け、単語ごとの出力を平均する
async function embed(texts) {
  const [s] = await load('e5');
  if (!TOK) { const t0 = performance.now(); TOK = new SpmTok(await (await fetch('models/e5small_spm.model')).arrayBuffer()); post({ type: 'log', msg: 'load e5 tokenizer ' + Math.round(performance.now() - t0) + 'ms' }); }
  const D = 384, out = new Float32Array(texts.length * D), toks = [];
  for (let k = 0; k < texts.length; k++) {
    const ids = TOK.encode('query: ' + texts[k], 128), n = ids.length, I = new BigInt64Array(n), M = new BigInt64Array(n).fill(1n), Z = new BigInt64Array(n);
    for (let i = 0; i < n; i++) I[i] = BigInt(ids[i]);
    const r = await s.run({ input_ids: new ort.Tensor('int64', I, [1, n]), attention_mask: new ort.Tensor('int64', M, [1, n]), token_type_ids: new ort.Tensor('int64', Z, [1, n]) });
    const h = r.last_hidden_state.data; let nn = 0;
    for (let d = 0; d < D; d++) { let a = 0; for (let i = 0; i < n; i++) a += h[i * D + d]; a /= n; out[k * D + d] = a; nn += a * a; }
    nn = Math.sqrt(nn) || 1; for (let d = 0; d < D; d++) out[k * D + d] /= nn; toks.push(n);
  }
  return { v: out, dim: D, toks };
}

self.onmessage = async ev => {
  const m = ev.data; cancel = false;
  try {
    if (m.cmd === 'cancel') { cancel = true; return; }
    if (m.cmd === 'probe') { const t0 = performance.now(); await load('swift'); post({ type: 'probe', id: m.id, ok: true, ms: performance.now() - t0, simd: true }); return; }
    if (m.cmd === 'swift') { const t0 = performance.now(), r = await swift(m.x, m.sr, m.id); post({ type: 'swift', id: m.id, ms: performance.now() - t0, ...r }, [r.pitch.buffer, r.conf.buffer]); return; }
    if (m.cmd === 'dfn') { const t0 = performance.now(), r = await dfn(m.x, m.sr, m.limDb || 0, m.id, m.C); post({ type: 'dfn', id: m.id, ms: performance.now() - t0, ...r }, [r.y.buffer, r.lsnr.buffer]); return; }
    if (m.cmd === 'bpitch') { const t0 = performance.now(), r = await bpitch(m.x, m.sr, m.id); post({ type: 'bpitch', id: m.id, ms: performance.now() - t0, ...r }, [r.post.buffer]); return; }
    if (m.cmd === 'embed') { const t0 = performance.now(), r = await embed(m.texts); post({ type: 'embed', id: m.id, ms: performance.now() - t0, ...r }, [r.v.buffer]); return; }
    if (m.cmd === 'release') { for (const k in SESS) { try { await SESS[k].release(); } catch (e) {} delete SESS[k]; } post({ type: 'released', id: m.id }); return; }
  } catch (e) { post({ type: 'error', id: m.id, msg: String(e && e.message || e) }); }
};
