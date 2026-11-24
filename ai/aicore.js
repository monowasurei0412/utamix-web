// AI 用の信号処理(推論の前後)。推論そのものは aiworker.js で ONNX Runtime を使う。
//  ・Resampler … カイザー窓 sinc の多相フィルタ(44.1k⇔48k, 44.1k→16k)
//  ・FFTN      … 2,3,5 の積の長さの FFT(DeepFilterNet3 は 960 点)
//  ・DFN       … DeepFilterNet3 の前処理(STFT・ERB 特徴・複素特徴)と後処理(マスク・ディープフィルタ・合成)
//    libDF(DeepFilterNet の参照実装)と同じ計算。Python で参照実装と照合済み(test/ai_core.js)
(function (root) {
  'use strict';
  // ---------- リサンプラー ----------
  function bessel0(x) { let s = 1, t = 1; for (let k = 1; k < 50; k++) { t *= (x / (2 * k)) ** 2; s += t; if (t < 1e-12 * s) break; } return s; }
  function gcd(a, b) { while (b) [a, b] = [b, a % b]; return a; }
  function Resampler(from, to, zc = 24, beta = 8.6) {
    const g = gcd(from, to), L = to / g, M = from / g; // 出力 = 入力 × L / M
    const fc = 0.5 * Math.min(1, L / M) * 0.94; // 入力サンプルあたりの遮断周波数(サイクル)
    const half = Math.ceil(zc / (2 * fc)), taps = 2 * half, ib = bessel0(beta);
    const tab = new Float32Array(L * taps); // tab[phase*taps + i] : 位相 phase/L のときの i 番目
    for (let ph = 0; ph < L; ph++) {
      const frac = ph / L; let s = 0;
      for (let i = 0; i < taps; i++) { const tau = (i - half + 1) - frac, a = Math.abs(tau) / half; // x[base + i - half + 1] の重み
        let h = 0; if (a < 1) { const u = 2 * fc * tau; h = 2 * fc * (u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u)) * bessel0(beta * Math.sqrt(1 - a * a)) / ib; }
        tab[ph * taps + i] = h; s += h; }
      for (let i = 0; i < taps; i++) tab[ph * taps + i] /= s; // 直流の大きさを 1 に
    }
    this.L = L; this.M = M; this.half = half; this.taps = taps; this.tab = tab;
  }
  Resampler.prototype.run = function (x, outLen) {
    const { L, M, half, taps, tab } = this, n = x.length, nOut = outLen != null ? outLen : Math.floor(n * L / M), y = new Float32Array(nOut);
    for (let j = 0; j < nOut; j++) {
      const num = j * M, base = Math.floor(num / L), ph = num - base * L, o = ph * taps, s0 = base - half + 1; let acc = 0;
      if (s0 >= 0 && s0 + taps <= n) { for (let i = 0; i < taps; i++) acc += x[s0 + i] * tab[o + i]; }
      else for (let i = 0; i < taps; i++) { const k = s0 + i; if (k >= 0 && k < n) acc += x[k] * tab[o + i]; }
      y[j] = acc;
    }
    return y;
  };
  const RS = {}; const rs = (a, b) => RS[a + '>' + b] || (RS[a + '>' + b] = new Resampler(a, b));
  function resample(x, from, to, outLen) { return from === to ? Float32Array.from(x) : rs(from, to).run(x, outLen); }

  // ---------- 2・3・5 の積の長さの FFT(前向き・正規化なし) ----------
  function FFTN(n) {
    this.n = n; const f = []; let m = n; for (const p of [4, 2, 3, 5]) while (m % p === 0) { f.push(p); m /= p; }
    if (m !== 1) throw new Error('FFTN: ' + n);
    this.f = f; this.c = new Float64Array(n); this.s = new Float64Array(n);
    for (let i = 0; i < n; i++) { this.c[i] = Math.cos(-2 * Math.PI * i / n); this.s[i] = Math.sin(-2 * Math.PI * i / n); }
    this.tr = new Float64Array(n); this.ti = new Float64Array(n);
  }
  FFTN.prototype.run = function (re, im) { // 結果は re, im に入れて返す
    const n = this.n, outR = this.tr, outI = this.ti; this._rec(n, re, im, 0, 1, outR, outI, 0, 0); re.set(outR); im.set(outI); };
  FFTN.prototype._rec = function (n, xr, xi, off, stride, yr, yi, yo, fi) {
    if (n === 1) { yr[yo] = xr[off]; yi[yo] = xi[off]; return; }
    const p = this.f[fi], m = n / p, N = this.n, cs = this.c, sn = this.s, step = N / n;
    for (let q = 0; q < p; q++) this._rec(m, xr, xi, off + q * stride, stride * p, yr, yi, yo + q * m, fi + 1);
    const tr = [0, 0, 0, 0, 0], ti = [0, 0, 0, 0, 0];
    for (let k = 0; k < m; k++) {
      for (let q = 0; q < p; q++) { const a = yr[yo + q * m + k], b = yi[yo + q * m + k], e = (q * k * step) % N, c = cs[e], s = sn[e]; tr[q] = a * c - b * s; ti[q] = a * s + b * c; }
      for (let s2 = 0; s2 < p; s2++) { let ar = 0, ai = 0; for (let q = 0; q < p; q++) { const e = (q * s2 * m * step) % N, c = cs[e], s = sn[e]; ar += tr[q] * c - ti[q] * s; ai += tr[q] * s + ti[q] * c; }
        yr[yo + k + s2 * m] = ar; yi[yo + k + s2 * m] = ai; }
    }
  };

  // ---------- DeepFilterNet3 ----------
  const DF = { SR: 48000, N: 960, H: 480, NE: 32, NDF: 96, ORD: 5 };
  function erbWidths(sr, n, nb, mn) {
    const f2e = f => 9.265 * Math.log1p(f / (24.7 * 9.265)), e2f = e => 24.7 * 9.265 * (Math.exp(e / 9.265) - 1);
    const fw = sr / n, lo = f2e(0), hi = f2e(sr / 2), step = (hi - lo) / nb; let prev = 0, over = 0; const out = [];
    for (let i = 1; i <= nb; i++) { const fb = Math.round(e2f(lo + i * step) / fw); let nf = fb - prev - over; if (nf < mn) { over = mn - nf; nf = mn; } else over = 0; out.push(nf); prev = fb; }
    out[nb - 1] += 1; const tl = out.reduce((a, b) => a + b, 0) - (n / 2 + 1); if (tl > 0) out[nb - 1] -= tl; return out;
  }
  function normAlpha(sr, hop, tau) { const a = Math.exp(-(hop / sr) / tau); for (let p = 3; ; p++) { const r = Math.round(a * 10 ** p) / 10 ** p; if (r < 1) return r; } }
  DF.EW = erbWidths(DF.SR, DF.N, DF.NE, 2); DF.A = normAlpha(DF.SR, DF.H, 1);
  DF.W = Float32Array.from({ length: DF.N }, (_, k) => Math.sin(Math.PI / 2 * Math.sin(Math.PI * (k + 0.5) / DF.N) ** 2));
  DF.WN = 1 / (DF.N * DF.N / (2 * DF.H));
  const F = DF.N / 2 + 1;
  // STFT: x(48kHz) の後ろに N 個の 0 を足して分析。返り値 re/im は [T][F]
  DF.analysis = function (x) {
    const { N, H, W, WN } = DF, T = Math.floor((x.length + N) / H), re = new Float32Array(T * F), im = new Float32Array(T * F), fft = DF.fft || (DF.fft = new FFTN(N)), br = new Float64Array(N), bi = new Float64Array(N);
    for (let t = 0; t < T; t++) { const s0 = t * H - (N - H); // [mem(N-H), 新しい H]
      for (let i = 0; i < N; i++) { const k = s0 + i; br[i] = (k >= 0 && k < x.length ? x[k] : 0) * W[i]; bi[i] = 0; }
      fft.run(br, bi); for (let f = 0; f < F; f++) { re[t * F + f] = br[f] * WN; im[t * F + f] = bi[f] * WN; } }
    return { re, im, T };
  };
  DF.features = function (S) {
    const { T, re, im } = S, { NE, NDF, EW, A } = DF, erb = new Float32Array(T * NE), spec = new Float32Array(2 * T * NDF);
    const st = Float64Array.from({ length: NE }, (_, i) => -60 - 30 * i / (NE - 1)), us = Float64Array.from({ length: NDF }, (_, i) => 0.001 - 0.0009 * i / (NDF - 1));
    for (let t = 0; t < T; t++) {
      let f = 0; for (let b = 0; b < NE; b++) { let s = 0; for (let j = 0; j < EW[b]; j++, f++) { const a = re[t * F + f], c = im[t * F + f]; s += a * a + c * c; }
        const v = 10 * Math.log10(s / EW[b] + 1e-10); st[b] = v * (1 - A) + st[b] * A; erb[t * NE + b] = (v - st[b]) / 40; }
      for (let k = 0; k < NDF; k++) { const a = re[t * F + k], c = im[t * F + k]; us[k] = Math.hypot(a, c) * (1 - A) + us[k] * A; const d = Math.sqrt(us[k]);
        spec[t * NDF + k] = a / d; spec[T * NDF + t * NDF + k] = c / d; }
    }
    return { erb, spec }; // erb:[1,1,T,32]  spec:[1,2,T,96]
  };
  // マスク m:[T][32]、係数 co:[T][96][10](次数5×実虚) を当てた複素スペクトル
  DF.apply = function (S, m, co) {
    const { T, re, im } = S, { NE, NDF, EW, ORD } = DF, or = new Float32Array(T * F), oi = new Float32Array(T * F), pad = 2;
    for (let t = 0; t < T; t++) {
      let f = 0; for (let b = 0; b < NE; b++) { const g = m[t * NE + b]; for (let j = 0; j < EW[b]; j++, f++) { or[t * F + f] = re[t * F + f] * g; oi[t * F + f] = im[t * F + f] * g; } }
      for (let k = 0; k < NDF; k++) { let ar = 0, ai = 0;
        for (let q = 0; q < ORD; q++) { const tt = t + q - pad; if (tt < 0 || tt >= T) continue; const xr = re[tt * F + k], xi = im[tt * F + k], o = (t * NDF + k) * 10 + q * 2, cr = co[o], ci = co[o + 1]; ar += xr * cr - xi * ci; ai += xr * ci + xi * cr; }
        or[t * F + k] = ar; oi[t * F + k] = ai; }
    }
    return { re: or, im: oi, T };
  };
  // 1フレームずつ: t のマスク(m)・係数(co)は「base からの番号」で渡す。返り値は [re(F), im(F)]
  DF.applyFrame = function (S, t, m, co, base) {
    const { T, re, im } = S, { NE, NDF, EW, ORD } = DF, or = new Float64Array(F), oi = new Float64Array(F), i0 = t - base, pad = 2;
    let f = 0; for (let b = 0; b < NE; b++) { const g = m[i0 * NE + b]; for (let j = 0; j < EW[b]; j++, f++) { or[f] = re[t * F + f] * g; oi[f] = im[t * F + f] * g; } }
    for (let k = 0; k < NDF; k++) { let ar = 0, ai = 0;
      for (let q = 0; q < ORD; q++) { const tt = t + q - pad; if (tt < 0 || tt >= T) continue; const xr = re[tt * F + k], xi = im[tt * F + k], o = (i0 * NDF + k) * 10 + q * 2, cr = co[o], ci = co[o + 1]; ar += xr * cr - xi * ci; ai += xr * ci + xi * cr; }
      or[k] = ar; oi[k] = ai; }
    return [or, oi];
  };
  // 順に足していく合成(フレームの順番どおりに push する)
  DF.synthStream = function (n) {
    const { N, H, W } = DF, d = N - H, y = new Float32Array(n), mem = new Float64Array(N - H), fft = DF.fft || (DF.fft = new FFTN(N)), br = new Float64Array(N), bi = new Float64Array(N);
    return {
      push([re, im], t) {
        for (let f = 0; f < F; f++) { br[f] = re[f]; bi[f] = -im[f]; } for (let f = F; f < N; f++) { br[f] = re[N - f]; bi[f] = im[N - f]; }
        fft.run(br, bi);
        for (let i = 0; i < N; i++) { const v = br[i] * W[i] + (i < N - H ? mem[i] : 0); if (i < H) { const k = t * H + i - d; if (k >= 0 && k < n) y[k] = v; } else mem[i - H] = v; }
      },
      end() { return y; },
    };
  };
  DF.synthesis = function (S, n) { // n: 元の長さ。遅れ(N-H)を取り除いて返す
    const { N, H, W } = DF, { T, re, im } = S, y = new Float32Array(T * H), mem = new Float64Array(N - H), fft = DF.fft || (DF.fft = new FFTN(N)), br = new Float64Array(N), bi = new Float64Array(N);
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < F; f++) { br[f] = re[t * F + f]; bi[f] = -im[t * F + f]; } // 逆変換 = conj(FFT(conj(X)))
      for (let f = F; f < N; f++) { br[f] = re[t * F + (N - f)]; bi[f] = im[t * F + (N - f)]; }
      fft.run(br, bi);
      for (let i = 0; i < N; i++) { const v = br[i] * W[i] + (i < N - H ? mem[i] : 0); if (i < H) y[t * H + i] = v; else mem[i - H] = v; }
    }
    const d = N - H; return y.subarray(d, d + n);
  };
  const api = { Resampler, resample, FFTN, DF };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.AICORE = api;
})(typeof self !== 'undefined' ? self : this);
