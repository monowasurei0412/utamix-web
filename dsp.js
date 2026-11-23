'use strict';
/* おまかせ歌みたMIX - 音声処理エンジン
   キー判定 / 音程解析(YIN+Viterbi) / ピッチ補正(PSOLA) / ボーカル処理 / ミックス
   ブラウザのWorkerとNode(テスト用)の両方で動く */
(function (root) {
  const SR = 44100;
  const HOP = 512;                 // 音程フレーム間隔(44.1kHz換算) = 11.6ms
  const HOP_SEC = HOP / SR;
  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const MAJOR = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1];

  // ---------- FFT ----------
  function FFT(n) {
    this.n = n;
    const bits = Math.round(Math.log2(n));
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos(2 * Math.PI * i / n);
      this.sin[i] = Math.sin(2 * Math.PI * i / n);
    }
  }
  FFT.prototype.forward = function (re, im) {
    const n = this.n, rev = this.rev, cos = this.cos, sin = this.sin;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const xr = re[j + half], xi = im[j + half];
          const tr = xr * cos[k] + xi * sin[k];
          const ti = xi * cos[k] - xr * sin[k];
          re[j + half] = re[j] - tr; im[j + half] = im[j] - ti;
          re[j] += tr; im[j] += ti;
        }
      }
    }
  };

  // ---------- Biquad (RBJ) ----------
  function biquad(type, f, q, gainDb, fs) {
    fs = fs || SR;
    const w = 2 * Math.PI * f / fs, c = Math.cos(w), s = Math.sin(w);
    const alpha = s / (2 * q), A = Math.pow(10, (gainDb || 0) / 40);
    let b0, b1, b2, a0, a1, a2;
    if (type === 'lp') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = b0; a0 = 1 + alpha; a1 = -2 * c; a2 = 1 - alpha; }
    else if (type === 'hp') { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = b0; a0 = 1 + alpha; a1 = -2 * c; a2 = 1 - alpha; }
    else if (type === 'bp') { b0 = alpha; b1 = 0; b2 = -alpha; a0 = 1 + alpha; a1 = -2 * c; a2 = 1 - alpha; } // 帯域通過(中心で0dB)
    else if (type === 'notch') { b0 = 1; b1 = -2 * c; b2 = 1; a0 = 1 + alpha; a1 = -2 * c; a2 = 1 - alpha; }
    else if (type === 'peak') { b0 = 1 + alpha * A; b1 = -2 * c; b2 = 1 - alpha * A; a0 = 1 + alpha / A; a1 = -2 * c; a2 = 1 - alpha / A; }
    else { // high shelf
      const sq = 2 * Math.sqrt(A) * alpha;
      b0 = A * ((A + 1) + (A - 1) * c + sq); b1 = -2 * A * ((A - 1) + (A + 1) * c); b2 = A * ((A + 1) + (A - 1) * c - sq);
      a0 = (A + 1) - (A - 1) * c + sq; a1 = 2 * ((A - 1) - (A + 1) * c); a2 = (A + 1) - (A - 1) * c - sq;
    }
    return { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 };
  }
  function filt(x, c) {
    let z1 = 0, z2 = 0;
    const b0 = c.b0, b1 = c.b1, b2 = c.b2, a1 = c.a1, a2 = c.a2;
    for (let i = 0; i < x.length; i++) {
      const v = x[i], y = b0 * v + z1;
      z1 = b1 * v - a1 * y + z2; z2 = b2 * v - a2 * y;
      x[i] = y;
    }
  }
  function decimate(x, D) {
    const t = new Float32Array(x);
    const c = biquad('lp', 0.4 * SR / D, 0.707, 0, SR);
    filt(t, c); filt(t, c);
    const n = Math.floor(t.length / D), y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = t[i * D];
    return y;
  }

  // ---------- キー判定 ----------
  const KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  function corr(chroma, prof, tonic) {
    let mx = 0, my = 0;
    for (let i = 0; i < 12; i++) { mx += chroma[i]; my += prof[i]; }
    mx /= 12; my /= 12;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < 12; i++) {
      const a = chroma[(i + tonic) % 12] - mx, b = prof[i] - my;
      sxy += a * b; sxx += a * a; syy += b * b;
    }
    return sxx > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
  }
  // 同じ音階(メジャーtonic=平行調マイナーtonic+9)のスコア
  function scaleScores(chroma) {
    const out = new Float64Array(12), mode = new Uint8Array(12);
    for (let t = 0; t < 12; t++) {
      const a = corr(chroma, KS_MAJ, t), b = corr(chroma, KS_MIN, (t + 9) % 12);
      out[t] = Math.max(a, b); mode[t] = b > a ? 1 : 0;
    }
    return { out, mode };
  }
  function detectKey(mono, progress) {
    const D = 4, fs = SR / D, N = 4096, H = 2048;
    const x = decimate(mono, D);
    const fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), mag = new Float64Array(N / 2);
    const win = new Float64Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
    const nFr = Math.max(0, Math.floor((x.length - N) / H) + 1);
    const kLo = Math.ceil(60 * N / fs), kHi = Math.floor(2000 * N / fs);
    const pk = []; // フレームごとのピーク [midi, weight,...]
    let zr = 0, zi = 0;
    for (let f = 0; f < nFr; f++) {
      const s = f * H;
      for (let i = 0; i < N; i++) { re[i] = x[s + i] * win[i]; im[i] = 0; }
      fft.forward(re, im);
      let mx = 0;
      for (let k = kLo - 1; k <= kHi + 1; k++) { mag[k] = Math.hypot(re[k], im[k]); if (mag[k] > mx) mx = mag[k]; }
      const arr = [];
      if (mx > 1e-6) {
        for (let k = kLo; k <= kHi; k++) {
          const m = mag[k];
          if (m > mag[k - 1] && m >= mag[k + 1] && m > 0.05 * mx) {
            const a = Math.log(mag[k - 1] + 1e-12), b = Math.log(m), c = Math.log(mag[k + 1] + 1e-12);
            const d = 0.5 * (a - c) / (a - 2 * b + c);
            const fr = (k + d) * fs / N;
            const midi = 69 + 12 * Math.log2(fr / 440), w = Math.sqrt(m);
            arr.push(midi, w);
            const ph = 2 * Math.PI * (midi - Math.round(midi));
            zr += w * Math.cos(ph); zi += w * Math.sin(ph);
          }
        }
      }
      // 倍音の除去: 3倍音(5度上)・5倍音(3度上)などが音名の集計に混ざると、キーが属調(5度上)に間違えやすい
      if (arr.length > 2) {
        const np = arr.length / 2, w0 = new Float64Array(np);
        for (let i = 0; i < np; i++) w0[i] = arr[2 * i + 1];
        for (let i = 0; i < np; i++) {
          const m = arr[2 * i];
          for (let j = 0; j < np; j++) {
            if (j === i) continue;
            const d = m - arr[2 * j];
            if (d < 11) continue;
            for (const hd of [19.02, 27.86, 31.02, 33.69, 38.04]) if (Math.abs(d - hd) < 0.35 && w0[j] > 0.5 * w0[i]) { arr[2 * i + 1] *= 0.3; break; }
          }
        }
      }
      pk.push(arr);
      if (progress && (f & 127) === 0) progress(f / nFr);
    }
    const tuning = Math.atan2(zi, zr) / (2 * Math.PI) * 100; // cent
    // コード感(その瞬間カラオケで鳴っている音名)を時間ごとに: 歌の音を選ぶときに、伴奏とぶつかる音を避けるため
    const hc = new Float32Array(nFr * 12), hcE = new Float32Array(nFr);
    for (let f = 0; f < nFr; f++) {
      const arr = pk[f], o = f * 12;
      for (let i = 0; i < arr.length; i += 2) {
        const m = arr[i] - tuning / 100, r = Math.round(m), dev = Math.abs(m - r);
        const w = arr[i + 1] * Math.max(0, 1 - 2.5 * dev) * (m < 52 ? 1.6 : m > 84 ? 0.6 : 1); // ベースは和音を決めるので重め
        hc[o + (((r % 12) + 12) % 12)] += w;
      }
    }
    { const tmp = new Float32Array(hc.length);
      for (let f = 0; f < nFr; f++) for (let p = 0; p < 12; p++) { let t = 0, c = 0; for (let q = Math.max(0, f - 1); q <= Math.min(nFr - 1, f + 1); q++) { t += (q === f ? 2 : 1) * hc[q * 12 + p]; c += q === f ? 2 : 1; } tmp[f * 12 + p] = t / c; }
      for (let f = 0; f < nFr; f++) { let mx = 0; for (let p = 0; p < 12; p++) mx = Math.max(mx, tmp[f * 12 + p]); hcE[f] = mx; for (let p = 0; p < 12; p++) hc[f * 12 + p] = mx > 0 ? tmp[f * 12 + p] / mx : 0; } }
    // ブロック(約4秒)ごとのクロマ
    const BL = 22, nB = Math.ceil(nFr / BL), blockSec = BL * H / fs;
    const total = new Float64Array(12), blocks = [];
    for (let b = 0; b < nB; b++) {
      const ch = new Float64Array(12);
      for (let f = b * BL; f < Math.min(nFr, (b + 1) * BL); f++) {
        const arr = pk[f];
        for (let i = 0; i < arr.length; i += 2) {
          const m = arr[i] - tuning / 100, r = Math.round(m), dev = Math.abs(m - r);
          const w = arr[i + 1] * Math.max(0, 1 - 2 * dev);
          ch[((r % 12) + 12) % 12] += w;
        }
      }
      for (let i = 0; i < 12; i++) total[i] += ch[i];
      blocks.push(ch);
    }
    const g = scaleScores(total);
    let best = 0;
    for (let t = 1; t < 12; t++) if (g.out[t] > g.out[best]) best = t;
    // 転調検出: Viterbi
    const SWITCH = 1.6;
    let maxE = 0;
    const en = blocks.map(ch => { let e = 0; for (let i = 0; i < 12; i++) e += ch[i]; if (e > maxE) maxE = e; return e; });
    let cost = new Float64Array(12);
    const back = [];
    for (let b = 0; b < nB; b++) {
      const sc = scaleScores(blocks[b]).out, w = maxE > 0 ? Math.min(1, en[b] / (0.3 * maxE)) : 0;
      let mn = 0;
      for (let t = 1; t < 12; t++) if (cost[t] < cost[mn]) mn = t;
      const nc = new Float64Array(12), bp = new Uint8Array(12);
      for (let t = 0; t < 12; t++) {
        if (cost[t] <= cost[mn] + SWITCH) { nc[t] = cost[t]; bp[t] = t; } else { nc[t] = cost[mn] + SWITCH; bp[t] = mn; }
        nc[t] += -sc[t] * w;
      }
      cost = nc; back.push(bp);
    }
    const path = new Uint8Array(nB);
    if (nB > 0) {
      let cur = 0;
      for (let t = 1; t < 12; t++) if (cost[t] < cost[cur]) cur = t;
      for (let b = nB - 1; b >= 0; b--) { path[b] = cur; cur = back[b][cur]; }
    }
    const segs = [];
    for (let b = 0; b < nB; b++) {
      if (segs.length && segs[segs.length - 1].tonic === path[b]) segs[segs.length - 1].t1 = (b + 1) * blockSec;
      else segs.push({ t0: b * blockSec, t1: (b + 1) * blockSec, tonic: path[b] });
    }
    if (!segs.length) segs.push({ t0: 0, t1: 1e9, tonic: best });
    segs[0].t0 = 0; segs[segs.length - 1].t1 = 1e9;
    if (segs.length > 1) { let lg = segs[0]; for (const sg of segs) if (Math.min(sg.t1, nB * blockSec) - sg.t0 > Math.min(lg.t1, nB * blockSec) - lg.t0) lg = sg; best = lg.tonic; } // 表示するキー = いちばん長い部分のキー
    const eS = Array.from(hcE).filter(v => v > 0).sort((p, q) => p - q), eMed = eS.length ? eS[eS.length >> 1] : 0;
    for (let f = 0; f < nFr; f++) if (hcE[f] < 0.12 * eMed) for (let p = 0; p < 12; p++) hc[f * 12 + p] = 0; // 伴奏がほぼ無い所は使わない
    let gb = 0; for (let t2 = 1; t2 < 12; t2++) if (g.out[t2] > g.out[gb]) gb = t2; let second = -1e9; for (let t2 = 0; t2 < 12; t2++) if (t2 !== gb && g.out[t2] > second) second = g.out[t2];
    const conf = Math.max(0, Math.min(1, (g.out[gb] - second) / 0.15)) * (segs.length > 1 ? 0.7 : 1); // 2番目の候補(相関)との差。0.15以上離れていれば自信あり
    return { tuning, tonic: best, minor: g.mode[best] === 1, score: g.out[best], conf, segs, hc, hcDt: H / fs, hcT0: N / 2 / fs };
  }
  function keyName(tonic, minor) {
    const maj = NOTE_NAMES[tonic] + ' メジャー', min = NOTE_NAMES[(tonic + 9) % 12] + ' マイナー';
    return minor ? min + '（' + maj + '）' : maj + '（' + min + '）';
  }

  // ---------- 音程解析 (YIN + Viterbi) ----------
  function trackPitch(mono, progress) {
    const fs = SR / 2, hop = HOP / 2, W = 1024, N = 2048, K = 5;
    const tauMin = Math.floor(fs / 1000), tauMax = Math.ceil(fs / 65);
    const PADF = W / 2 + 64;
    const mh = Float32Array.from(mono); { const hc = biquad('hp', 60, 0.707); filt(mh, hc); } // v4.0: 解析用だけ。ブーン音・低い振動で音程を取り違えないように
    const xd = decimate(mh, 2);
    const nF = Math.floor(xd.length / hop) + 1;
    const x = new Float64Array(PADF + nF * hop + W + tauMax + 8);
    x.set(xd, PADF);
    const ps = new Float64Array(x.length + 1);
    for (let i = 0; i < x.length; i++) ps[i + 1] = ps[i] + x[i] * x[i];
    const fft = new FFT(N);
    const ar = new Float64Array(N), ai = new Float64Array(N), br = new Float64Array(N), bi = new Float64Array(N);
    const d = new Float64Array(tauMax + 2);
    const cTau = new Float32Array(nF * K), cCost = new Float32Array(nF * K), cN = new Uint8Array(nF);
    const rms = new Float32Array(nF);
    const cr1 = new Float64Array(tauMax + 3), cr2 = new Float64Array(tauMax + 3);
    const pick = (k, s, Ea, cor) => {
      let run = 0;
        d[0] = 1;
        for (let t = 1; t <= tauMax + 1; t++) {
          const dv = Math.max(0, Ea + (ps[s + t + W] - ps[s + t]) - 2 * cor[t] / N);
          run += dv;
          d[t] = run > 0 ? dv * t / run : 1;
        }
        let cn = 0;
        for (let t = tauMin; t <= tauMax; t++) {
          const v = d[t];
          if (v < 0.6 && v < d[t - 1] && v <= d[t + 1]) {
            const a = d[t - 1], c = d[t + 1], den = a - 2 * v + c;
            const off = den > 1e-12 ? 0.5 * (a - c) / den : 0;
            const tau = t + off, cv = v - 0.25 * (a - c) * off + 0.04 * Math.log2(tau / tauMin);
            if (cn < K) { cTau[k * K + cn] = tau; cCost[k * K + cn] = cv; cn++; }
            else {
              let w = 0;
              for (let j = 1; j < K; j++) if (cCost[k * K + j] > cCost[k * K + w]) w = j;
              if (cv < cCost[k * K + w]) { cTau[k * K + w] = tau; cCost[k * K + w] = cv; }
            }
          }
        }
        cN[k] = cn;
    };
    // 2フレームを1組にして FFT を半分に(実部=1つ目、虚部=2つ目)
    for (let k = 0; k < nF; k += 2) {
      const s = k * hop, s2 = s + hop, two = k + 1 < nF;
      const Ea = ps[s + W] - ps[s], Eb = two ? ps[s2 + W] - ps[s2] : 0;
      rms[k] = Math.sqrt(Ea / W); if (two) rms[k + 1] = Math.sqrt(Eb / W);
      const ua = Ea >= 1e-9, ub = two && Eb >= 1e-9;
      if (!ua && !ub) continue;
      ar.fill(0); ai.fill(0); br.fill(0); bi.fill(0);
      for (let i = 0; i < W; i++) { ar[i] = x[s + i]; ai[i] = two ? x[s2 + i] : 0; }
      for (let i = 0; i < W + tauMax; i++) { br[i] = x[s + i]; bi[i] = two ? x[s2 + i] : 0; }
      fft.forward(ar, ai); fft.forward(br, bi);
      // 分けて、相互スペクトル P1 = A1·conj(B1), P2 = A2·conj(B2) を作り、Z = P1 + i·P2
      for (let q = 0; q <= N / 2; q++) {
        const j = (N - q) % N;
        const a1r = 0.5 * (ar[q] + ar[j]), a1i = 0.5 * (ai[q] - ai[j]), a2r = 0.5 * (ai[q] + ai[j]), a2i = -0.5 * (ar[q] - ar[j]);
        const b1r = 0.5 * (br[q] + br[j]), b1i = 0.5 * (bi[q] - bi[j]), b2r = 0.5 * (bi[q] + bi[j]), b2i = -0.5 * (br[q] - br[j]);
        const p1r = a1r * b1r + a1i * b1i, p1i = a1i * b1r - a1r * b1i, p2r = a2r * b2r + a2i * b2i, p2i = a2i * b2r - a2r * b2i;
        ar[q] = p1r - p2i; ai[q] = p1i + p2r;
        if (q > 0 && q < N / 2) { ar[j] = p1r + p2i; ai[j] = -p1i + p2r; }
      }
      fft.forward(ar, ai);
      if (ua) pick(k, s, Ea, ar);
      if (ub) pick(k + 1, s2, Eb, ai);
      if (progress && (k & 254) === 0) progress(k / nF);
    }

    // ゲート
    const sorted = Float32Array.from(rms).sort();
    const gate = Math.max(1e-4, 0.015 * sorted[Math.floor(sorted.length * 0.95)]);
    // Viterbi
    const S = K + 1, UNV = 0.30, VU = 0.08;
    let prev = new Float64Array(S), cur = new Float64Array(S);
    const bp = new Uint8Array(nF * S);
    prev.fill(0);
    for (let k = 0; k < nF; k++) {
      const cn = rms[k] < gate ? 0 : cN[k];
      const pn = k > 0 ? (rms[k - 1] < gate ? 0 : cN[k - 1]) : 0;
      for (let j = 0; j < S; j++) {
        if (j < K && j >= cn) { cur[j] = 1e18; continue; }
        let bestC = 1e18, bestI = K;
        // from unvoiced
        let c = prev[K] + (j === K ? 0 : VU);
        if (c < bestC) { bestC = c; bestI = K; }
        for (let i = 0; i < pn; i++) {
          let tc;
          if (j === K) tc = VU;
          else {
            const st = Math.abs(12 * Math.log2(cTau[k * K + j] / cTau[(k - 1) * K + i]));
            tc = Math.min(0.3, 0.02 * st);
          }
          c = prev[i] + tc;
          if (c < bestC) { bestC = c; bestI = i; }
        }
        cur[j] = bestC + (j === K ? UNV : cCost[k * K + j]);
        bp[k * S + j] = bestI;
      }
      const t = prev; prev = cur; cur = t;
    }
    const f0 = new Float32Array(nF), ap = new Float32Array(nF).fill(1), comp = new Float32Array(nF).fill(1);
    let st = 0;
    for (let j = 1; j < S; j++) if (prev[j] < prev[st]) st = j;
    for (let k = nF - 1; k >= 0; k--) {
      f0[k] = st === K ? 0 : fs / cTau[k * K + st];
      if (st !== K) { ap[k] = cCost[k * K + st]; // 周期性の低さ(0=きれいな声 〜 大きいほど息っぽい・測りにくい)
        // v4.0 候補の競合: 別の高さ(1半音以上ちがう)の候補が、ほぼ同じくらい良いなら「どちらか分からない」
        let alt = 1e9; for (let j = 0; j < cN[k]; j++) { if (j === st) continue; const r = cTau[k * K + j] / cTau[k * K + st]; if (Math.abs(12 * Math.log2(r)) < 1) continue; if (r > 1.5 && Math.abs(r - Math.round(r)) < 0.04) continue; /* 2倍・3倍の周期(低いオクターブ)は、周期的な声ならいつも候補に出るので競合に数えない */ alt = Math.min(alt, cCost[k * K + j]); }
        comp[k] = Math.max(0, Math.min(1, (alt - cCost[k * K + st]) / 0.12)); }
      st = bp[k * S + st];
    }
    // 短すぎる有声区間を除去
    let a = -1;
    for (let k = 0; k <= nF; k++) {
      const v = k < nF && f0[k] > 0;
      if (v && a < 0) a = k;
      if (!v && a >= 0) { if (k - a < 4) for (let i = a; i < k; i++) f0[i] = 0; a = -1; }
    }
    // オクターブの誤検出を直す: まわり(±12フレーム)の中央値から8半音以上はなれた所は、オクターブ単位で寄せる
    { let a2 = -1; for (let k = 0; k <= nF; k++) { const v = k < nF && f0[k] > 0; if (v && a2 < 0) a2 = k; if (!v && a2 >= 0) { if (k - a2 >= 10) { const m = []; for (let i = a2; i < k; i++) m.push(12 * Math.log2(f0[i] / 440)); for (let i = a2; i < k; i++) { const w = []; for (let j = Math.max(a2, i - 12); j < Math.min(k, i + 13); j++) w.push(m[j - a2]); w.sort((x, y) => x - y); const md = w[w.length >> 1], d = m[i - a2] - md; if (Math.abs(d) > 8) { const o = Math.round(d / 12); if (o) { f0[i] *= Math.pow(2, -o); ap[i] = Math.max(ap[i], 0.3); } } } } a2 = -1; } } }
    // 母音の立ち上がり検出用: 1kHz以下の音量包絡(5.8ms刻み)
    const lpx = new Float32Array(mono), lpc = biquad('lp', 1000, 0.707);
    filt(lpx, lpc); filt(lpx, lpc);
    const amp = new Float32Array(Math.floor(lpx.length / 256));
    for (let i = 0; i < amp.length; i++) { let e = 0; for (let j = i * 256; j < i * 256 + 256; j++) e += lpx[j] * lpx[j]; amp[i] = Math.sqrt(e / 256); }
    const conf = pitchConf(xd, fs, hop, f0, ap, comp);
    return { f0, amp, ap, conf, hopSec: HOP_SEC };
  }
  // v4.0 音程の信頼度(0〜1): 周期性 × 候補の競合 × 時間の連続性 × 倍音の整合
  //   倍音の整合: 選んだ f0 の整数倍に音が集まっているか(雑音・息)、奇数倍音だけ極端に弱くないか(本当は倍の高さ)、
  //   f0 の半分・1.5倍の所に強い音がないか(本当は半分の高さ) を見る。どれか1つで切り捨てず、掛け合わせて弱める
  function pitchConf(xd, fs, hop, f0, ap, comp) {
    const nF = f0.length, conf = new Float32Array(nF), N = 2048, fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), win = new Float64Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
    const hcon = new Float32Array(nF).fill(-1), bin = fs / N;
    const band = (f, w) => { const hw = Math.max(w * f, 2.5 * bin), a = Math.max(1, Math.floor((f - hw) / bin)), b = Math.min(N / 2 - 1, Math.ceil((f + hw) / bin)); let e = 0; for (let q = a; q <= b; q++) e += re[q]; return e; };
    for (let k = 0; k < nF; k += 2) {
      const f = f0[k] > 0 ? f0[k] : (k + 1 < nF && f0[k + 1] > 0 ? f0[k + 1] : 0); if (!(f > 0)) continue;
      const c = k * hop; // フレーム k の中心(曲の時間 k*HOP_SEC)
      const s0 = c - N / 2; if (s0 < 0 || s0 + N > xd.length) continue;
      for (let i = 0; i < N; i++) { re[i] = xd[s0 + i] * win[i]; im[i] = 0; }
      fft.forward(re, im); for (let q = 0; q < N / 2; q++) re[q] = re[q] * re[q] + im[q] * im[q];
      let tot = 0; const lo = Math.floor(Math.max(70, 0.7 * f) / bin), hi = Math.min(N / 2 - 1, Math.floor(Math.min(5000, 10.5 * f) / bin)); for (let q = lo; q <= hi; q++) tot += re[q];
      if (!(tot > 1e-12)) continue;
      let har = 0, odd = 0, even = 0; const nh = Math.max(1, Math.min(10, Math.floor(5000 / f)));
      for (let h = 1; h <= nh; h++) { const e = band(h * f, 0.03); har += e; if (h <= 6) (h % 2 ? (odd += e) : (even += e)); }
      const sub = band(0.5 * f, 0.03) + band(1.5 * f, 0.03), hr = har / tot;
      let v = Math.max(0, Math.min(1, (hr - 0.25) / 0.35)); // 倍音に集まっている割合
      if (nh >= 4 && odd < 0.08 * even) v *= 0.5; // 奇数倍音がほとんどない → 本当は倍の高さかも
      if (sub > 0.35 * band(f, 0.03)) v *= 0.5; // 半分の高さの倍音が強い → 本当は半分の高さかも
      hcon[k] = v; if (k + 1 < nF) hcon[k + 1] = v;
    }
    for (let k = 0; k < nF; k++) {
      if (!(f0[k] > 0)) continue;
      const per = Math.max(0, Math.min(1, (0.42 - ap[k]) / 0.3)); // 周期性
      let cont = 1; for (const d of [-1, 1]) { const j = k + d; if (j >= 0 && j < nF && f0[j] > 0) { const jump = Math.abs(12 * Math.log2(f0[j] / f0[k])); if (jump > 1.5) cont *= 0.6; } } // 1フレームで1.5半音以上の跳び
      const hc = hcon[k] < 0 ? 0.7 : hcon[k];
      conf[k] = per * (0.4 + 0.6 * comp[k]) * cont * (0.3 + 0.7 * hc);
    }
    return conf;
  }
  // v4.0 声の安定性(0〜1): となり合う声帯周期どうしの波形の似ている度合い。声が乱れている所(がらがら声・裏返り・不規則な振動)は低い
  function voiceStability(x, f0) {
    const nF = f0.length, out = new Float32Array(nF);
    for (let k = 0; k < nF; k++) {
      if (!(f0[k] > 0)) continue;
      const T = Math.round(SR / f0[k]), c = k * HOP; if (c - 2 * T < 0 || c + 2 * T >= x.length) continue;
      let best = -1; // 1周期先(±6%)と比べて、いちばん似ている所の相関
      for (let L = Math.round(T * 0.94); L <= Math.round(T * 1.06); L++) { let sxy = 0, sxx = 0, syy = 0; for (let i = c - T; i < c; i += 2) { const a = x[i], b = x[i + L]; sxy += a * b; sxx += a * a; syy += b * b; } const r = sxy / Math.sqrt(sxx * syy + 1e-18); if (r > best) best = r; }
      out[k] = Math.max(0, best);
    }
    return out;
  }

  // ---------- 歌とお手本の時間合わせ(DTW) ----------
  // 走ったり(早く)もたったり(遅く)した所も、お手本のどの音を歌っているのかを対応づける
  // 戻り値: 歌のフレーム k → お手本のフレーム(曲の時間)
  function alignToRef(midi, refNotes, offF, band) {
    const n = midi.length, m = refNotes.length, B = band || 130, W = 2 * B + 1, INF = 1e9;
    const D2 = new Float32Array(n * W).fill(INF), bp = new Uint8Array(n * W); // 0:斜め 1:横(お手本だけ進む) 2:縦(歌だけ進む)
    // 歌の高さの中心(オクターブちがいで歌っていても合わせる)
    const cost = (k, j) => {
      const a = midi[k], b = j >= 0 && j < m ? refNotes[j] : NaN, va = a === a, vb = b === b;
      if (!va && !vb) return 0; if (va !== vb) return 0.55;
      let d = (a - b) % 12; if (d > 6) d -= 12; if (d < -6) d += 12; d = Math.abs(d);
      return Math.min(1, d / 2.5);
    };
    for (let k = 0; k < n; k++) {
      const c0 = k + offF;
      for (let w = 0; w < W; w++) {
        const j = c0 - B + w; if (j < -1 || j > m) continue;
        const c = cost(k, j);
        if (k === 0) { D2[w] = c + Math.abs(w - B) * 0.002; continue; }
        // 前の行 k-1 の同じ j → 歌だけ進む(w+1 の位置), j-1 → 斜め(w), 同じ行の j-1 → お手本だけ進む
        let best = INF, b = 0;
        const dg = D2[(k - 1) * W + w]; if (dg < best) { best = dg + c; b = 0; }
        if (w + 1 < W) { const dv = D2[(k - 1) * W + w + 1] + c + 0.25; if (dv < best) { best = dv; b = 2; } }
        if (w > 0) { const dh = D2[k * W + w - 1] + c + 0.25; if (dh < best) { best = dh; b = 1; } }
        D2[k * W + w] = best; bp[k * W + w] = b;
      }
    }
    let w = 0, bv = INF; for (let q = 0; q < W; q++) if (D2[(n - 1) * W + q] < bv) { bv = D2[(n - 1) * W + q]; w = q; }
    const map = new Float32Array(n); let k = n - 1;
    const seen = new Float32Array(n).fill(-1);
    while (k >= 0) {
      const j = k + offF - B + w; if (seen[k] < 0) seen[k] = j;
      const b = bp[k * W + w];
      if (k === 0 && b !== 1) break;
      if (b === 0) k--; else if (b === 2) { k--; w++; } else { w--; if (w < 0) break; }
    }
    for (let q = 0; q < n; q++) map[q] = seen[q] >= 0 ? seen[q] : q + offF;
    // ずれ(お手本 - 歌)を前後0.45秒の中央値でならす: 外した音を「となりの音」に合わせてしまう細かいゆがみを消し、フレーズごとの走り/もたりだけ残す
    const R = 39, dv = new Float32Array(n), sm = new Float32Array(n), tmp = [];
    for (let q = 0; q < n; q++) dv[q] = map[q] - q;
    for (let q = 0; q < n; q++) { tmp.length = 0; for (let j = Math.max(0, q - R); j <= Math.min(n - 1, q + R); j++) if (midi[j] === midi[j]) tmp.push(dv[j]); if (!tmp.length) { sm[q] = dv[q]; continue; } tmp.sort((a, c) => a - c); sm[q] = tmp[tmp.length >> 1]; }
    for (let q = 0; q < n; q++) map[q] = q + sm[q];
    return map;
  }

  // ---------- 補正カーブの計画 ----------
  function planCorrection(f0, key, opts) {
    const nF = f0.length, strength = opts.strength, chromatic = opts.mode === 'chromatic';
    const A4 = 440 * Math.pow(2, (key ? key.tuning : 0) / 1200);
    const midi = new Float32Array(nF), target = new Float32Array(nF), shift = new Float32Array(nF), cfull = new Float32Array(nF), tonic = new Int8Array(nF);
    const hard = !!opts.hard;
    // ノートごとの手直し: dn=高さ(半音), amt=合わせる強さ, mod=揺れの大きさ
    const edits = (opts.edits || []).slice().sort((a, b) => a.k - b.k), notesOut = [];
    const slowA = new Float32Array(nF), amtA = new Float32Array(nF), modA = new Float32Array(nF).fill(1 - (opts.drift || 0)), dnA = new Float32Array(nF), drfA = new Float32Array(nF), meanA = new Float32Array(nF);
    // お手本の歌(ref.notes: 曲の時間のフレームごとの音)があれば、その音に強く引き寄せる。音痴で2〜3半音ずれていても正しい音に直せる
    const ref = opts.ref && opts.ref.notes ? opts.ref : null, REFPEN = 2.6, REFW = 16;
    let refMap = null;
    if (ref && !opts.noAlign) { // 時間合わせ: 走った/もたった所も、お手本の同じ音に対応させる
      const md = new Float32Array(nF); for (let k = 0; k < nF; k++) md[k] = f0[k] > 0 ? 69 + 12 * Math.log2(f0[k] / A4) : NaN;
      refMap = alignToRef(md, ref.notes, Math.round((opts.offsetSec || 0) / HOP_SEC), 130);
    }
    const refAt = (k, sungS) => { // 対応するお手本の音(なければ前後0.18秒で一番近い音)。歌っている高さに近いオクターブへ
      const j0 = refMap ? Math.round(refMap[k]) : Math.round(k + (opts.offsetSec || 0) / HOP_SEC), rn = ref.notes, rs = opts.refShift || 0; // rs: カラオケのキーを変えたぶん
      for (let d = 0; d <= REFW; d++) for (const j of d ? [j0 - d, j0 + d] : [j0]) { if (j < 0 || j >= rn.length) continue; const v = rn[j] + rs; if (v === v) return v + 12 * Math.round((sungS - v) / 12); }
      return NaN;
    };
    const findEdit = (k0, k1) => { for (const e of edits) { if (e.k >= k1) break; if (e.k >= k0) return e; } return null; };
    for (let k = 0; k < nF; k++) { midi[k] = f0[k] > 0 ? 69 + 12 * Math.log2(f0[k] / A4) : NaN; target[k] = NaN; }
    // 歌い手のくせ(いつも少し低め/高め)を先に見つけて、音を選ぶときだけ差し引く
    let bias = 0;
    if (!ref) {
      let zr = 0, zi = 0, c = 0;
      for (let k = 2; k < nF - 2; k++) { const m = midi[k]; if (m === m && midi[k - 2] === midi[k - 2] && midi[k + 2] === midi[k + 2] && Math.abs(midi[k + 2] - midi[k - 2]) < 0.12) { const a2 = 2 * Math.PI * (m - Math.round(m)); zr += Math.cos(a2); zi += Math.sin(a2); c++; } }
      const R = c ? Math.hypot(zr, zi) / c : 0, b = Math.atan2(zi, zr) / (2 * Math.PI);
      if (opts.dbgBias) opts.dbgBias.push(b, R, c);
      if (c > 200 && Math.abs(b) > 0.08 && R > (Math.abs(b) > 0.25 ? 0.3 : 0.15)) bias = b; // 毎回ばらばらにずれる人(R小)は、くせとは見なさない
    }
    const ksegs = key ? key.segs : null;
    let si = 0;
    function tonicAt(k) {
      if (opts.tonic != null) return opts.tonic;
      if (!ksegs) return 0;
      const t = k * HOP_SEC + (opts.offsetSec || 0);
      while (si + 1 < ksegs.length && t >= ksegs[si].t1) si++;
      while (si > 0 && t < ksegs[si].t0) si--;
      return ksegs[si].tonic;
    }
    const inScale = (n, tn) => chromatic || MAJOR[(((n - tn) % 12) + 12) % 12] === 1;
    const HC = !ref && key && key.hc && key.hc.length && opts.harmony !== false ? key.hc : null, hcN = HC ? HC.length / 12 : 0;
    const harmAt = (k0, k1) => { // 区間で伴奏が鳴らしている音名の強さ(0〜1)。伴奏が無ければ null
      if (!HC) return null;
      const t0 = k0 * HOP_SEC + (opts.offsetSec || 0), t1 = k1 * HOP_SEC + (opts.offsetSec || 0);
      const f0i = Math.max(0, Math.round((t0 - key.hcT0) / key.hcDt)), f1i = Math.min(hcN - 1, Math.round((t1 - key.hcT0) / key.hcDt));
      if (f1i < f0i) return null;
      const h = new Float32Array(12); let any = 0;
      for (let f = f0i; f <= f1i; f++) for (let p = 0; p < 12; p++) { h[p] += HC[f * 12 + p]; any += HC[f * 12 + p]; }
      if (!(any > 0)) return null;
      let mx = 0; for (let p = 0; p < 12; p++) mx = Math.max(mx, h[p]); for (let p = 0; p < 12; p++) h[p] /= mx;
      return h;
    };
    // 他のパート(メイン・ほかのハモリ)と一緒に鳴る音。ハモリの音を選ぶとき、それと半音/全音でぶつからない方を選ぶ
    const AG = opts.against && opts.against.length ? opts.against.slice().sort((x, y) => x.t0 - y.t0) : null;
    const agCost = (n, ts0, ts1) => { if (!AG) return 0; let c = 0, lo = 0, hi = AG.length;
      while (lo < hi) { const md = (lo + hi) >> 1; if (AG[md].t0 < ts0 - 8) lo = md + 1; else hi = md; }
      for (let q = lo; q < AG.length && AG[q].t0 < ts1; q++) { const g = AG[q], ov = Math.min(ts1, g.t1) - Math.max(ts0, g.t0); if (ov <= 0) continue;
        const w = ov / Math.max(0.05, ts1 - ts0), iv = (((n - g.n) % 12) + 12) % 12;
        c += w * (iv === 1 || iv === 11 ? 0.9 : iv === 2 || iv === 10 ? 0.35 : iv === 6 ? 0.35 : iv === 3 || iv === 4 || iv === 8 || iv === 9 ? -0.25 : iv === 0 ? 0.2 : 0); }
      return c; };
    const HW = 0.45, CW = 0.4; // 伴奏にある音を選びやすく / 伴奏の音と半音でぶつかる音を避ける
    const harmCost = (n, h) => { const pc = ((n % 12) + 12) % 12, nb = Math.max(h[(pc + 1) % 12], h[(pc + 11) % 12]); return -HW * h[pc] + CW * Math.max(0, nb - h[pc]); };
    // くせが半音の半分近く(例: -0.45)だと「低め」か「半音上を高めに歌っている」か分からない → 音階と伴奏に合う方を選ぶ
    if (Math.abs(bias) > 0.25) {
      const alt = bias - Math.sign(bias), fit = bb => { let sc = 0, c = 0;
        for (let k = 2; k < nF - 2; k += 2) { const m = midi[k]; if (!(m === m) || !(midi[k - 2] === midi[k - 2]) || !(midi[k + 2] === midi[k + 2]) || Math.abs(midi[k + 2] - midi[k - 2]) > 0.12) continue;
          const n = Math.round(m - bb); sc += inScale(n, tonicAt(k)) ? 1 : 0;
          if (HC) { const f = Math.round((k * HOP_SEC + (opts.offsetSec || 0) - key.hcT0) / key.hcDt); if (f >= 0 && f < hcN) sc += HC[f * 12 + ((n % 12) + 12) % 12]; }
          c++; }
        return c ? sc / c : 0; };
      si = 0; const f1 = fit(bias); si = 0; const f2 = fit(alt); si = 0;
      if (f2 > f1 + 0.05) bias = alt; else if (!(f1 > f2 + 0.05) && Math.abs(bias) > 0.4) bias = 0;
      if (Math.abs(bias) > 0.7) bias = 0;
    }
    const PEN = 0.5, SW = 2.0, CL = ref ? 4 : 2, drift = opts.drift || 0;
    const autoN = opts.autoNotes || 0, autoFix = [], lowConf = [], natDz = opts.deadzone || 0, skipWhy = { unsure: 0, unstable: 0, bigNoRef: 0, breathy: 0 };
    const medAround = (a, k, r, lo, hi) => { const v = []; for (let j = Math.max(lo, k - r); j < Math.min(hi, k + r + 1); j++) if (a[j] === a[j]) v.push(a[j]); v.sort((x, y) => x - y); return v.length ? v[v.length >> 1] : a[k]; };
    const nat = opts.natural == null ? 0 : opts.natural, natFl = opts.natFl == null ? 0.5 : opts.natFl; // nat: 自然な補正(ノートの中心だけ動かす)の割合 / natFl: 長い音のふらつきを戻す量
    const speed = opts.speed == null ? 0.7 : opts.speed; // 補正の速さ: 歌い出しのしゃくり・語尾のフォールをどれだけ直すか
    const WIN = 7; // 中心の音程 = 前後7フレーム(約0.17秒 = ビブラート1周期)の平均 → ビブラートは残してふらつきは直す
    const amp = opts.amp || null, ampAt = k => amp ? Math.max(amp[2 * k] || 0, amp[2 * k + 1] || 0) : 0;
    let sumErr = 0, nV = 0, a = -1;
    for (let k = 0; k <= nF; k++) {
      const v = k < nF && f0[k] > 0;
      if (v && a < 0) a = k;
      if (v || a < 0) continue;
      const L = k - a, base = a;
      a = -1;
      // 平滑化(メディアン5 → 平均5)
      const med = new Float32Array(L), s = new Float32Array(L), tmp = [];
      for (let i = 0; i < L; i++) {
        tmp.length = 0;
        for (let j = Math.max(0, i - 2); j <= Math.min(L - 1, i + 2); j++) tmp.push(midi[base + j]);
        tmp.sort((p, q) => p - q); med[i] = tmp[tmp.length >> 1];
      }
      let lo = 1e9, hi = -1e9;
      for (let i = 0; i < L; i++) {
        let sm = 0, c = 0;
        for (let j = Math.max(0, i - 2); j <= Math.min(L - 1, i + 2); j++) { sm += med[j]; c++; }
        s[i] = sm / c - bias; if (s[i] < lo) lo = s[i]; if (s[i] > hi) hi = s[i];
      }
      const rnA = ref ? new Float32Array(L).map((_, i) => refAt(base + i, s[i])) : null;
      if (rnA) for (let i = 0; i < L; i++) { const v2 = rnA[i]; if (v2 === v2) { if (v2 < lo) lo = v2; if (v2 > hi) hi = v2; } }
      const nLo = Math.floor(lo) - 1, S = Math.min(250, Math.ceil(hi) + 1 - nLo + 1);
      // (1) 音の高さの変わり目(DP): 大まかなノートの並び
      let prev = new Float64Array(S), cur = new Float64Array(S);
      const bp = new Uint8Array(L * S), dpn = new Int16Array(L);
      for (let i = 0; i < L; i++) {
        const tn = tonicAt(base + i); tonic[base + i] = tn;
        let mn = 0;
        for (let j = 1; j < S; j++) if (prev[j] < prev[mn]) mn = j;
        for (let j = 0; j < S; j++) {
          const n = nLo + j;
          let e = Math.abs(s[i] - n);
          if (rnA && rnA[i] === rnA[i]) { if (n !== rnA[i]) e += REFPEN; }
          else if (!inScale(n, tn)) e += PEN;
          if (i > 0 && prev[mn] + SW < prev[j]) { cur[j] = prev[mn] + SW + e; bp[i * S + j] = mn; }
          else { cur[j] = prev[j] + e; bp[i * S + j] = j; }
        }
        const t = prev; prev = cur; cur = t;
      }
      let stt = 0;
      for (let j = 1; j < S; j++) if (prev[j] < prev[stt]) stt = j;
      for (let i = L - 1; i >= 0; i--) { dpn[i] = nLo + stt; stt = bp[i * S + stt]; }
      // (2) 区切り: 声の大きさのくぼみ(言葉の区切り)と、安定した音の変わり目(しゃくりの始まりまでさかのぼる)
      const cuts = new Map(); // フレーム → 'dip' | 'pitch'
      if (amp) {
        const av = new Float32Array(L); for (let i = 0; i < L; i++) av[i] = ampAt(base + i);
        const sm = new Float32Array(L); for (let i = 0; i < L; i++) { let t = 0, c = 0; for (let j = Math.max(0, i - 1); j <= Math.min(L - 1, i + 1); j++) { t += av[j]; c++; } sm[i] = t / c; }
        for (let i = 4; i < L - 4; i++) {
          let isMin = true; for (let j = i - 3; j <= i + 3; j++) if (sm[j] < sm[i]) { isMin = false; break; }
          if (!isMin) continue;
          let pb = 0, pa = 0; for (let j = Math.max(0, i - 10); j < i; j++) pb = Math.max(pb, sm[j]); for (let j = i + 1; j <= Math.min(L - 1, i + 10); j++) pa = Math.max(pa, sm[j]);
          if (sm[i] < 0.55 * pb && sm[i] < 0.55 * pa) { let j = i + 1; const th = sm[i] + 0.35 * (pa - sm[i]); while (j < L - 1 && sm[j] < th) j++; cuts.set(j, 'dip'); }
        }
      }
      for (let i = 1; i < L; i++) if (dpn[i] !== dpn[i - 1]) {
        let la = i, lb = i; while (la > 0 && dpn[la - 1] === dpn[i - 1]) la--; while (lb < L && dpn[lb] === dpn[i]) lb++;
        if (i - la < 10 || lb - i < 6) continue; // 短い揺れ・しゃくりは区切りにしない
        // 音が落ち着いているか: 揺れ幅がビブラート程度(±0.45)で、前半と後半で高さが動いていない(しゃくり・フォールではない)
        const steady = (x0, x1) => { if (x1 - x0 < 4) return false; let mn = 1e9, mx = -1e9, a = 0, b = 0; const h = (x0 + x1) >> 1; for (let q = x0; q < x1; q++) { if (s[q] < mn) mn = s[q]; if (s[q] > mx) mx = s[q]; if (q < h) a += s[q]; else b += s[q]; } return mx - mn < 1.0 && Math.abs(a / (h - x0) - b / (x1 - h)) < 0.35; };
        if (!steady(Math.max(la, i - 16), i - 3) || !steady(i + 3, Math.min(lb, i + 16))) continue; // しゃくり・フォール(音が動いている所)は前後のノートの一部
        const old = dpn[i - 1]; let j = i; // 前の音から離れ始めた所まで戻る
        while (j > la + 3 && Math.abs(s[j - 1] - old) > 0.35) j--;
        let near = false; for (const c of cuts.keys()) if (Math.abs(c - j) < 7) { near = true; break; }
        if (!near) cuts.set(j, 'pitch');
      }
      // 手で「つなげた」所の区切りは消し、手で「分けた」所には区切りを入れる(ノート編集)
      if (opts.joins) for (const uj of opts.joins) { const r = uj - base; if (r <= 0 || r >= L) continue; for (const c0 of [...cuts.keys()]) if (Math.abs(c0 - r) <= 3) cuts.delete(c0); }
      const user = new Set(); if (opts.cuts) for (const uc of opts.cuts) { const r = uc - base; if (r >= 2 && r <= L - 2) { for (const c0 of [...cuts.keys()]) if (Math.abs(c0 - r) <= 2) cuts.delete(c0); cuts.set(r, 'user'); user.add(r); } }
      const bnd = [0].concat([...cuts.keys()].filter(c => user.has(c) || (c >= 4 && c <= L - 4)).sort((x, y) => x - y)), segs = [];
      for (let q = 0; q < bnd.length; q++) { const s0 = bnd[q], s1 = q + 1 < bnd.length ? bnd[q + 1] : L; if (s1 - s0 >= 1) segs.push([s0, s1]); }
      for (let q = segs.length - 1; q > 0; q--) if (segs[q][1] - segs[q][0] < 4 && !user.has(segs[q][0]) && !user.has(segs[q][1])) { segs[q - 1][1] = segs[q][1]; segs.splice(q, 1); }
      if (segs.length > 1 && segs[0][1] - segs[0][0] < 4 && !user.has(segs[0][1])) { segs[1][0] = 0; segs.shift(); }
      // (2b) v3.1: 1つにまとめられたノートの中に「別の高さで落ち着いている所」が2つ以上あれば分ける
      //   つなげて歌った音(音量のくぼみが無い)は1ノートになりやすく、真ん中の高さが音と音の間になって補正できなかった
      //   ビブラートは約0.17秒の平均でならしてから見るので、ビブラートでは分かれない。手で「つなげた」所は分けない
      const splitTh = opts.splitTh == null ? 0.7 : opts.splitTh, minRun = opts.splitRun || 6, stabTh = opts.splitStab || 0.4;
      if (splitTh < 5 && L >= 2 * minRun) {
        const sl = new Float32Array(L); for (let i = 0; i < L; i++) { let a2 = 0, c2 = 0; for (let j = Math.max(0, i - 5); j <= Math.min(L - 1, i + 5); j++) { a2 += s[j]; c2++; } sl[i] = a2 / c2; }
        const joinNear = r => opts.joins && opts.joins.some(uj => Math.abs(uj - base - r) <= 6);
        for (let q = 0; q < segs.length; q++) {
          const [s0, s1] = segs[q]; if (s1 - s0 < 2 * minRun) continue;
          const runs = []; let r0 = -1; // 落ち着いている所: 前後4フレームでの動きが小さい
          for (let i = s0; i <= s1; i++) { const st = i < s1 && Math.abs(sl[Math.min(s1 - 1, i + 3)] - sl[Math.max(s0, i - 3)]) < stabTh; if (st && r0 < 0) r0 = i; if (!st && r0 >= 0) { if (i - r0 >= minRun) { const v = []; for (let j = r0; j < i; j++) v.push(sl[j]); v.sort((x, y) => x - y); runs.push([r0, i, v[v.length >> 1]]); } r0 = -1; } }
          const edgeMed = (a2, b2) => { const v = []; for (let j = a2; j < b2; j++) v.push(s[j]); v.sort((x, y) => x - y); return v[v.length >> 1]; };
          for (let again = 1, guard = 0; again && guard < 50; guard++) { again = 0; for (let r = runs.length - 1; r > 0; r--) { const A2 = runs[r - 1], B2 = runs[r], jump = Math.abs(edgeMed(B2[0], Math.min(B2[1], B2[0] + 6)) - edgeMed(Math.max(A2[0], A2[1] - 6), A2[1]));
            if (Math.abs(B2[2] - A2[2]) < splitTh || jump < 0.8 * splitTh) { A2[1] = B2[1]; const v = []; for (let j = A2[0]; j < A2[1]; j++) v.push(sl[j]); v.sort((x, y) => x - y); A2[2] = v[v.length >> 1]; runs.splice(r, 1); again = 1; break; } } } // 近い高さ・ゆっくり移った(ふらつき)はまとめる
          for (let r = runs.length - 2; r > 0; r--) if (runs[r][1] - runs[r][0] < 2 * minRun + 4 && Math.abs(runs[r - 1][2] - runs[r + 1][2]) < splitTh) { runs[r - 1][1] = runs[r + 1][1]; runs.splice(r, 2); r = Math.min(r, runs.length - 1); } // 一瞬はなれて同じ高さに戻る所(声の裏返り・音飛び)は分けない
          if (runs.length < 2) continue;
          const cutsIn = [];
          for (let r = 1; r < runs.length; r++) { const a2 = runs[r - 1], b2 = runs[r], mid = (a2[2] + b2[2]) / 2; let c = a2[1]; // 高さが真ん中を横切る所で分ける(前の音から離れ始めたら次の音: しゃくりは次の音の一部)
            for (let i = a2[1]; i < b2[0]; i++) if ((s[i] - mid) * (b2[2] - a2[2]) >= 0) { c = i; break; }
            let c2 = c; while (c2 > a2[1] && Math.abs(s[c2 - 1] - a2[2]) > 0.35) c2--; c = Math.max(a2[1], Math.min(c, c2 + 2));
            if (c - s0 >= 4 && s1 - c >= 4 && !joinNear(c)) cutsIn.push(c); }
          if (opts.dbgSplit) opts.dbgSplit.push({ base, s0, s1, runs: runs.map(r => [r[0], r[1], +r[2].toFixed(2)]), cutsIn });
          if (!cutsIn.length) continue;
          const parts = []; let p = s0; for (const c of cutsIn) { parts.push([p, c]); p = c; } parts.push([p, s1]);
          segs.splice(q, 1, ...parts); q += parts.length - 1;
        }
      }
      // (3) ノートごとの音: しゃくり・語尾を除いた真ん中の部分の高さで決める(大きく外した所に引っぱられない)
      const notes = new Int16Array(L);
      for (const [s0, s1] of segs) {
        const len = s1 - s0, c0 = s0 + Math.floor(len * 0.3), c1 = Math.max(c0 + 1, s1 - Math.floor(len * 0.15));
        const core = []; for (let i = c0; i < c1; i++) core.push(s[i]); core.sort((x, y) => x - y);
        const ctr = core[core.length >> 1], mid = (c0 + c1) >> 1, tn = tonicAt(base + mid), rn = rnA ? rnA[mid] : NaN;
        const hh = rn === rn ? null : harmAt(base + c0, base + c1);
        let best = Math.round(ctr), bc = 1e9;
        for (let n = Math.round(ctr) - 3; n <= Math.round(ctr) + 3; n++) {
          let e = Math.abs(ctr - n);
          if (rn === rn) { if (n !== rn) e += REFPEN; }
          else {
            if (!inScale(n, tn)) e += hh ? PEN * Math.max(0, 1 - 1.4 * hh[((n % 12) + 12) % 12]) : PEN; // 伴奏がはっきり鳴らしている音なら、キーの外の音でもOK
            if (hh) e += harmCost(n, hh);
          }
          if (AG) e += agCost(n, (base + c0) * HOP_SEC + (opts.offsetSec || 0), (base + c1) * HOP_SEC + (opts.offsetSec || 0));
          // DPが長く選んだ音なら少し優先(ゆっくり音を変える歌い方のため)
          let agree = 0; for (let i = c0; i < c1; i++) if (dpn[i] === n) agree++;
          e -= 0.15 * agree / (c1 - c0);
          if (e < bc) { bc = e; best = n; }
        }
        for (let i = s0; i < s1; i++) notes[i] = best;
      }
      // (4) ノートごとの補正量: 中心の音程とのズレ。歌い出し・語尾は速く追いかけて、しゃくり・フォールも直す
      const pre = new Float64Array(L + 1), c = new Float32Array(L), cn = new Float32Array(L), manual = new Uint8Array(L);
      for (let i = 0; i < L; i++) pre[i + 1] = pre[i] + midi[base + i];
      for (let q = 0; q < segs.length; q++) {
        const [p0, q0] = segs[q], len = q0 - p0;
        const e = findEdit(base + p0, base + q0), dn = e && e.dn ? e.dn : 0;
        const am = e && e.amt != null ? e.amt : (hard || (e && e.dn) || (e && strength === 0 && (e.tr != null || e.drf || (e.mod != null && e.mod !== 1))) ? 1 : strength), md = e && e.mod != null ? e.mod : 1 - drift;
        const Z = Math.round(speed * Math.min(12, len / 3)), Zr = q === segs.length - 1 ? Math.round(speed * Math.min(9, len / 4)) : 0;
        let cen = 0;
        for (let i = p0; i < q0; i++) {
          const u = Math.max(p0, i - WIN), w = Math.min(q0 - 1, i + WIN);
          let slow = (pre[w + 1] - pre[u]) / (w - u + 1);
          const zw = i - p0 < Z ? 1 - (i - p0) / Z : (q0 - 1 - i < Zr ? 1 - (q0 - 1 - i) / Zr : 0);
          if (zw > 0) { const u2 = Math.max(p0, i - 1), w2 = Math.min(q0 - 1, i + 1), fast = (pre[w2 + 1] - pre[u2]) / (w2 - u2 + 1); slow = slow * (1 - zw) + fast * zw; }
          slowA[base + i] = slow; amtA[base + i] = am; modA[base + i] = md; dnA[base + i] = dn; cen += slow;
          const tgt = notes[p0] + dn;
          const lim = zw > 0 ? Math.max(CL, 2.5) : CL;
          c[i] = hard ? tgt - midi[base + i] : (e ? tgt - slow : Math.max(-lim, Math.min(lim, tgt - slow)));
        }
        { // 自然な補正: ノートの「真ん中の高さ」だけを正しい音へ平行移動。しゃくり・ビブラート・音の移り方は歌ったまま残す(ケロケロしない)
          const a0 = p0 + Math.floor(len * 0.3), a1 = Math.max(a0 + 1, q0 - Math.floor(len * 0.15)), vs = []; for (let i = a0; i < a1; i++) vs.push(midi[base + i]); vs.sort((x, y) => x - y);
          const center = vs[vs.length >> 1], R2 = 12, ra = Math.min(8, len >> 2);
          let off = notes[p0] + dn - center;
          // 自信がないときは無理に直さない(元の歌のまま): ①となりの音のちょうど間くらい(どちらに直すべきか判断できない) ②音程がうまく測れていない ③とても短い音
          if (!hard && !e) {
            let conf = 1;
            if (!ref) { const dev = Math.abs(center - Math.round(center)), amb = Math.abs(off) > 0.3 ? 1 - Math.max(0, Math.min(1, (0.7 - Math.abs(off)) / 0.25)) * Math.max(0, Math.min(1, (dev - 0.3) / 0.15)) : 1; conf *= amb + (1 - amb) * (opts.ambW || 0); } // ambW: 2つの音のちょうど間の音も、音階・伴奏で選んだ音へ直す割合
            let sd = 0; for (let i = a0; i < a1; i++) sd += (midi[base + i] - center) ** 2; sd = Math.sqrt(sd / Math.max(1, a1 - a0)); if (sd > 0.8) conf *= Math.max(0, 1 - (sd - 0.8) / 0.6);
            if (len < 6) conf *= 0.3 + 0.7 * (opts.shortW || 0); // shortW: 短い音も直す割合
            if (opts.conf || opts.stab) { // v4.0 補正適性: 音程の推定に自信があり、声が安定しているノートだけ動かす(どれか1つで切らず掛け合わせる)
              let mc = 0, ms = 0, cnt = 0; for (let i = a0; i < a1; i++) if (midi[base + i] === midi[base + i]) { mc += opts.conf ? opts.conf[base + i] : 1; ms += opts.stab ? opts.stab[base + i] : 1; cnt++; }
              if (cnt) { mc /= cnt; ms /= cnt; const fc = Math.max(0, Math.min(1, (mc - 0.2) / 0.3)), fs2 = Math.max(0, Math.min(1, (ms - 0.6) / 0.3)), su = fc * fs2;
                conf *= su; if (su < 0.5 && Math.abs(off) > 0.05) skipWhy[fc <= fs2 ? 'unsure' : 'unstable']++; }
            }
            if (opts.safeNat && !ref && Math.abs(off) > 0.5) { // v4.0 ナチュラル・お手本なし: キーと和音だけを根拠に大きくは動かさない(0.5半音から弱め、1.0半音で0)
              const w = Math.max(0, Math.min(1, (1.0 - Math.abs(off)) / 0.5)); if (w < 0.5) skipWhy.bigNoRef++; off *= w; }
            const dz = natDz; if (Math.abs(off) < dz) off = 0; else if (Math.abs(off) < 2 * dz) off *= (Math.abs(off) - dz) / dz; // ほとんど合っている音はさわらない
            if (conf < 0.999) { off *= conf; if (conf < 0.5) lowConf.push(base + p0); }
          }
          { // 長い音でゆっくり上下にふらつく所は、ふらつきだけ少し戻す(頭のしゃくり・語尾は触らない)
            // v3.0.1: 戻すのは「ゆっくりのふらつき(0.3半音くらいまで)」だけ。0.6半音以上の動き(しゃくり・フォール・音のつなぎ=歌い方)の前後0.07秒は触らない
            //   (実際の歌声で、なめらかに下がる/上がる所を最大2.2半音も平らにしていた = ケロケロ・ロボット声の原因)
            const dvA = new Float32Array(len), wA = new Float32Array(len), wB = new Float32Array(len);
            for (let i = p0; i < q0; i++) { const u = Math.max(p0, i - R2), w = Math.min(q0 - 1, i + R2), ls = (pre[w + 1] - pre[u]) / (w - u + 1); dvA[i - p0] = center - ls; const dd = Math.abs(midi[base + i] - center); wA[i - p0] = Math.max(0, Math.min(1, (0.6 - Math.max(Math.abs(dvA[i - p0]), 0.5 * dd)) / 0.3)); }
            for (let i = 0; i < len; i++) { let m = 1; for (let j = Math.max(0, i - 6); j <= Math.min(len - 1, i + 6); j++) m = Math.min(m, wA[j]); wB[i] = m; }
            for (let i = p0; i < q0; i++) { let a2 = 0, c2 = 0; for (let j = Math.max(p0, i - 4); j <= Math.min(q0 - 1, i + 4); j++) { a2 += wB[j - p0]; c2++; }
              const ramp = Math.max(0, Math.min(1, (i - p0 - ra) / 6, (q0 - 1 - i - ra) / 6));
              cn[i] = hard ? c[i] : Math.max(-1.2, Math.min(1.2, off + natFl * ramp * (a2 / c2) * dvA[i - p0])); }
          }
          if (!hard && autoN > 0 && !e) { // おまかせのノート調整(手で直したノートは触らない)
            // (a) 大きすぎるしゃくり・頭のふらつき: 深さ0.9半音までは味として残し、それ以上は小さく
            const HZ = Math.min(14, len >> 1); let tamed = 0;
            const legato = q > 0 && segs[q - 1][1] === p0, dirPrev = legato ? Math.sign(notes[segs[q - 1][0]] - notes[p0]) : 0; // 前の音から続けて移ってくる所は「つなぎ」なので触らない
            if (false) for (let i = p0; i < p0 + HZ; i++) { const d = midi[base + i] - center, ad = Math.abs(d); if (ad > 0.9 && !(legato && Math.sign(d) === dirPrev)) { /* v3.0: 深いしゃくりは歌い方(表現)なので自動では変えない */ const d2 = Math.sign(d) * (0.9 + (ad - 0.9) * 0.35); cn[i] += autoN * (d2 - d) * Math.min(1, (p0 + HZ - i) / 4); tamed = Math.max(tamed, ad); } }
            if (tamed) autoFix.push({ k0: base + p0, k1: base + p0 + HZ, type: 'scoop', amt: Math.round(tamed * 10) / 10 });
            // (b) 声の裏返り・一瞬の音程の飛び(6フレーム以内でまわりから1半音以上はなれる所)をならす
            for (let i = p0 + HZ; i < q0 - 2; i++) { const md = medAround(midi, base + i, 6, base + p0, base + q0); if (Math.abs(midi[base + i] - md) <= 1) continue; let j = i; while (j < q0 && Math.abs(midi[base + j] - medAround(midi, base + j, 6, base + p0, base + q0)) > 1) j++; if (j - i <= 6 && (!opts.ap || (() => { let m2 = 0; for (let q3 = i; q3 < j; q3++) m2 = Math.max(m2, opts.ap[base + q3] || 0); return m2 > 0.18; })())) { for (let q2 = Math.max(p0, i - 2); q2 < Math.min(q0, j + 2); q2++) { const w = q2 < i || q2 >= j ? 0.5 : 1; cn[q2] += autoN * w * (medAround(midi, base + q2, 6, base + p0, base + q0) - midi[base + q2]); } autoFix.push({ k0: base + i, k1: base + j, type: 'crack' }); } i = j; }
            // (c) 長い音で音程がゆっくり下がる/上がる(ふらつき)は上で戻している → 大きかったら印だけ
            if (len > 40) { let mx = 0; for (let i = p0 + ra; i < q0 - ra; i++) { const u = Math.max(p0, i - R2), w = Math.min(q0 - 1, i + R2); mx = Math.max(mx, Math.abs((pre[w + 1] - pre[u]) / (w - u + 1) - center)); } if (mx > 0.3) autoFix.push({ k0: base + p0, k1: base + q0, type: 'drift', amt: Math.round(mx * 10) / 10 }); }
          }
        }
        if (e && e.drf) for (let i = p0; i < q0; i++) { drfA[base + i] = e.drf; meanA[base + i] = cen / len; } // ノートごとの「ふらつきを抑える」
        notesOut.push({ k0: base + p0, k1: base + q0, note: notes[p0], center: cen / len, dn, amt: e && e.amt != null ? e.amt : null, mod: e && e.mod != null ? e.mod : null });
      }
      if (!hard) for (let q = 1; q < segs.length; q++) { // 音のつなぎ目: ふつうは歌い手自身の音の移り方(速さ・形)を使う。ノート編集で「つなぎ」を決めた所はその速さで
        const p0 = segs[q][0]; if (segs[q - 1][1] !== p0) continue;
        const ep = findEdit(base + segs[q - 1][0], base + p0), en = findEdit(base + p0, base + segs[q][1]);
        const a = notes[segs[q - 1][0]] + (ep && ep.dn ? ep.dn : 0), b = notes[p0] + (en && en.dn ? en.dn : 0); if (Math.abs(a - b) < 0.01) continue;
        if (en && en.tr != null) { // 手で決めたつなぎ: 0=すぐ切り替え 〜 1=ゆっくりなめらか
          const T = Math.max(2, Math.round(2 + en.tr * 24)), i0 = p0 - (T >> 1);
          for (let i = Math.max(segs[q - 1][0], i0); i < Math.min(segs[q][1], i0 + T); i++) { let w = (i - i0 + 0.5) / T; w = w * w * (3 - 2 * w); c[i] = Math.max(-6, Math.min(6, a + w * (b - a) - slowA[base + i])); manual[i] = 1; }
          continue;
        }
        const sa = s[Math.max(0, p0 - 6)], sb = s[Math.min(L - 1, p0 + 8)]; if (Math.abs(sb - sa) < 0.4 || (sb - sa) * (b - a) <= 0) continue; // 歌い手が同じ向きに動いているときだけ
        const T = Math.round(4 + (1 - speed) * 10);
        for (let i = Math.max(segs[q - 1][0], p0 - 5); i < Math.min(segs[q][1], p0 + T); i++) {
          const w = Math.max(0, Math.min(1, (s[i] - sa) / (sb - sa))), tg = a + w * (b - a);
          c[i] = Math.max(-4, Math.min(4, tg - slowA[base + i]));
        }
      }
      if (!hard && nat > 0) { // 自然な補正の、となりのノートとのつなぎ: 補正量を歌い手の音の移りに合わせてなめらかに入れ替える
        for (let q = 1; q < segs.length; q++) { const p0 = segs[q][0]; if (segs[q - 1][1] !== p0) continue;
          const A = cn[p0 - 1], B = cn[p0], H = 5, i0 = Math.max(segs[q - 1][0], p0 - H), i1 = Math.min(segs[q][1] - 1, p0 + H - 1);
          for (let i = i0; i <= i1; i++) { let w = (i - i0 + 0.5) / (i1 - i0 + 1); w = w * w * (3 - 2 * w); cn[i] = A + w * (B - A); } }
        for (let i = 0; i < L; i++) if (!manual[i]) c[i] = nat * cn[i] + (1 - nat) * c[i];
      }
      for (let i = 0; i < L; i++) {
        let sm = 0, cnt = 0;
        for (let j = Math.max(0, i - 1); j <= Math.min(L - 1, i + 1); j++) { sm += c[j]; cnt++; }
        const cs = hard ? c[i] : sm / cnt, kk = base + i;
        cfull[kk] = cs;
        // 合わせる強さ × ズレ ＋ 揺れ(ビブラート)を大きく/小さく
        shift[kk] = amtA[kk] * cs + (modA[kk] - 1) * (midi[kk] - slowA[kk]) + (hard ? 0 : (1 - amtA[kk]) * dnA[kk]) + (drfA[kk] ? (1 - amtA[kk]) * drfA[kk] * (meanA[kk] - slowA[kk]) : 0);
        target[kk] = notes[i] + dnA[kk];
        sumErr += Math.abs(cs - dnA[kk]); nV++;
      }
    }
    if (!hard && nat > 0 && opts.ap) { // 息っぽい・音程がはっきりしない所は補正しない(息の音が周期的になって「ジー」「ロボット」っぽくなるのを防ぐ)
      const ap = opts.ap, w = new Float32Array(nF); for (let k = 0; k < nF; k++) w[k] = Math.max(0, Math.min(1, (0.42 - (ap[k] == null ? 0 : ap[k])) / 0.22));
      for (let k = 0; k < nF; k++) { if (!(midi[k] === midi[k])) continue; let s2 = 0, c2 = 0; for (let j = Math.max(0, k - 3); j <= Math.min(nF - 1, k + 3); j++) { s2 += w[j]; c2++; } shift[k] *= nat * (s2 / c2) + (1 - nat); }
    }
    if (!hard && opts.pen) applyPen(opts.pen, midi, shift, slowA);
    // 正直な目安: ノートののばしている所(30〜85%)の平均の高さと、合わせる音との差(補正前/後)
    let eB = 0, eA = 0, eW = 0;
    for (const n of notesOut) { const L = n.k1 - n.k0; if (L < 8) continue; const tg = n.note + (n.dn || 0); let a = 0, b = 0, c = 0; for (let k = n.k0 + Math.floor(L * 0.3); k < n.k0 + Math.floor(L * 0.85); k++) if (midi[k] === midi[k]) { a += midi[k]; b += midi[k] + shift[k]; c++; } if (c < 3) continue; eB += Math.abs(a / c - tg) * L; eA += Math.abs(b / c - tg) * L; eW += L; }
    const errBefore = eW ? eB / eW * 100 : (nV ? sumErr / nV * 100 : 0), errAfterReal = eW ? eA / eW * 100 : null;
    return { shift, target, midi, cfull, tonic, slow: slowA, notes: notesOut, autoFix, lowConf, skipWhy, bias, refMap, errBefore, errAfter: errAfterReal != null ? errAfterReal : errBefore * (1 - (hard ? 1 : strength)), voicedSec: nV * HOP_SEC };
  }

  // ペンで描いた線(d: フレーム→音の高さ)・なじませブラシ(s: フレーム→強さ) を、まわりと自然につないで補正量に反映
  function applyPen(pen, midi, shift, slow) {
    const nF = midi.length, out = new Float32Array(nF); for (let k = 0; k < nF; k++) out[k] = midi[k] + shift[k];
    const V = k => k >= 0 && k < nF && midi[k] === midi[k];
    // 1) なじませブラシ: そこの音程の線を、前後をならしたなめらかな線に(段差・ガタガタ・急なしゃくりを丸める)
    const sw = new Float32Array(nF); if (pen.s) for (const [k, v] of pen.s) if (k >= 0 && k < nF) sw[k] = Math.max(sw[k], v);
    if (pen.s && pen.s.length) {
      const w2 = new Float32Array(nF); for (let k = 0; k < nF; k++) if (sw[k] > 0) for (let j = -4; j <= 4; j++) if (k + j >= 0 && k + j < nF) w2[k + j] = Math.max(w2[k + j], sw[k] * (1 - Math.abs(j) / 5)); // ふちはだんだん
      const o2 = new Float32Array(out), R = 7;
      for (let k = 0; k < nF; k++) { if (!(w2[k] > 0) || !V(k)) continue; let a = 0, c = 0; for (let j = k - R; j <= k + R; j++) if (V(j)) { let ok = true; for (let q = Math.min(j, k); q <= Math.max(j, k); q++) if (!V(q)) { ok = false; break; } if (!ok) continue; const w = 1 - Math.abs(j - k) / (R + 1); a += out[j] * w; c += w; } if (c) o2[k] = out[k] + w2[k] * (a / c - out[k]); }
      out.set(o2);
    }
    // 2) ペン: 描いた所はその高さに。すき間(3フレームまで)は線でつなぎ、端は前後6フレームかけてもとの線へ戻す
    if (pen.d && pen.d.length) {
      const d = new Float32Array(nF).fill(NaN); for (const [k, v] of pen.d) if (k >= 0 && k < nF) d[k] = v;
      for (let k = 0; k < nF; k++) if (d[k] === d[k]) { let j = k + 1; while (j < nF && !(d[j] === d[j]) && j - k <= 4) j++; if (j < nF && j - k > 1 && j - k <= 4 && d[j] === d[j]) for (let q = k + 1; q < j; q++) d[q] = d[k] + (d[j] - d[k]) * (q - k) / (j - k); }
      const sm = new Float32Array(d); for (let k = 1; k < nF - 1; k++) if (d[k] === d[k] && d[k - 1] === d[k - 1] && d[k + 1] === d[k + 1]) sm[k] = (d[k - 1] + 2 * d[k] + d[k + 1]) / 4;
      const keep = pen.vib == null ? 1 : pen.vib, E = 8;
      const vibAt = k => keep && slow[k] === slow[k] && slow[k] !== 0 ? keep * (midi[k] - slow[k]) : 0;
      const tg = new Float32Array(nF).fill(NaN), w = new Float32Array(nF);
      for (let k = 0; k < nF; k++) if (sm[k] === sm[k]) { tg[k] = sm[k]; w[k] = 1; }
      for (let k = 0; k < nF; k++) if (sm[k] === sm[k]) { // 描いた線の端の外側: 端の高さを少し延ばして、だんだん元の線に
        if (!(sm[k - 1] === sm[k - 1])) for (let j = 1; j <= E && k - j >= 0 && !(sm[k - j] === sm[k - j]); j++) { const ww = 1 - j / (E + 1); if (ww > w[k - j]) { w[k - j] = ww; tg[k - j] = sm[k]; } }
        if (!(sm[k + 1] === sm[k + 1])) for (let j = 1; j <= E && k + j < nF && !(sm[k + j] === sm[k + j]); j++) { const ww = 1 - j / (E + 1); if (ww > w[k + j]) { w[k + j] = ww; tg[k + j] = sm[k]; } }
      }
      for (let k = 0; k < nF; k++) if (w[k] > 0 && V(k)) { const ww = w[k] * w[k] * (3 - 2 * w[k]); out[k] = out[k] + ww * (tg[k] + vibAt(k) - out[k]); }
    }
    for (let k = 0; k < nF; k++) if (midi[k] === midi[k]) shift[k] = Math.max(-12, Math.min(12, out[k] - midi[k]));
  }

  // ---------- ハモリ / ボイスチェンジ ----------
  const PCS = [0, 2, 4, 5, 7, 9, 11];
  // 音階の中で deg 段だけ上下した音(例: +2 = 3度上, -2 = 3度下, +7 = オクターブ上)
  function harmNote(n, tn, deg) {
    const rel = (((n - tn) % 12) + 12) % 12, oct = Math.floor((n - tn) / 12);
    let d = 0; for (let i = 0; i < 7; i++) if (PCS[i] <= rel) d = i;
    const D = oct * 7 + d + deg, o2 = Math.floor(D / 7);
    return tn + 12 * o2 + PCS[D - 7 * o2];
  }
  // ハモリが複数あるときは、全部のハモリの音を一緒に決める(ハモリ同士・メイン・伴奏が半音/全音でぶつからないように)
  function harmonyPlan(plan, deg, key, offsetSec, set, ai) {
    const nF = plan.target.length, shift = new Float32Array(nF), target = new Float32Array(nF);
    if (ai && ai.role) return aiHarmonyPlan(plan, key, offsetSec, ai);
    const degs = Array.from(new Set((set && set.length ? set : [deg]).concat([deg]))).sort((x, y) => x - y);
    const HC = key && key.hc && key.hc.length ? key.hc : null, hcN = HC ? HC.length / 12 : 0, pick = new Map();
    const ivBad = iv => { iv = ((iv % 12) + 12) % 12; return iv === 1 || iv === 11 ? 2 : iv === 2 || iv === 10 ? 0.7 : iv === 6 ? 0.8 : 0; };
    if (plan.notes && (HC || degs.length > 1)) for (const sg of plan.notes) {
      const k0 = sg.k0 + Math.floor((sg.k1 - sg.k0) * 0.2), k1 = Math.max(k0 + 1, sg.k1 - 1), n = plan.target[k0];
      if (!(n === n)) continue;
      let h = null;
      if (HC) {
        const hh = new Float32Array(12); let any = 0;
        const f0i = Math.max(0, Math.round((k0 * HOP_SEC + (offsetSec || 0) - key.hcT0) / key.hcDt)), f1i = Math.min(hcN - 1, Math.round((k1 * HOP_SEC + (offsetSec || 0) - key.hcT0) / key.hcDt));
        for (let f = f0i; f <= f1i; f++) for (let q = 0; q < 12; q++) { hh[q] += HC[f * 12 + q]; any += HC[f * 12 + q]; }
        if (any > 0) { let mx = 0; for (let q = 0; q < 12; q++) mx = Math.max(mx, hh[q]); for (let q = 0; q < 12; q++) hh[q] /= mx; h = hh; }
      }
      const tn = plan.tonic[k0];
      // 各ハモリの候補: そのままの度数 / 1つ外側(3度→4度など)。オクターブはそのまま
      const cands = degs.map(d => Math.abs(d) >= 7 || d === 0 ? [d] : [d, d + Math.sign(d)]);
      const single = d => { const x = harmNote(n, tn, d), pc = ((x % 12) + 12) % 12; let c = ivBad(x - n) * 1.0 + (pc === ((n % 12) + 12) % 12 && Math.abs(d) < 7 ? 0.5 : 0);
        if (h) { const nb = Math.max(h[(pc + 1) % 12], h[(pc + 11) % 12]); c += -h[pc] + 0.9 * Math.max(0, nb - h[pc]); }
        return c; };
      let best = null, bc = 1e9;
      const cur = [];
      const rec = i => {
        if (i === degs.length) {
          let c = 0; const xs = cur.map(d => harmNote(n, tn, d));
          for (let a2 = 0; a2 < cur.length; a2++) { c += single(cur[a2]) + (cur[a2] !== degs[a2] ? 0.35 : 0);
            for (let b2 = a2 + 1; b2 < cur.length; b2++) { const iv = xs[b2] - xs[a2]; c += ivBad(iv) * 0.8 + (iv === 0 ? 1.2 : 0); } } // ハモリ同士: 半音・全音のぶつかり、同じ音の重なりを避ける
          if (c < bc) { bc = c; best = cur.slice(); }
          return;
        }
        for (const d of cands[i]) { cur.push(d); rec(i + 1); cur.pop(); }
      };
      rec(0);
      const mine = best[degs.indexOf(deg)];
      if (mine !== deg) pick.set(sg.k0, mine);
    }
    const notes = plan.notes || [];
    let ni = 0;
    for (let k = 0; k < nF; k++) {
      while (ni < notes.length && notes[ni].k1 <= k) ni++;
      const curDeg = ni < notes.length && notes[ni].k0 <= k && pick.has(notes[ni].k0) ? pick.get(notes[ni].k0) : deg;
      const n = plan.target[k];
      if (n !== n) { target[k] = NaN; continue; }
      const x = harmNote(n, plan.tonic[k], curDeg);
      target[k] = x; shift[k] = x - n + plan.cfull[k];
    }
    // v3.6: 機械っぽさ対策
    // ① 音が変わる所で「何度上か(3度→4度など)」が一瞬で切りかわり、ハモリだけ音程が段になって跳んでいた(ケロケロ) → 約90msかけてなめらかに
    // ② メインと完全に同じ動き(平行移動)だと機械的 → ハモリだけのゆっくりした小さなゆらぎ(±5セントくらい)を足す
    const iv = new Float32Array(nF); for (let k = 0; k < nF; k++) iv[k] = target[k] === target[k] ? target[k] - plan.target[k] : NaN;
    const W2 = 4, ivS = new Float32Array(nF);
    for (let k = 0; k < nF; k++) { if (!(iv[k] === iv[k])) { ivS[k] = NaN; continue; } let a = 0, c = 0; for (let j = k - W2; j <= k + W2; j++) if (j >= 0 && j < nF && iv[j] === iv[j]) { a += iv[j]; c++; } ivS[k] = a / c; }
    let seed = 1234567 + 7919 * (deg + 20), r = 0, r2 = 0; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
    for (let k = 0; k < nF; k++) { r += (rnd() * 0.06 - r * 0.02); r2 += (r - r2) * 0.03; // ゆっくりした小さなゆらぎ
      if (ivS[k] === ivS[k]) shift[k] += ivS[k] - iv[k] + Math.max(-0.07, Math.min(0.07, r2)); }
    return { shift, target };
  }

  // ---------- v4.4 AIハモリアレンジャー ----------
  // メロディの音と「その瞬間にカラオケが鳴らしている和音(音名の強さ)」から、各ハモリ声部の音を1音ずつ決める。
  // 1音ずつ別々に決めず、前後のつながり(跳躍・平行5度/8度・音程差の急な変化)も含めて一番よい並びを探す(ビタビ探索)。
  // 和音がはっきりしない所(音名の強さが平ら)は、和音を無理に当てはめず、曲の音階だけで決める。
  const AI_ROLES = ['up', 'down', 'up6', 'down6', 'high', 'low', 'oct', 'octDown'];
  const AI_IV = { // 役割ごとの「メロディとの音程差(半音)」の好み(値が小さいほど好ましい。無い音程は使わない)
    up: { 3: 0, 4: 0, 5: 0.5, 7: 0.9, 8: 1.1, 9: 1.0, 2: 1.7, 6: 2.2, 1: 3.2 },
    down: { '-3': 0, '-4': 0, '-5': 0.4, '-7': 0.8, '-8': 0.6, '-9': 0.6, '-2': 1.7, '-6': 2.2, '-1': 3.2 },
    up6: { 8: 0, 9: 0, 7: 0.6, 5: 0.9, 12: 0.9, 10: 1.3, 3: 1.1, 4: 1.1, 6: 2.2, 11: 3.2 },
    down6: { '-8': 0, '-9': 0, '-7': 0.6, '-5': 0.9, '-12': 0.9, '-10': 1.3, '-3': 1.1, '-4': 1.1, '-6': 2.2, '-11': 3.2 },
    high: { 8: 0.2, 9: 0.2, 7: 0.7, 12: 0.9, 5: 0.8, 10: 1.4 },
    low: { '-9': 0.2, '-8': 0.2, '-7': 0.6, '-12': 0.9, '-5': 0.5, '-10': 1.4 },
    oct: { 12: 0 }, octDown: { '-12': 0 }
  };
  function noteChroma(key, k0, k1, offsetSec) {
    const HC = key && key.hc && key.hc.length ? key.hc : null; if (!HC) return null;
    const hcN = HC.length / 12, hh = new Float32Array(12); let any = 0;
    const f0i = Math.max(0, Math.round((k0 * HOP_SEC + (offsetSec || 0) - key.hcT0) / key.hcDt)), f1i = Math.min(hcN - 1, Math.round((k1 * HOP_SEC + (offsetSec || 0) - key.hcT0) / key.hcDt));
    for (let f = f0i; f <= f1i; f++) for (let q = 0; q < 12; q++) { hh[q] += HC[f * 12 + q]; any += HC[f * 12 + q]; }
    if (!(any > 0)) return null;
    let mx = 0, mean = 0; for (let q = 0; q < 12; q++) mx = Math.max(mx, hh[q]); for (let q = 0; q < 12; q++) { hh[q] /= mx; mean += hh[q] / 12; }
    // 和音の確かさ: 強い音名が3〜4個に集まっているほど高い(平らだと0に近い)
    const so = Array.from(hh).sort((a, b) => b - a), top = (so[0] + so[1] + so[2]) / 3, rest = so.slice(4).reduce((a, b) => a + b, 0) / 8;
    return { h: hh, conf: Math.max(0, Math.min(1, (top - rest - 0.15) / 0.45)) };
  }
  function arrangeVoices(plan, key, offsetSec, roles, ov, range) {
    const notes = (plan.notes || []).filter(sg => sg.k1 - sg.k0 >= 3), N = notes.length, out = {}, stat = { notes: N, chordConf: 0, lowConf: 0, clash: {}, maxShift: {}, leaps: {} };
    const info = notes.map(sg => { const k0 = sg.k0 + Math.floor((sg.k1 - sg.k0) * 0.2), n = Math.round(plan.target[k0]); const c = n === n ? noteChroma(key, k0, Math.max(k0 + 1, sg.k1 - 1), offsetSec) : null; return { k0: sg.k0, k1: sg.k1, n, tn: plan.tonic[k0], c, short: (sg.k1 - sg.k0) * HOP_SEC < 0.16 }; });
    let cs = 0, cc = 0; for (const it of info) if (it.c) { cs += it.c.conf; cc++; if (it.c.conf < 0.35) stat.lowConf++; } stat.chordConf = cc ? cs / cc : 0;
    const inScale = (x, tn) => PCS.includes((((x - tn) % 12) + 12) % 12);
    const order = AI_ROLES.filter(r => roles.includes(r)), chosen = {};
    for (const role of order) {
      const pref = AI_IV[role], ivs = Object.keys(pref).map(Number), C = [], cost = [];
      const userOv = ov && ov[role] || {};
      for (let i = 0; i < N; i++) {
        const it = info[i], cand = [], cv = [];
        if (!(it.n === it.n)) { C.push([NaN]); cost.push([0]); continue; }
        const fixed = userOv[it.k0];
        for (const iv of (fixed != null ? [fixed - it.n] : ivs)) {
          const x = it.n + iv, pc = ((x % 12) + 12) % 12; let c = fixed != null ? 0 : pref[iv];
          if (fixed == null) {
            const chordW = it.short ? 0.5 : 1; // 短い経過音は和音から外れてもよい(半音・全音を無条件に禁止しない)
            if (it.c && it.c.conf >= 0.35) { const h = it.c.h, nb = Math.max(h[(pc + 1) % 12], h[(pc + 11) % 12]); c += chordW * (-1.8 * h[pc] + 1.0 * Math.max(0, nb - h[pc])) + (inScale(x, it.tn) || h[pc] > 0.7 ? 0 : 0.8); }
            else c += inScale(x, it.tn) ? 0 : 2.5;
            if (role === 'low' || role === 'high') c += it.c && it.c.conf >= 0.35 ? -0.8 * it.c.h[pc] : 0; // 低音/高音コーラスは和音の音をより強く選ぶ
            const im = ((iv % 12) + 12) % 12; if (!it.short && (im === 1 || im === 2 || im === 10 || im === 11)) c += 1.5; // メロディと半音・全音でぶつかる(長い音だけ)
            if (im === 0 && role !== 'oct' && role !== 'octDown') c += 1.2; // メロディと同じ音名(オクターブ違い)はハモリにならない
            if (range) { const lo = range[0] - 2, hi = range[1] + 2; if (x < lo) c += 0.6 * (lo - x); if (x > hi) c += 0.6 * (x - hi); } // v4.5 歌い手の音域から外れる音(低すぎる下ハモ・高すぎる上ハモ)をさける
            c += (role === 'up' || role === 'down' ? 0.12 * Math.max(0, Math.abs(iv) - 5) : 0.05 * Math.max(0, Math.abs(iv) - 9)); // 大きく音程を変えるほど声が不自然になりやすい
            for (const r2 in chosen) { const y = chosen[r2][i]; if (!(y === y)) continue; const d = ((x - y) % 12 + 12) % 12;
              c += x === y ? 1.6 : (d === 1 || d === 11) ? 1.2 * chordW : (d === 2 || d === 10) ? 0.5 * chordW : 0;
              const a1 = AI_ROLES.indexOf(r2), above = ['up', 'up6', 'high', 'oct'].includes(role), above2 = ['up', 'up6', 'high', 'oct'].includes(r2);
              if (above && !above2 && x <= y) c += 1.5; if (!above && above2 && x >= y) c += 1.5; if (above === above2 && a1 >= 0 && Math.abs(x - y) < 3) c += 0.6; }
          }
          cand.push(x); cv.push(c);
        }
        C.push(cand); cost.push(cv);
      }
      // ビタビ: つながっている音(切れ目0.3秒未満)だけ、つながりのコストを足す
      const back = [], acc = [];
      for (let i = 0; i < N; i++) {
        const a = new Float64Array(C[i].length), b = new Int16Array(C[i].length).fill(-1);
        const link = i > 0 && (info[i].k0 - info[i - 1].k1) * HOP_SEC < 0.3 && C[i - 1][0] === C[i - 1][0] && C[i][0] === C[i][0];
        for (let j = 0; j < C[i].length; j++) {
          if (i === 0 || !link) { let best = 0; if (i > 0) { best = 1e18; for (let q = 0; q < acc[i - 1].length; q++) best = Math.min(best, acc[i - 1][q]); } a[j] = best + cost[i][j]; continue; }
          let bv = 1e18, bq = 0; const x = C[i][j], iv = x - info[i].n;
          for (let q = 0; q < C[i - 1].length; q++) { const xp = C[i - 1][q], ivp = xp - info[i - 1].n, L = Math.abs(x - xp);
            const Lm = Math.abs(info[i].n - info[i - 1].n); let t = 0.15 * Math.max(0, L - Math.max(2, Lm)) + (L > 7 && L > Lm + 3 ? 1.5 : 0) + 0.08 * Math.abs(iv - ivp); /* メロディが跳ぶ所は一緒に動いてよい */
            const pi = ((ivp % 12) + 12) % 12, ci = ((iv % 12) + 12) % 12; if (x !== xp && info[i].n !== info[i - 1].n && pi === ci && (ci === 0 || ci === 7) && role !== 'oct' && role !== 'octDown') t += 1.2; // 平行5度・8度
            const v = acc[i - 1][q] + t; if (v < bv) { bv = v; bq = q; } }
          a[j] = bv + cost[i][j]; b[j] = bq;
        }
        acc.push(a); back.push(b);
      }
      const res = new Float32Array(N).fill(NaN);
      if (N) { let j = 0; for (let q = 1; q < acc[N - 1].length; q++) if (acc[N - 1][q] < acc[N - 1][j]) j = q;
        for (let i = N - 1; i >= 0; i--) { res[i] = C[i][j]; const bj = back[i][j]; if (bj >= 0) j = bj; else if (i > 0) { j = 0; for (let q = 1; q < acc[i - 1].length; q++) if (acc[i - 1][q] < acc[i - 1][j]) j = q; } } }
      chosen[role] = res;
      let clash = 0, mxs = 0, leaps = 0;
      for (let i = 0; i < N; i++) { const x = res[i]; if (!(x === x)) continue; const d = ((x - info[i].n) % 12 + 12) % 12; if ((d === 1 || d === 11 || d === 2 || d === 10) && !info[i].short) clash++; mxs = Math.max(mxs, Math.abs(x - info[i].n)); if (i > 0 && res[i - 1] === res[i - 1] && (info[i].k0 - info[i - 1].k1) * HOP_SEC < 0.3 && Math.abs(x - res[i - 1]) > Math.max(7, Math.abs(info[i].n - info[i - 1].n) + 4)) leaps++; /* メロディより大きく跳ぶ所 */ }
      stat.clash[role] = clash; stat.maxShift[role] = mxs; stat.leaps[role] = leaps;
      out[role] = info.map((it, i) => [it.k0, it.k1, it.n, res[i]]);
    }
    return { voices: out, stat };
  }

  // v4.4: AIアレンジャーが決めた音でハモリの音程を作る。音の中の動き(しゃくり・ビブラート)はメインの歌に合わせる。
  //   vib < 1 のとき: 長い音(0.4秒以上)のビブラートの揺れをその割合に小さくする(ハモリはメインより控えめな揺れにすると重なりがきれい)
  function aiHarmonyPlan(plan, key, offsetSec, ai) {
    const nF = plan.target.length, shift = new Float32Array(nF), target = new Float32Array(nF).fill(NaN);
    const roles = Array.from(new Set((ai.roles || []).concat([ai.role]))), A = arrangeVoices(plan, key, offsetSec, roles, ai.ov, ai.range || null), V = A.voices[ai.role] || [];
    const iv = new Float32Array(nF).fill(NaN);
    for (const [k0, k1, n, x] of V) if (x === x && n === n) for (let k = k0; k < k1 && k < nF; k++) iv[k] = x - n;
    let last = NaN; for (let k = 0; k < nF; k++) { if (iv[k] === iv[k]) last = iv[k]; else if (plan.target[k] === plan.target[k] && last === last) iv[k] = last; } // 音と音の間(すべり)は直前の音程差のまま
    last = NaN; for (let k = nF - 1; k >= 0; k--) { if (iv[k] === iv[k]) last = iv[k]; else if (plan.target[k] === plan.target[k] && last === last) iv[k] = last; }
    const W2 = 4, ivS = new Float32Array(nF).fill(NaN);
    for (let k = 0; k < nF; k++) { if (!(iv[k] === iv[k])) continue; let a = 0, c = 0; for (let j = k - W2; j <= k + W2; j++) if (j >= 0 && j < nF && iv[j] === iv[j]) { a += iv[j]; c++; } ivS[k] = a / c; } // 約90msかけて音程差を切りかえる(段差のケロケロ防止)
    const vib = ai.vib == null ? 1 : Math.max(0, Math.min(1, ai.vib)), longN = new Uint8Array(nF);
    if (vib < 1) for (const sg of plan.notes || []) if ((sg.k1 - sg.k0) * HOP_SEC >= 0.4) for (let k = sg.k0; k < sg.k1 && k < nF; k++) longN[k] = 1;
    for (let k = 0; k < nF; k++) {
      const n = plan.target[k]; if (!(n === n) || !(ivS[k] === ivS[k])) continue;
      target[k] = n + ivS[k]; shift[k] = ivS[k] + plan.cfull[k];
      if (longN[k] && plan.slow && plan.midi[k] === plan.midi[k] && plan.slow[k] === plan.slow[k]) shift[k] -= (1 - vib) * (plan.midi[k] - plan.slow[k]);
      shift[k] = Math.max(-13, Math.min(13, shift[k]));
    }
    return { shift, target, ai: A.stat };
  }

  // ---------- v4.5 ボーカル解析(ルールベースの音響計測。機械学習モデルではない) ----------
  // 既にある音程(f0)と声の波形から、音域・声質・歌い方・区間ごとの変化を測る。性別は推定しない。
  function vocalProfile(x, f0, amp) {
    const N = 2048, fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), win = new Float64Array(N), nF = f0.length;
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
    const bin = SR / N, band = (mg, a, b) => { let e = 0; for (let i = Math.max(1, Math.floor(a / bin)); i <= Math.min(N / 2 - 1, Math.ceil(b / bin)); i++) e += mg[i]; return e; };
    const fr = [], mg = new Float64Array(N / 2); let peakDb = -120;
    for (let k = 0; k < nF; k += 2) {
      const c = k * HOP, s0 = c - N / 2; if (s0 < 0 || s0 + N > x.length) continue;
      let e = 0; for (let i = 0; i < N; i++) { const v = x[s0 + i]; e += v * v; } const db = 10 * Math.log10(e / N + 1e-12); if (db < -55) continue;
      for (let i = 0; i < N; i++) { re[i] = x[s0 + i] * win[i]; im[i] = 0; }
      fft.forward(re, im); let tot = 0, cen = 0; for (let i = 1; i < N / 2; i++) { mg[i] = re[i] * re[i] + im[i] * im[i]; tot += mg[i]; cen += mg[i] * i * bin; }
      const F = f0[k] > 0 ? f0[k] : 0, o = { k, db, cen: cen / (tot + 1e-20), v: F > 0 };
      o.sib = band(mg, 5000, 10000) / (tot + 1e-20); o.body = band(mg, 120, 350) / (tot + 1e-20); o.pres = band(mg, 2000, 5000) / (tot + 1e-20); o.mud = band(mg, 250, 500) / (tot + 1e-20);
      if (F > 0) {
        o.m = 69 + 12 * Math.log2(F / 440);
        const h = n => { const c2 = n * F / bin; let mx = 0; for (let i = Math.max(1, Math.floor(c2 - 2)); i <= Math.min(N / 2 - 1, Math.ceil(c2 + 2)); i++) mx = Math.max(mx, mg[i]); return 10 * Math.log10(mx + 1e-20); };
        o.h12 = h(1) - h(2); // 1倍音と2倍音の差(大きいほど柔らかい・裏声っぽい傾向)
        const T = Math.round(SR / F), L = 1024, a0 = c - L / 2; if (a0 >= 0 && a0 + L + T < x.length) { let xy = 0, xx = 0, yy = 0; for (let i = 0; i < L; i++) { const p = x[a0 + i], q = x[a0 + i + T]; xy += p * q; xx += p * p; yy += q * q; } const r = Math.max(1e-4, Math.min(0.9999, xy / Math.sqrt(xx * yy + 1e-20))); o.hnr = 10 * Math.log10(r / (1 - r)); }
      }
      peakDb = Math.max(peakDb, db); fr.push(o);
    }
    const V = fr.filter(o => o.v && o.m === o.m), q = (arr, p) => { if (!arr.length) return null; const a = arr.slice().sort((u, w) => u - w); return a[Math.min(a.length - 1, Math.floor(p * a.length))]; };
    const ms = V.map(o => o.m), med = q(ms, 0.5), p5 = q(ms, 0.05), p95 = q(ms, 0.95);
    const hnrs = V.filter(o => o.hnr != null).map(o => o.hnr), h12s = V.map(o => o.h12), h12m = q(h12s, 0.5);
    // 裏声の可能性: 本人の音域の上側(上40%)で、1倍音と2倍音の差が本人の中央値より 6dB 以上大きい
    const hiTh = p5 != null ? p5 + (p95 - p5) * 0.6 : 0; let fal = 0; for (const o of V) { o.fal = o.m >= hiTh && o.h12 > h12m + 6; if (o.fal) fal++; }
    // ささやき: 声の大きさはあるのに音程が取れない所(歌っている区間の中)
    const U = fr.filter(o => !o.v), wsp = U.filter(o => o.db > peakDb - 30 && o.sib < 0.35).length;
    // のばした音: 0.4秒以上つづく有声の連なり → ゆれ(セント)・ビブラート・しゃくり・フォール
    const runs = []; let r0 = -1; for (let k = 0; k <= nF; k++) { const on = k < nF && f0[k] > 0; if (on && r0 < 0) r0 = k; if (!on && r0 >= 0) { if ((k - r0) * HOP_SEC >= 0.4) runs.push([r0, k]); r0 = -1; } }
    let vib = 0, scoop = 0, fall = 0, longN = 0; const stab = [], oct = [];
    for (const [a, b] of runs) {
      const m = []; for (let k = a; k < b; k++) m.push(69 + 12 * Math.log2(f0[k] / 440)); const n = m.length, W2 = Math.max(3, Math.round(0.075 / HOP_SEC));
      const sm = m.map((_, i) => { let s2 = 0, c2 = 0; for (let j = i - W2; j <= i + W2; j++) if (j >= 0 && j < n) { s2 += m[j]; c2++; } return s2 / c2; });
      const det = m.map((v, i) => v - sm[i]), mid = det.slice(Math.floor(n * 0.2), Math.ceil(n * 0.8)); let zc = 0, ss = 0; for (let i = 1; i < mid.length; i++) { if ((mid[i - 1] < 0) !== (mid[i] < 0)) zc++; ss += mid[i] * mid[i]; }
      const rate = zc / 2 / (mid.length * HOP_SEC), depth = Math.sqrt(ss / Math.max(1, mid.length)) * 2.83 * 100; if (rate >= 4 && rate <= 8 && depth >= 30) vib++;
      const core = sm.slice(Math.floor(n * 0.3), Math.ceil(n * 0.7)), cm = core.reduce((u, w) => u + w, 0) / Math.max(1, core.length); stab.push(Math.sqrt(core.reduce((u, w) => u + (w - cm) ** 2, 0) / Math.max(1, core.length)) * 100);
      const k8 = Math.max(2, Math.round(0.08 / HOP_SEC)), st0 = m.slice(0, k8).reduce((u, w) => u + w, 0) / k8, en0 = m.slice(n - k8).reduce((u, w) => u + w, 0) / k8;
      if (cm - st0 > 0.5) scoop++; if (cm - en0 > 0.5) fall++; if (n * HOP_SEC >= 1.0) longN++;
      for (let i = 2; i < n - 2; i++) if (Math.abs(m[i] - m[i - 2]) > 10.5 && Math.abs(m[i] - m[i - 2]) < 13.5) oct.push(a + i);
    }
    // 区間ごと(6秒・2秒おき)の変化 → なめらかにして、同じ様子が6秒以上続く所だけ区間にする
    const durS = nF * HOP_SEC, secs = [];
    for (let t0 = 0; t0 < durS; t0 += 2) {
      const w = fr.filter(o => o.k * HOP_SEC >= t0 && o.k * HOP_SEC < t0 + 6), wv = w.filter(o => o.v && o.m === o.m); if (wv.length < 15) { secs.push(null); continue; }
      const hn = q(wv.filter(o => o.hnr != null).map(o => o.hnr), 0.5), wm = q(wv.map(o => o.m), 0.5), lv = q(wv.map(o => o.db), 0.5), fs2 = wv.filter(o => o.fal).length / wv.length, ws = w.filter(o => !o.v && o.db > peakDb - 30).length / w.length;
      secs.push({ t0, m: wm, lv, hnr: hn, fal: fs2, wsp: ws });
    }
    const lvAll = q(V.map(o => o.db), 0.5), hnAll = q(hnrs, 0.5);
    const hRef = q(secs.filter(o => o && o.hnr != null).map(o => o.hnr), 0.75) || hnAll; // 本人がはっきり歌っている所の基準
    const lab = o => !o ? null : (o.wsp > 0.25 || (o.hnr != null && (o.hnr < hRef - 8 || o.hnr < 11))) ? 'breathy' : o.fal > 0.3 ? 'falsetto' : (o.lv > lvAll + 2 && o.m > med + 1) ? 'strong' : (o.lv < lvAll - 4) ? 'soft' : 'normal';
    const raw = secs.map(lab), sm2 = raw.map((v, i) => { const c3 = [raw[i - 1], v, raw[i + 1]].filter(z => z); const cnt = {}; for (const z of c3) cnt[z] = (cnt[z] || 0) + 1; let best = v, bc = 0; for (const z in cnt) if (cnt[z] > bc) { bc = cnt[z]; best = z; } return best; });
    const segs = []; sm2.forEach((v, i) => { if (!v) return; const t0 = i * 2, l = segs[segs.length - 1]; if (l && l.lab === v && t0 - l.t1 <= 2.01) l.t1 = t0 + 2; else segs.push({ lab: v, t0, t1: t0 + 2 }); });
    const segs2 = []; for (const z of segs.filter(z => z.t1 - z.t0 >= 6)) { const l = segs2[segs2.length - 1]; if (l && l.lab === z.lab) l.t1 = z.t1; else segs2.push(Object.assign({}, z)); }
    const sections = segs2.map(z => { const ws = secs.filter(o => o && o.t0 >= z.t0 && o.t0 < z.t1); return { lab: z.lab, t0: z.t0, t1: Math.min(durS, z.t1 + 4), m: q(ws.map(o => o.m), 0.5) }; });
    const dbs = V.map(o => o.db);
    return {
      ver: 1, frames: fr.length, voiced: V.length,
      range: p5 == null ? null : { p5, p25: q(ms, 0.25), med, p75: q(ms, 0.75), p95, lo: q(ms, 0.01), hi: q(ms, 0.99) },
      f0Med: med == null ? null : 440 * Math.pow(2, (med - 69) / 12), f0P5: p5 == null ? null : 440 * Math.pow(2, (p5 - 69) / 12),
      cen: q(V.map(o => o.cen), 0.5), hnr: hnAll, h12: h12m, sib: q(U.filter(o => o.db > peakDb - 35).map(o => o.sib), 0.5), sibV: q(V.map(o => o.sib), 0.5),
      body: q(V.map(o => o.body), 0.5), mud: q(V.map(o => o.mud), 0.5), pres: q(V.map(o => o.pres), 0.5),
      dyn: dbs.length > 20 ? q(dbs, 0.9) - q(dbs, 0.1) : null, falShare: V.length ? fal / V.length : 0, wspShare: fr.length ? wsp / fr.length : 0,
      runs: runs.length, longN, vibShare: runs.length ? vib / runs.length : 0, scoopShare: runs.length ? scoop / runs.length : 0, fallShare: runs.length ? fall / runs.length : 0,
      stab: q(stab, 0.5), octJumps: oct.length, sections, win: secs.map((o, i) => o && { t0: o.t0, m: +o.m.toFixed(1), lv: +o.lv.toFixed(1), hnr: o.hnr == null ? null : +o.hnr.toFixed(1), fal: +o.fal.toFixed(2), wsp: +o.wsp.toFixed(2), lab: sm2[i] })
    };
  }


  // v4.7 原曲とカラオケの照合(引き算)を強くする:
  //  ・曲中の時間のずれ(ドリフト): 10秒ごとにずれを測り、直線で近似して少しずつずらす(サンプル以下は直線補間)
  //  ・周波数特性・位相・音量・左右の差(マスタリングのちがい): 周波数ごとの複素ゲイン H(f) を最小二乗で求めて、原曲 − H×カラオケ(STFT)
  //  ・ねらいどおり引けたかは、歌の少ない低音(150Hz以下)がどれだけ残ったかで判定(残りが多いときは引き算しない)
  function driftLags(om, im, lag0) {
    const SEG = 10 * SR, W = 3 * SR, pts = [];
    for (let c = SEG / 2; c + W < im.length; c += SEG) { let best = -1e30, bl = 0, e0 = 0;
      for (let d = -96; d <= 96; d += 2) { let sm = 0, e = 0; for (let i = c; i < c + W; i += 3) { const j = i - (lag0 + d); if (j < 0 || j >= om.length) continue; sm += om[j] * im[i]; e += om[j] * om[j]; } const v = e > 0 ? sm / Math.sqrt(e) : -1e30; if (v > best) { best = v; bl = d; } }
      for (let d = bl - 2; d <= bl + 2; d++) { let sm = 0, e = 0; for (let i = c; i < c + W; i += 3) { const j = i - (lag0 + d); if (j < 0 || j >= om.length) continue; sm += om[j] * im[i]; e += om[j] * om[j]; } const v = e > 0 ? sm / Math.sqrt(e) : -1e30; if (v >= best) { best = v; bl = d; } }
      for (const f of [-0.5, 0.5]) void f; pts.push([c + W / 2, lag0 + bl]); void e0; }
    if (pts.length < 3) return null;
    const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length, my = pts.reduce((a, p) => a + p[1], 0) / pts.length; let sxy = 0, sxx = 0; for (const [x, y] of pts) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; }
    const a = sxy / sxx, b = my - a * mx, span = Math.abs(a) * im.length; return { a, b, span, pts };
  }
  function shiftDrift(o, N, lag0, dr) { const y = new Float32Array(N); for (let i = 0; i < N; i++) { const l = dr && dr.span > 1.5 ? dr.a * i + dr.b : lag0, x = i - l, j = Math.floor(x), f = x - j; if (j >= 0 && j + 1 < o.length) y[i] = o[j] * (1 - f) + o[j + 1] * f; else if (j >= 0 && j < o.length) y[i] = o[j]; } return y; }
  function specSub(y, inst) {
    const N = 2048, H = 512, nb = N / 2 + 1, fft = new FFT(N), w = new Float64Array(N); for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
    const nr = new Float64Array(nb), ni = new Float64Array(nb), dd = new Float64Array(nb), a = new Float64Array(N), b = new Float64Array(N), c = new Float64Array(N), d = new Float64Array(N), L = y.length;
    for (let s0 = 0; s0 + N <= L; s0 += H * 2) { for (let i = 0; i < N; i++) { a[i] = y[s0 + i] * w[i]; b[i] = 0; c[i] = inst[s0 + i] * w[i]; d[i] = 0; } fft.forward(a, b); fft.forward(c, d);
      for (let k = 0; k < nb; k++) { nr[k] += a[k] * c[k] + b[k] * d[k]; ni[k] += b[k] * c[k] - a[k] * d[k]; dd[k] += c[k] * c[k] + d[k] * d[k]; } }
    const hr = new Float64Array(nb), hi = new Float64Array(nb); for (let k = 0; k < nb; k++) { const den = dd[k] + 1e-9; hr[k] = nr[k] / den; hi[k] = ni[k] / den; }
    const out = new Float32Array(L), ws = 1.5; // hann^2 を N/4 で重ねた和
    for (let s0 = 0; s0 + N <= L; s0 += H) { for (let i = 0; i < N; i++) { a[i] = y[s0 + i] * w[i]; b[i] = 0; c[i] = inst[s0 + i] * w[i]; d[i] = 0; } fft.forward(a, b); fft.forward(c, d);
      for (let k = 0; k < nb; k++) { a[k] -= hr[k] * c[k] - hi[k] * d[k]; b[k] -= hr[k] * d[k] + hi[k] * c[k]; } for (let k = 1; k < N / 2; k++) { a[N - k] = a[k]; b[N - k] = -b[k]; }
      for (let k = 0; k < N; k++) b[k] = -b[k]; fft.forward(a, b); for (let i = 0; i < N; i++) out[s0 + i] += a[i] / N * w[i] / ws; }
    return { r: out, gainDb: 20 * Math.log10(Math.sqrt(hr.slice(20, 400).reduce((s2, v, k) => s2 + v * v + hi[k + 20] ** 2, 0) / 380) + 1e-9) };
  }
  function lowRatio(y, r) { const lp = biquad('lp', 150, 0.707), yl = Float32Array.from(y), rl = Float32Array.from(r); filt(yl, lp); filt(yl, lp); filt(rl, lp); filt(rl, lp); let ey = 0, er = 0; for (let i = 0; i < yl.length; i += 2) { ey += yl[i] * yl[i]; er += rl[i] * rl[i]; } return ey > 0 ? er / ey : 1; }
  // ---------- v4.6 原曲解析(ルールベースの信号処理。機械学習モデルは使わない) ----------
  // 原曲(歌入り)からカラオケを引いて歌だけを取り出し(同じ音源のときだけ精度が出る)、メインの音程・ハモリの音・ダブル(左右の広がり)・
  // リバーブの長さと深さ・ディレイ(語尾のやまびこ)・声の音色(帯域)・強弱を推定する。結果には信頼度を付ける。
  function origAnalyze(oL, oR, iL, iR, opt, progress) {
    const P = progress || (() => {}), n = Math.min(oL.length, oR.length), om = new Float32Array(n); for (let i = 0; i < n; i++) om[i] = 0.5 * (oL[i] + oR[i]);
    P(0.02); const C = opt.ctx; const al = alignRef(om, C.instMono, C.env, 'full', false), lag = al.lag, N = iL.length; P(0.15);
    // チャンネルごとに、ずらした原曲 − g×カラオケ (g は最小二乗)
    const sub = (o, inst) => { const y = new Float32Array(N); let xi = 0, ii = 0; for (let i = 0; i < N; i++) { const j = i - lag; const v = j >= 0 && j < n ? o[j] : 0; y[i] = v; xi += v * inst[i]; ii += inst[i] * inst[i]; } const g = ii > 0 ? xi / ii : 0; for (let i = 0; i < N; i++) y[i] -= g * inst[i]; return { y, g }; };
    let sl = sub(oL, iL), sr = sub(oR, iR), how = 'time', ratio = al.ratio, drift = null;
    // v4.7: 時間のずれ・周波数特性のちがいも合わせた引き算を試し、低音がよく消えた方を使う
    try { const rTime = (() => { const y0 = new Float32Array(N), r0 = new Float32Array(N); for (let i = 0; i < N; i++) { const j = i - lag; y0[i] = j >= 0 && j < n ? 0.5 * (oL[j] + oR[j]) : 0; r0[i] = 0.5 * (sl.y[i] + sr.y[i]); } return lowRatio(y0, r0); })();
      let rs = 1, xL = null, xR = null; if (rTime > 0.02) { drift = driftLags(om, C.instMono, lag); const yL = shiftDrift(oL, N, lag, drift), yR = shiftDrift(oR, N, lag, drift); xL = specSub(yL, iL); xR = specSub(yR, iR); rs = lowRatio(Float32Array.from(yL, (v, i) => 0.5 * (v + yR[i])), Float32Array.from(xL.r, (v, i) => 0.5 * (v + xR.r[i]))); } // 引き算がほぼ完全なら追加の処理はしない
      if (xL && rs < rTime * 0.8) { sl = { y: xL.r }; sr = { y: xR.r }; how = 'spec'; ratio = rs; } else ratio = rTime;
    } catch (e) { void e; }
    const M = new Float32Array(N), Sd = new Float32Array(N); for (let i = 0; i < N; i++) { M[i] = 0.5 * (sl.y[i] + sr.y[i]); Sd[i] = 0.5 * (sl.y[i] - sr.y[i]); }
    const subOK = ratio < 0.3, sepConf = subOK ? Math.max(0, Math.min(1, 1 - ratio / 0.3)) : 0; P(0.25);
    // メインの音程(取り出した歌の中央)
    const tp = trackPitch(M, p => P(0.25 + p * 0.35)), f0 = tp.f0, nF = f0.length, midi = new Float32Array(nF).fill(NaN);
    // ハモリ: メインの倍音をかくした残りで、倍音の和がいちばん大きい音(メインから2半音以上・オクターブ以外)
    const NF = 4096, fft = new FFT(NF), re = new Float64Array(NF), im = new Float64Array(NF), win = new Float64Array(NF), bin = SR / NF, mg = new Float64Array(NF / 2);
    for (let i = 0; i < NF; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (NF - 1));
    const step = 2, hm = new Float32Array(nF).fill(NaN), hs = new Float32Array(nF);
    for (let k = 0; k < nF; k += step) {
      if (!(f0[k] > 0)) continue; const c = k * HOP, s0 = c - NF / 2; if (s0 < 0 || s0 + NF > N) continue;
      for (let i = 0; i < NF; i++) { re[i] = M[s0 + i] * win[i]; im[i] = 0; } fft.forward(re, im);
      let gmx = 0; for (let i = 0; i < NF / 2; i++) { mg[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]); gmx = Math.max(gmx, mg[i]); }
      // メインの音: 倍音の和がいちばん大きい音(2つの声が重なると、自己相関の方法は2オクターブ下などをまちがえて拾うため)。基音が鳴っていることが条件
      const pk = f => { const cb = f / bin; let mx = 0; for (let i = Math.max(1, Math.floor(cb - 1.5)); i <= Math.min(NF / 2 - 1, Math.ceil(cb + 1.5)); i++) mx = Math.max(mx, mg[i]); return mx; };
      let bS = 0, bM = NaN; for (let q = 40 * 3; q <= 88 * 3; q++) { const m = q / 3, f = 440 * Math.pow(2, (m - 69) / 12); if (pk(f) < 0.08 * gmx) continue; let sm = 0; for (let h = 1; h <= 8 && h * f < 7000; h++) sm += Math.pow(0.84, h - 1) * pk(h * f); if (sm > bS) { bS = sm; bM = m; } }
      if (!(bM === bM)) continue; midi[k] = bM; if (k + 1 < nF && f0[k + 1] > 0) midi[k + 1] = bM;
      const m0 = bM, F0 = 440 * Math.pow(2, (bM - 69) / 12), mask = new Uint8Array(NF / 2); for (let h = 1; h * F0 < 6000; h++) { const cb = h * F0 / bin, w = Math.max(2, cb * 0.025); for (let i = Math.max(0, Math.floor(cb - w)); i <= Math.min(NF / 2 - 1, Math.ceil(cb + w)); i++) mask[i] = 1; }
      const sal = f => { let sm = 0, cnt = 0; for (let h = 1; h <= 6 && h * f < 6000; h++) { const cb = h * f / bin; let mx = 0, hit = 0; for (let i = Math.max(1, Math.floor(cb - 1.5)); i <= Math.min(NF / 2 - 1, Math.ceil(cb + 1.5)); i++) { if (mask[i]) { hit = 1; continue; } mx = Math.max(mx, mg[i]); } if (hit && h <= 2) return { v: 0, cnt: 0 }; if (!hit) { sm += mx / h; cnt++; } } return { v: cnt >= 3 ? sm / cnt : 0, cnt }; };
      const main = (() => { let sm = 0, c2 = 0; for (let h = 1; h <= 6 && h * F0 < 6000; h++) { sm += pk(h * F0) / h; c2++; } return sm / Math.max(1, c2); })();
      let best = 0, bm = NaN;
      for (let m = Math.round(m0) - 12; m <= Math.round(m0) + 12; m++) { const d = Math.abs(m - m0); if (d < 1.5 || Math.abs(d - 12) < 0.6) continue; const r = sal(440 * Math.pow(2, (m - 69) / 12)); if (r.v > best) { best = r.v; bm = m; } }
      // ±0.5半音で細かく合わせる
      if (bm === bm) for (const dd of [-0.3, 0.3]) { const r = sal(440 * Math.pow(2, (bm + dd - 69) / 12)); if (r.v > best * 1.02) { best = r.v; bm = bm + dd; } }
      const ratio = main > 0 ? best / main : 0; if (ratio >= 0.22) { hm[k] = bm; hs[k] = ratio; }
    }
    P(0.75);
    // ハモリの音にまとめる: 同じ半音が0.15秒以上つづく所。直前0.3秒のメインと同じ音(リバーブの残り)はハモリにしない
    const notes = []; let cur = null;
    const close = k2 => { if (cur && (cur.k1 - cur.k0) * HOP_SEC >= 0.15) notes.push(cur); cur = null; };
    for (let k = 0; k < nF; k += step) {
      const v = hm[k]; if (!(v === v)) { if (cur && k - cur.k1 > 2 * step) close(); continue; } const r = Math.round(v);
      let echo = false; for (let j = Math.max(0, k - Math.round(0.3 / HOP_SEC)); j < k - 2; j++) if (midi[j] === midi[j] && Math.abs(midi[j] - r) < 0.5 && !(Math.abs(midi[k] - r) < 0.5)) { echo = true; break; }
      if (echo) continue;
      if (cur && cur.m === r && k - cur.k1 <= 3 * step) { cur.k1 = k + step; cur.s += hs[k]; cur.c++; } else { close(); cur = { k0: k, k1: k + step, m: r, s: hs[k], c: 1 }; }
    }
    close();
    const harm = notes.map(z => { const km = (z.k0 + z.k1) >> 1; let mm = NaN; for (let d = 0; d < 6 && !(mm === mm); d++) mm = midi[km + d] === midi[km + d] ? midi[km + d] : midi[km - d]; const conf = Math.max(0, Math.min(1, (z.s / z.c - 0.22) / 0.5)) * sepConf;
      return { t0: z.k0 * HOP_SEC, t1: z.k1 * HOP_SEC, m: z.m, mainM: mm === mm ? Math.round(mm * 10) / 10 : null, iv: mm === mm ? Math.round(z.m - mm) : null, conf: Math.round(conf * 100) / 100 }; });
    // 語尾(メインが0.5秒以上とぎれる所)で、残響の減り方とやまびこ
    const E = new Float32Array(Math.floor(N / 441)); for (let b = 0; b < E.length; b++) { let e = 0; for (let i = b * 441; i < (b + 1) * 441; i++) e += M[i] * M[i] + Sd[i] * Sd[i]; E[b] = 10 * Math.log10(e / 441 + 1e-12); } // 10ms
    const tails = [], beat = C.grid && C.grid.bpm ? 60 / C.grid.bpm : 0.5;
    for (let k = 1; k < nF; k++) { if (!(f0[k - 1] > 0) || f0[k] > 0) continue; let g = 0; while (k + g < nF && !(f0[k + g] > 0)) g++; const gap = g * HOP_SEC; if (gap < 0.5) continue;
      let pre = 0, pc = 0; const t = k * HOP_SEC, b0 = Math.round(t / 0.01); for (let b = b0 - 30; b < b0; b++) if (b >= 0) { pre += E[b]; pc++; } pre /= Math.max(1, pc);
      const xs = [], ys = []; for (let b = b0 + 5; b < Math.min(E.length, b0 + Math.min(80, Math.round(gap / 0.01) - 5)); b++) { if (E[b] < pre - 55) break; xs.push((b - b0) * 0.01); ys.push(E[b]); }
      let rt = null, depth = null; if (xs.length >= 8) { const mx = xs.reduce((a, v) => a + v, 0) / xs.length, my = ys.reduce((a, v) => a + v, 0) / ys.length; let sxy = 0, sxx = 0; for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; } const sl2 = sxy / sxx; if (sl2 < -5) rt = Math.min(6, -60 / sl2); depth = ys[Math.min(ys.length - 1, 5)] - pre; }
      // やまびこ: 語尾の前後(前1.5秒〜後ろ最大1.5秒)の音量(dB)から、なだらかな減り方(直線)を引いた残りの自己相関(0.15〜1.0秒)。
      //   残響だけなら山が出ず、同じ間隔のくり返し(ディレイ)があると、その間隔に山が出る
      let echo = null; { let r0 = k - 1; while (r0 > 0 && f0[r0 - 1] > 0) r0--; const rb0 = Math.round(r0 * HOP_SEC / 0.01), lv = []; for (let b = rb0; b < b0; b++) lv.push(E[b]); lv.sort((x, y) => x - y); const lvM = lv.length ? lv[lv.length >> 1] : -60;
        let dry = b0; while (dry > rb0 && E[dry - 1] < lvM - 4) dry--; // 歌そのものが終わった所(そのあとの小さい音はやまびこ・残響)
        const a0 = dry, a1 = Math.min(E.length, b0 + Math.min(150, Math.round(gap / 0.01)));
        // 歌が終わった後に、音量が3dB以上「また上がる」所(やまびこの返り)を探す。残響だけなら下がり続ける
        if (a1 - a0 > 40) { const sm = []; for (let b = a0; b < a1; b++) { let v = 0, c3 = 0; for (let d2 = -2; d2 <= 2; d2++) if (b + d2 >= 0 && b + d2 < E.length) { v += Math.max(E[b + d2], lvM - 30); c3++; } sm.push(v / c3); }
          const bumps = []; let lo = sm[0]; for (let i = 1; i < sm.length - 1; i++) { lo = Math.min(lo, sm[i]); if (sm[i] >= sm[i - 1] && sm[i] > sm[i + 1] && sm[i] - lo >= 3 && sm[i] > lvM - 24) { bumps.push(i); lo = sm[i]; } }
          let on = rb0; for (let b = b0 - 1; b > rb0 + 3; b--) if (E[b] - E[b - 3] >= 6) { on = b; break; } // 最後の音の出だし
          if (bumps.length) { const sp = bumps.length >= 2 ? (bumps[1] - bumps[0]) * 0.01 : null, fromOn = (a0 + bumps[0] - on) * 0.01; const lagE = sp && sp >= 0.12 ? sp : fromOn; if (lagE >= 0.12 && lagE <= 1.6) echo = { lag: Math.round(lagE * 100) / 100, str: bumps.length }; } } }
      tails.push({ t: Math.round(t * 100) / 100, gap: Math.round(gap * 100) / 100, rt: rt && Math.round(rt * 100) / 100, depth: depth && Math.round(depth * 10) / 10, echo });
    }
    const med = a => { a = a.filter(v => v != null).sort((x, y) => x - y); return a.length ? a[a.length >> 1] : null; };
    const rts = tails.map(z => z.rt), eT = tails.filter(z => z.echo), lags = eT.map(z => z.echo.lag), lagM = med(lags);
    const fracs = [0.25, 0.5, 0.75, 1, 1.5, 2], lagBeat = lagM ? fracs.reduce((a, f) => Math.abs(f * beat - lagM) < Math.abs(a * beat - lagM) ? f : a, 0.5) : null;
    // 左右の広がり(歌っている間): ダブル・コーラスがあると大きくなる(ステレオのリバーブでも少し大きくなる)
    const sing = [], tail = []; for (let k = 0; k < nF; k += 4) { const i0 = k * HOP, i1 = Math.min(N, i0 + HOP * 4); let em = 0, es = 0; for (let i = i0; i < i1; i++) { em += M[i] * M[i]; es += Sd[i] * Sd[i]; } if (em < 1e-8) continue; (f0[k] > 0 ? sing : tail).push(10 * Math.log10((es + 1e-12) / em)); }
    const wSing = med(sing), wTail = med(tail);
    // 音色(帯域)と強弱: 取り出した歌 と あなたの歌 をくらべる
    const bands = [[100, 250], [250, 500], [500, 1000], [1000, 2000], [2000, 4000], [4000, 8000], [8000, 14000]];
    const bandDb = (x, fz) => { const acc = new Float64Array(bands.length); let cnt = 0; for (let k = 0; k < fz.length; k += 6) { if (!(fz[k] > 0)) continue; const s0 = k * HOP - 1024; if (s0 < 0 || s0 + 2048 > x.length) continue; const r2 = new Float64Array(2048), i2 = new Float64Array(2048); for (let i = 0; i < 2048; i++) r2[i] = x[s0 + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / 2047)); new FFT(2048).forward(r2, i2); let tot = 0; const e = bands.map(([a, b]) => { let v = 0; for (let i = Math.floor(a / (SR / 2048)); i <= Math.ceil(b / (SR / 2048)); i++) v += r2[i] * r2[i] + i2[i] * i2[i]; tot += v; return v; }); e.forEach((v, j) => acc[j] += v / (tot + 1e-20)); cnt++; } return cnt ? Array.from(acc, v => 10 * Math.log10(v / cnt + 1e-12)) : null; };
    const user = opt.user && opt.user.vocal ? opt.user : null;
    const bO = bandDb(M, f0), bU = user && user.pitch ? bandDb(user.vocal, user.pitch.f0) : null;
    const dynOf = (x, fz) => { const v = []; const W = Math.round(SR * 0.4); for (let i = 0; i + W < x.length; i += W) { const k = Math.round((i + W / 2) / HOP); if (fz[k] > 0) { let e = 0; for (let j = i; j < i + W; j += 2) e += x[j] * x[j]; v.push(10 * Math.log10(e / (W / 2) + 1e-12)); } } v.sort((a, b) => a - b); return v.length > 10 ? v[Math.floor(v.length * 0.9)] - v[Math.floor(v.length * 0.1)] : null; };
    const dO = dynOf(M, f0), dU = user && user.pitch ? dynOf(user.vocal, user.pitch.f0) : null;
    // 原曲とカラオケのキーのちがい(ずれていると引き算が効かない)
    let keyShift = 0; try { const ko = detectKey(om); keyShift = ((ko.tonic - C.key.tonic + 18) % 12) - 6; } catch (e) {}
    const mainNotes = []; { let a = -1, mm = 0; for (let k = 0; k <= nF; k++) { const v = k < nF ? midi[k] : NaN, r = v === v ? Math.round(v) : null; if (a >= 0 && (r !== mm || k === nF)) { if ((k - a) * HOP_SEC >= 0.08) mainNotes.push([Math.round(a * HOP_SEC * 100) / 100, Math.round(k * HOP_SEC * 100) / 100, mm]); a = -1; } if (r != null && a < 0) { a = k; mm = r; } } }
    P(1);
    return { ver: 2, mode: subOK ? 'subtract' : 'mix', how, ratio, driftMs: drift ? Math.round(drift.span / SR * 1000 * 10) / 10 : null, lagSec: lag / SR, sepConf, keyShift, voicedSec: f0.filter(v => v > 0).length * HOP_SEC, mainNotes, harm,
      reverb: { rt: med(rts), n: rts.filter(v => v != null).length, depth: med(tails.map(z => z.depth)) }, echo: { lag: lagM, beats: lagBeat, share: tails.length ? eT.length / tails.length : 0, n: eT.length, at: eT.map(z => z.t) },
      tails, width: { sing: wSing, tail: wTail }, bandsOrig: bO, bandsUser: bU, dynOrig: dO, dynUser: dU, bpm: C.grid && C.grid.bpm, stem: opt.wantStem ? M : undefined };
  }

  // v4.6+ 採譜AI(Basic Pitch)の「音ごとの確率」(フレーム×88鍵, fps)から、メイン以外の声(ハモリ)の音を取り出し、信号処理の結果とまとめる
  //   メイン(信号処理で推定)から1.5半音以内・オクターブちがい・直前0.3秒のメインと同じ音(残響)は除く。0.15秒以上つづく音だけ
  function harmFromNotes(post, fps, mainNotes, sepConf, thr) {
    thr = thr || 0.4; const nB = 88, nT = Math.floor(post.length / nB), mainAt = new Float32Array(nT).fill(NaN);
    for (const [a, b, m] of mainNotes) for (let f = Math.max(0, Math.floor(a * fps)); f < Math.min(nT, Math.ceil(b * fps)); f++) mainAt[f] = m;
    // メインが信号処理で取れなかった所(2つの声で音程が取りにくい所): AIで鳴っている音のうち、直前(3秒以内)のメインに近い音をメインとする
    { let last = NaN, lf = -1e9; for (let f = 0; f < nT; f++) { if (mainAt[f] === mainAt[f]) { last = mainAt[f]; lf = f; continue; } if (!(last === last) || (f - lf) / fps > 3) continue;
      let bm = -1, bd = 99; for (let i = 0; i < nB; i++) if (post[f * nB + i] >= thr) { const d = Math.abs(21 + i - last); if (d < bd && d <= 7) { bd = d; bm = 21 + i; } } if (bm >= 0) { mainAt[f] = bm; last = bm; lf = f; } } }
    const W = Math.round(0.3 * fps), hm = new Int16Array(nT).fill(-1), hp = new Float32Array(nT);
    for (let f = 0; f < nT; f++) { const m0 = mainAt[f]; if (!(m0 === m0)) continue; let best = 0, bm = -1;
      for (let i = 0; i < nB; i++) { const v = post[f * nB + i]; if (v < thr) continue; const m = 21 + i, d = Math.abs(m - m0); if (d < 1.5 || Math.abs(d - 12) < 0.6 || Math.abs(d - 24) < 0.6) continue;
        let echo = false; for (let j = Math.max(0, f - W); j < f - 1; j++) if (Math.abs(mainAt[j] - m) < 0.5) { echo = true; break; } if (echo) continue; if (v > best) { best = v; bm = m; } }
      if (bm >= 0) { hm[f] = bm; hp[f] = best; } }
    const out = []; let cur = null; const close = () => { if (cur && (cur.f1 - cur.f0) / fps >= 0.15) out.push(cur); cur = null; };
    for (let f = 0; f < nT; f++) { const v = hm[f]; if (v < 0) { if (cur && f - cur.f1 > 2) close(); continue; } if (cur && cur.m === v && f - cur.f1 <= 3) { cur.f1 = f + 1; cur.s += hp[f]; cur.c++; } else { close(); cur = { f0: f, f1: f + 1, m: v, s: hp[f], c: 1 }; } }
    close();
    return out.map(z => { const fm = (z.f0 + z.f1) >> 1, mm = mainAt[fm]; return { t0: z.f0 / fps, t1: z.f1 / fps, m: z.m, mainM: mm === mm ? mm : null, iv: mm === mm ? Math.round(z.m - mm) : null, conf: Math.round(Math.min(1, z.s / z.c) * sepConf * 100) / 100, src: 'ai' }; });
  }
  // 2つの結果をまとめる: 両方が同じ時間・同じ音程差 → 確かさを上げる。片方だけ → そのまま(AIだけのものは少し下げる)
  function mergeHarm(dsp, ai) {
    const ov = (a, b) => Math.min(a.t1, b.t1) - Math.max(a.t0, b.t0), out = [];
    for (const a of ai) { const d = dsp.find(x => ov(x, a) > 0.5 * Math.min(x.t1 - x.t0, a.t1 - a.t0) && x.iv === a.iv); out.push(Object.assign({}, a, { conf: Math.min(1, Math.round((d ? Math.max(a.conf, d.conf) + 0.15 : a.conf * 0.85) * 100) / 100), src: d ? 'both' : 'ai' })); }
    for (const d of dsp) if (!out.some(a => ov(a, d) > 0.5 * Math.min(a.t1 - a.t0, d.t1 - d.t0))) out.push(Object.assign({}, d, { src: 'dsp' }));
    return out.sort((x, y) => x.t0 - y.t0);
  }
  function robotShift(plan) { // 全部の音を1つの高さにそろえる
    const v = []; for (let k = 0; k < plan.midi.length; k++) if (plan.midi[k] === plan.midi[k]) v.push(plan.midi[k]);
    v.sort((a, b) => a - b);
    const note = v.length ? Math.round(v[v.length >> 1]) : 60, shift = new Float32Array(plan.midi.length);
    for (let k = 0; k < shift.length; k++) if (plan.midi[k] === plan.midi[k]) shift[k] = Math.max(-12, Math.min(12, note - plan.midi[k]));
    return shift;
  }

  // ---------- ピッチシフト (PSOLA) ----------
  function psola(x, f0, shift, progress, warp, formant, guard) { // guard(v4.0): ピッチマークの検証(となりの周期と似ていない所は音程を動かさない)と、声の出だしのマークを波の山に合わせる
    const fm = formant > 0 ? formant : 1;
    const n = x.length, nF = f0.length;
    const out = new Float32Array(n), wsum = new Float32Array(n), alOut = new Float32Array(Math.ceil(n / HOP) + 2).fill(0);
    const DEF = 220;
    function periodAt(pos) { // 0 = 無声
      const fp = pos / HOP, i0 = Math.min(nF - 1, Math.floor(fp)), i1 = Math.min(nF - 1, i0 + 1), fr = fp - i0;
      const a = f0[i0], b = f0[i1];
      if (a > 0 && b > 0) return SR / (a + (b - a) * fr);
      const near = fr < 0.5 ? a : b;
      return near > 0 ? SR / near : 0;
    }
    function shiftAt(pos) {
      const fp = pos / HOP, i0 = Math.min(nF - 1, Math.floor(fp)), i1 = Math.min(nF - 1, i0 + 1), fr = fp - i0;
      return shift[i0] + (shift[i1] - shift[i0]) * fr;
    }
    // 分析マーク
    const marks = [], per = [], voi = [], mok = [];
    let m = 0, wasV = false;
    while (m < n) {
      let T = periodAt(m);
      const v = T > 0;
      if (!v) T = DEF;
      /* v4.0: 声の出だしのマークを波の山に合わせる案は、マーク検証と組み合わせると大きく下げたときに倍の高さの成分が出たため採用しない(test/v4_engine.js) */
      wasV = v;
      marks.push(m); per.push(T); voi.push(v); mok.push(1);
      let c = Math.round(m + T);
      if (v && c < n) {
        const h = Math.min(400, Math.round(T / 2)), R = Math.max(1, Math.round(0.1 * T));
        if (m - h >= 0 && c + R + h < n && c - R - h >= 0) {
          let best = -1e30, bd = 0, bn = 0;
          for (let dd = -R; dd <= R; dd++) {
            const p = c + dd;
            let sxy = 0, e = 0, em = 0;
            for (let j = -h; j <= h; j += 2) { const y = x[p + j], z = x[m + j]; sxy += z * y; e += y * y; em += z * z; }
            const val = sxy / Math.sqrt(e + 1e-12);
            if (val > best) { best = val; bd = dd; bn = sxy / Math.sqrt(e * em + 1e-18); }
          }
          c += bd;
          if (guard) mok[mok.length - 1] = Math.max(0, Math.min(1, (bn - 0.55) / 0.3)); // となりの周期との相関が低い = 声帯の振動が不規則 / マークがずれている
        }
      }
      m = Math.max(m + 16, c);
    }
    // 合成 (warp: 出力時刻→元の時刻 の区分線形マップ。リズム補正用)
    const nM = marks.length;
    if (guard) { const sm = Float32Array.from(mok); for (let i = 0; i < nM; i++) { let a = 0, c = 0; for (let j = Math.max(0, i - 3); j <= Math.min(nM - 1, i + 3); j++) { a += mok[j]; c++; } sm[i] = a / c; } for (let i = 0; i < nM; i++) mok[i] = sm[i]; } // 1周期ごとに補正が入ったり切れたりしないよう、前後3周期でならす
    const wi = warp ? warp.ain : null, wo = warp ? warp.aout : null;
    let ts = 0, ai = 0, cnt = 0, wj = 0, useed = 12345;
    const urnd = () => (useed = (useed * 1664525 + 1013904223) >>> 0) / 4294967296, UTU = 1400, UJ = 1.5, USTEP = 0.5, UG = 1.63; // 長さ1400の粒を4枚重ね、読む所を±1050ずらす。UG: ずらして重ねると小さくなる(-4.2dB)ぶん
    while (ts < n) {
      let ta = ts, slope = 1;
      if (warp) {
        const tsec = ts / SR;
        while (wj + 2 < wo.length && tsec >= wo[wj + 1]) wj++;
        slope = (wi[wj + 1] - wi[wj]) / (wo[wj + 1] - wo[wj]);
        ta = (wi[wj] + (tsec - wo[wj]) * slope) * SR;
      }
      while (ai + 1 < nM && Math.abs(marks[ai + 1] - ta) <= Math.abs(marks[ai] - ta)) ai++;
      if (!voi[ai] && Math.abs(slope - 1) > 2e-3) { // 息・子音・消え際を伸び縮みするとき: 同じ波を200Hzで繰り返すと「ジー」というジャミ音になる → 長めの粒を毎回少しずらした所から読む
        const TU = UTU, rd = Math.max(0, Math.min(n - 1, Math.round(ta + (urnd() - 0.5) * UJ * TU)));
        const i0 = Math.max(0, Math.ceil(ts - TU)), i1 = Math.min(n - 1, Math.floor(ts + TU)), kk = Math.PI / TU;
        for (let i = i0; i <= i1; i++) { const u = i - ts, w = 0.5 + 0.5 * Math.cos(kk * u), sp = rd + u; out[i] += w * (sp >= 0 && sp < n ? x[sp | 0] : 0) * UG; wsum[i] += w; }
        ts += Math.round(TU * USTEP); if (progress && (++cnt & 4095) === 0) progress(ts / n); continue;
      }
      const a = marks[ai], T = per[ai];
      const st = voi[ai] ? shiftAt(a) * (guard ? mok[ai] : 1) : 0;
      const ident = Math.abs(st) < 0.01, r = ident ? 1 : Math.pow(2, st / 12);
      // 声の太さ(フォルマント): 切り出した1周期ぶんの波を伸び縮みさせる。大きく下げるときは窓が届くように太さも下げる
      const al = Math.max(0.5, Math.min(1.6, Math.min(fm, 1.33 * r))), Tw = T / al;
      if (voi[ai]) { const fj = Math.min(alOut.length - 1, Math.floor(ts / HOP)); if (!alOut[fj] || al < alOut[fj]) alOut[fj] = al; }
      const i0 = Math.max(0, Math.ceil(ts - Tw)), i1 = Math.min(n - 1, Math.floor(ts + Tw));
      const k = Math.PI / Tw;
      for (let i = i0; i <= i1; i++) {
        const u = i - ts, w = 0.5 + 0.5 * Math.cos(k * u);
        const sp = a + u * al;
        let val;
        if (sp < 1 || sp >= n - 2) val = (sp >= 0 && sp < n) ? x[Math.min(n - 1, Math.round(sp))] : 0;
        else {
          const j = Math.floor(sp), fr = sp - j;
          if (fr === 0) val = x[j];
          else {
            const y0 = x[j - 1], y1 = x[j], y2 = x[j + 1], y3 = x[j + 2];
            const c1 = 0.5 * (y2 - y0), c2 = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3, c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
            val = ((c3 * fr + c2) * fr + c1) * fr + y1;
          }
        }
        out[i] += w * val; wsum[i] += w;
      }
      let nx;
      if (!ident) nx = ts + T / r;
      else if (Math.abs(slope - 1) < 1e-4) nx = ai + 1 < nM ? marks[ai + 1] + (ts - ta) : ts + T;
      else nx = ts + T;
      ts = Math.max(ts + 8, nx);
      if (progress && (++cnt & 4095) === 0) progress(ts / n);
    }
    for (let i = 0; i < n; i++) out[i] = wsum[i] > 1e-4 ? out[i] / wsum[i] : x[i];
    let last = 1; for (let j = 0; j < alOut.length; j++) { if (alOut[j]) last = alOut[j]; else alOut[j] = last; }
    out._al = alOut;
    return out;
  }

  // ---------- スペクトルで声を変える(声の太さ・ささやき・ボコーダー) ----------
  // 声の「響きの形」(スペクトル包絡)を true envelope 法で求め、それを伸び縮みさせたり、別の音に移したりする
  function makeEnv(N0) {
    // 包絡はなめらかなので、スペクトルを4つずつ(山の高さ=最大値で)まとめた小さいFFTで求めて、元の細かさに戻す(約5倍速い)
    const DS = 4, N = N0 / DS, K0 = N0 / 2 + 1, K = N / 2 + 1, fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N);
    const A = new Float64Array(K), E = new Float64Array(K), L = new Float64Array(K), Eo = new Float64Array(K0);
    function cep(src, nc) {
      for (let k = 0; k < K; k++) re[k] = src[k];
      for (let k = 1; k < N / 2; k++) re[N - k] = src[k];
      im.fill(0); fft.forward(re, im);
      for (let q = nc + 1; q < N - nc; q++) re[q] = 0;
      re[nc] *= 0.5; re[N - nc] *= 0.5;
      im.fill(0); fft.forward(re, im);
      for (let k = 0; k < K; k++) E[k] = re[k] / N;
    }
    return function env(logmag, nc, iters) { // true envelope: 山の頂点をなぞる包絡
      nc = Math.min(nc, N / 2 - 2);
      for (let k = 0; k < K; k++) { let m = -1e9; for (let j = Math.max(0, k * DS - 2); j <= Math.min(K0 - 1, k * DS + 1); j++) if (logmag[j] > m) m = logmag[j]; L[k] = m; }
      cep(L, nc);
      for (let it = 0; it < iters; it++) { for (let k = 0; k < K; k++) A[k] = L[k] > E[k] ? L[k] : E[k]; cep(A, nc); }
      for (let k = 0; k < K0; k++) { const sk = k / DS, a = Math.min(K - 1, Math.floor(sk)), b = Math.min(K - 1, a + 1), fr = sk - a; Eo[k] = E[a] + (E[b] - E[a]) * fr; }
      return Eo;
    };
  }
  function polyblep(t, dt) { if (t < dt) { t /= dt; return t + t - t * t - 1; } if (t > 1 - dt) { t = (t - 1) / dt; return t * t + t + t + 1; } return 0; }
  function carrierWave(notes, n) { // ボコーダー用: 音程どおりののこぎり波(無声の所はノイズ)
    const c = new Float32Array(n); let ph = 0, f = 0, seed = 12345;
    for (let i = 0; i < n; i++) {
      const m = notes[Math.min(notes.length - 1, Math.floor(i / HOP))];
      if (m === m) {
        const ft = 440 * Math.pow(2, (m - 69) / 12); f = f ? f + (ft - f) * 0.01 : ft;
        const dt = f / SR; c[i] = 0.5 * (2 * ph - 1 - polyblep(ph, dt)); ph += dt; if (ph >= 1) ph -= 1;
      } else { seed = (seed * 1664525 + 1013904223) >>> 0; c[i] = (seed / 4294967296 - 0.5) * 0.6; }
    }
    return c;
  }
  // o: { beta: フレームごとの太さの倍率, f0: フレームごとの出力の高さ(Hz,0=無声), whisper: 0〜1, notes: ボコーダーの音(MIDI), vocAmt: 0〜1 }
  function spectralVoice(x, o, progress) {
    const N = 2048, H = HOP, K = N / 2 + 1, n = x.length, nF = Math.ceil(n / H) + 1;
    const fft = new FFT(N), env = makeEnv(N), envC = makeEnv(N);
    const re = new Float64Array(N), im = new Float64Array(N), cr = new Float64Array(N), ci = new Float64Array(N);
    const win = new Float64Array(N); for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
    const lm = new Float64Array(K), lc = new Float64Array(K), Ev = new Float64Array(K), Ew = new Float64Array(K);
    const car = o.notes ? carrierWave(o.notes, n) : null;
    const y = new Float32Array(n), ws = new Float32Array(n);
    let seed = 987;
    const gauss = () => { seed = (seed * 1664525 + 1013904223) >>> 0; const u = (seed + 1) / 4294967297; seed = (seed * 1664525 + 1013904223) >>> 0; const v = seed / 4294967296; return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    for (let f = 0; f < nF; f++) {
      const s0 = f * H - N / 2;
      let e0 = 0;
      for (let i = 0; i < N; i++) { const j = s0 + i, v = j >= 0 && j < n ? x[j] * win[i] : 0; re[i] = v; im[i] = 0; e0 += v * v; }
      if (e0 < 1e-10) continue;
      fft.forward(re, im);
      for (let k = 0; k < K; k++) lm[k] = 0.5 * Math.log(re[k] * re[k] + im[k] * im[k] + 1e-12);
      const fq = o.f0 ? o.f0[Math.min(o.f0.length - 1, f)] : 0;
      const nc = fq > 0 ? Math.max(18, Math.min(90, Math.floor(0.5 * SR / fq))) : 60;
      const E = env(lm, nc, 2);
      for (let k = 0; k < K; k++) Ev[k] = E[k];
      // 1) 声の太さ: 包絡を周波数方向に beta 倍に伸ばす
      const beta = o.beta ? o.beta[Math.min(o.beta.length - 1, f)] : 1;
      if (Math.abs(beta - 1) > 0.01) {
        for (let k = 0; k < K; k++) {
          const sk = Math.min(K - 1, k / beta), a = Math.floor(sk), b = Math.min(K - 1, a + 1), fr = sk - a;
          Ew[k] = Ev[a] + (Ev[b] - Ev[a]) * fr;
          const g = Math.exp(Math.max(-2.5, Math.min(2.5, Ew[k] - Ev[k])));
          re[k] *= g; im[k] *= g;
        }
        for (let k = 0; k < K; k++) Ev[k] = Ew[k];
      }
      // 2) ボコーダー: のこぎり波に声の響きを移す
      if (car && o.vocAmt > 0) {
        let ex = 0; for (let k = 0; k < K; k++) ex += re[k] * re[k] + im[k] * im[k];
        for (let i = 0; i < N; i++) { const j = s0 + i; cr[i] = j >= 0 && j < n ? car[j] * win[i] : 0; ci[i] = 0; }
        fft.forward(cr, ci);
        for (let k = 0; k < K; k++) lc[k] = 0.5 * Math.log(cr[k] * cr[k] + ci[k] * ci[k] + 1e-12);
        const Ec = envC(lc, 30, 1);
        let ev = 0;
        for (let k = 0; k < K; k++) { const g = Math.exp(Math.max(-12, Math.min(12, Ev[k] - Ec[k]))); cr[k] *= g; ci[k] *= g; ev += cr[k] * cr[k] + ci[k] * ci[k]; }
        const nrm = ev > 0 ? Math.sqrt(ex / ev) : 0, a = o.vocAmt;
        for (let k = 0; k < K; k++) { re[k] = (1 - a) * re[k] + a * nrm * cr[k]; im[k] = (1 - a) * im[k] + a * nrm * ci[k]; }
      }
      // 3) ささやき: 声の響きはそのまま、声帯の振動(音程)を息の音に置き換える
      if (o.whisper > 0) {
        let ex = 0, ew = 0;
        for (let k = 0; k < K; k++) ex += re[k] * re[k] + im[k] * im[k];
        for (let k = 0; k < K; k++) { const m = Math.exp(Ev[k]); cr[k] = m * gauss(); ci[k] = m * gauss(); ew += cr[k] * cr[k] + ci[k] * ci[k]; }
        const nrm = ew > 0 ? Math.sqrt(ex / ew) : 0, a = o.whisper, ka = Math.sqrt(1 - a), kb = Math.sqrt(a) * nrm;
        for (let k = 0; k < K; k++) { re[k] = ka * re[k] + kb * cr[k]; im[k] = ka * im[k] + kb * ci[k]; }
      }
      for (let k = 1; k < N / 2; k++) { re[N - k] = re[k]; im[N - k] = -im[k]; }
      im[0] = 0; im[N / 2] = 0;
      fft.forward(im, re); // 逆変換(実部と虚部を入れ替えて順変換)
      for (let i = 0; i < N; i++) { const j = s0 + i; if (j < 0 || j >= n) continue; y[j] += re[i] / N * win[i]; ws[j] += win[i] * win[i]; }
      if (progress && (f & 255) === 0) progress(f / nF);
    }
    for (let i = 0; i < n; i++) y[i] = ws[i] > 1e-3 ? y[i] / ws[i] : 0;
    return y;
  }
  // ---------- カラオケのキー変更(和音もOK): 位相ボコーダーで伸ばして(ピーク位相固定) → 元の長さに縮める ----------
  function pvStretch(x, a, progress, p0, p1) { // 長さを a 倍(音程はそのまま)
    const N = 4096, Hs = 1024, Ha = Hs / a, K = N / 2 + 1, n = x.length;
    const nF = Math.max(1, Math.floor((n - 1) / Ha) + 1), outN = Math.ceil(n * a) + N;
    const fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), win = new Float64Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
    const y = new Float32Array(outN), ws = new Float32Array(outN);
    const mag = new Float64Array(K), ph = new Float64Array(K), prevPh = new Float64Array(K), sumPh = new Float64Array(K), outPh = new Float64Array(K), pk = new Int32Array(K);
    let first = true;
    for (let f = 0; f < nF; f++) {
      const ca = f * Ha, s0 = Math.round(ca) - N / 2, so = f * Hs - N / 2;
      for (let i = 0; i < N; i++) { const j = s0 + i; re[i] = j >= 0 && j < n ? x[j] * win[i] : 0; im[i] = 0; }
      fft.forward(re, im);
      for (let k = 0; k < K; k++) { mag[k] = Math.hypot(re[k], im[k]); ph[k] = Math.atan2(im[k], re[k]); }
      if (first) { for (let k = 0; k < K; k++) sumPh[k] = ph[k]; first = false; }
      else {
        // ピーク(山)を探して、山の位相だけ進め、周りのビンは山と同じだけ回す(identity phase locking)
        let np = 0; for (let k = 2; k < K - 2; k++) if (mag[k] > mag[k - 1] && mag[k] >= mag[k + 1] && mag[k] > mag[k - 2] && mag[k] >= mag[k + 2]) pk[np++] = k;
        if (!np) pk[np++] = 1;
        let b = 0;
        for (let q = 0; q < np; q++) {
          const k = pk[q], om = 2 * Math.PI * k / N, dp = ph[k] - prevPh[k] - om * Ha, w = dp - 2 * Math.PI * Math.round(dp / (2 * Math.PI));
          const nph = sumPh[k] + (om + w / Ha) * Hs, rot = nph - ph[k];
          const end = q + 1 < np ? (pk[q] + pk[q + 1]) >> 1 : K - 1;
          for (; b <= end; b++) outPh[b] = ph[b] + rot;
        }
        for (let k = 0; k < K; k++) sumPh[k] = outPh[k];
      }
      for (let k = 0; k < K; k++) prevPh[k] = ph[k];
      for (let k = 0; k < K; k++) { re[k] = mag[k] * Math.cos(sumPh[k]); im[k] = mag[k] * Math.sin(sumPh[k]); if (k > 0 && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; } }
      im[0] = 0; im[N / 2] = 0;
      fft.forward(im, re);
      for (let i = 0; i < N; i++) { const j = so + i; if (j < 0 || j >= outN) continue; y[j] += re[i] / N * win[i]; ws[j] += win[i] * win[i]; }
      if (progress && (f & 63) === 0) progress(p0 + (p1 - p0) * f / nF);
    }
    for (let i = 0; i < outN; i++) y[i] = ws[i] > 1e-3 ? y[i] / ws[i] : 0;
    return y;
  }
  function resampleTo(y, n, r) { // y を r 倍の速さで読んで長さ n に(r>1は低域通過してから)。係数は表にして速く
    const out = new Float32Array(n), A = 16, PH = 512, fc = Math.min(1, 1 / r) * 0.95, T = new Float32Array(PH * 2 * A);
    for (let ph = 0; ph < PH; ph++) { const fr = ph / PH; let sm = 0;
      for (let q = 0; q < 2 * A; q++) { const d = fr + A - 1 - q, xx = Math.PI * d * fc, sinc = Math.abs(d) < 1e-9 ? 1 : Math.sin(xx) / xx, w = 0.5 + 0.5 * Math.cos(Math.PI * d / A); T[ph * 2 * A + q] = sinc * w; sm += sinc * w; }
      for (let q = 0; q < 2 * A; q++) T[ph * 2 * A + q] /= sm; }
    const L = y.length;
    for (let i = 0; i < n; i++) {
      const c = i * r, ci = Math.floor(c), ph = Math.min(PH - 1, Math.round((c - ci) * PH)), base = ph * 2 * A, j0 = ci - A + 1;
      let s2 = 0;
      if (j0 >= 0 && j0 + 2 * A <= L) for (let q = 0; q < 2 * A; q++) s2 += y[j0 + q] * T[base + q];
      else for (let q = 0; q < 2 * A; q++) { const j = j0 + q; if (j >= 0 && j < L) s2 += y[j] * T[base + q]; }
      out[i] = s2;
    }
    return out;
  }
  function pitchShiftMusic(L, R, st, progress) {
    if (!st) return [L, R];
    const a = Math.pow(2, st / 12), n = L.length;
    const sL = pvStretch(L, a, progress, 0, 0.4), sR = R === L ? sL : pvStretch(R, a, progress, 0.4, 0.8);
    const oL = resampleTo(sL, n, a); if (progress) progress(0.9);
    const oR = R === L ? oL : resampleTo(sR, n, a);
    return [oL, oR];
  }

  // 入力側のフレーム情報(音・高さ)を、リズム補正後の時間に並べ直す
  function toOutFrames(vals, tOut, nOut) {
    const o = new Float32Array(nOut).fill(NaN);
    for (let k = 0; k < vals.length; k++) { const v = vals[k]; if (v !== v) continue; const j = Math.round(tOut[k] / HOP_SEC); if (j >= 0 && j < nOut) o[j] = v; }
    for (let j = 1; j < nOut - 1; j++) if (o[j] !== o[j] && o[j - 1] === o[j - 1] && o[j + 1] === o[j + 1]) o[j] = o[j - 1]; // 1フレームの抜けを埋める
    for (let j = 1; j < nOut - 2; j++) if (o[j] !== o[j] && o[j - 1] === o[j - 1] && o[j + 2] === o[j + 2]) { o[j] = o[j - 1]; o[j + 1] = o[j + 2]; }
    return o;
  }

  // ---------- ボーカル処理 ----------
  function activeRms(x) {
    const B = 2048; let pk = 0; const e = [];
    for (let s = 0; s + B <= x.length; s += B) {
      let sm = 0; for (let i = s; i < s + B; i++) sm += x[i] * x[i];
      sm /= B; e.push(sm); if (sm > pk) pk = sm;
    }
    let tot = 0, c = 0;
    for (const v of e) if (v > pk * 1e-3) { tot += v; c++; }
    return c ? Math.sqrt(tot / c) : 0;
  }
  function compress(x, thr, ratio, knee, attMs, relMs) {
    const aE = Math.exp(-1 / (0.005 * SR)), aA = Math.exp(-1 / (attMs * SR / 1000)), aR = Math.exp(-1 / (relMs * SR / 1000));
    const sl = 1 - 1 / ratio;
    let env = 0, gr = 0;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      env = aE * env + (1 - aE) * v * v;
      const over = 10 * Math.log10(env + 1e-12) - thr;
      let red;
      if (over <= -knee / 2) red = 0;
      else if (over >= knee / 2) red = over * sl;
      else { const t = over + knee / 2; red = sl * t * t / (2 * knee); }
      gr = red > gr ? aA * gr + (1 - aA) * red : aR * gr + (1 - aR) * red;
      x[i] = gr > 1e-4 ? v * Math.pow(10, -gr / 20) : v;
    }
  }
  function processVocal(x) {
    const y = new Float32Array(x);
    const hp = biquad('hp', 90, 0.707);
    filt(y, hp); filt(y, hp);
    filt(y, biquad('peak', 280, 1.0, -2));
    filt(y, biquad('peak', 3500, 0.8, 2));
    filt(y, biquad('hs', 9000, 0.707, 2.5));
    const r = activeRms(y);
    if (r > 0) { const g = 0.1 / r; for (let i = 0; i < y.length; i++) y[i] *= g; }
    compress(y, -22, 3, 6, 6, 140);
    return y;
  }
  function reverb(x) {
    const TAIL = Math.round(2.5 * SR), n = x.length + TAIL, PRE = Math.round(0.025 * SR);
    const inp = new Float32Array(n);
    inp.set(x, 0);
    filt(inp, biquad('hp', 250, 0.707)); filt(inp, biquad('lp', 7000, 0.707));
    const combT = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], apT = [556, 441, 341, 225];
    const FB = 0.82, DAMP = 0.3;
    function chan(spread) {
      const out = new Float32Array(n);
      const cb = combT.map(t => new Float32Array(t + spread)), ci = new Int32Array(8), cs = new Float32Array(8);
      const ab = apT.map(t => new Float32Array(t + spread)), ap = new Int32Array(4);
      for (let i = 0; i < n; i++) {
        const v = i >= PRE ? inp[i - PRE] * 0.015 : 0;
        let o = 0;
        for (let c = 0; c < 8; c++) {
          const buf = cb[c], idx = ci[c], y = buf[idx];
          cs[c] = y * (1 - DAMP) + cs[c] * DAMP;
          buf[idx] = v + cs[c] * FB;
          ci[c] = idx + 1 === buf.length ? 0 : idx + 1;
          o += y;
        }
        for (let c = 0; c < 4; c++) {
          const buf = ab[c], idx = ap[c], bo = buf[idx], y = bo - o;
          buf[idx] = o + bo * 0.5;
          ap[c] = idx + 1 === buf.length ? 0 : idx + 1;
          o = y;
        }
        out[i] = o;
      }
      return out;
    }
    return [chan(0), chan(23)];
  }


  // ---------- オンセット包絡 / テンポ / タイミング ----------
  const ENV_DT = 256 / SR;
  function onsetEnv(mono, progress) {
    const x = decimate(mono, 2), N = 1024, H = 128, fft = new FFT(N);
    const re = new Float64Array(N), im = new Float64Array(N), prev = new Float32Array(N / 2), win = new Float64Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
    const nFr = Math.max(0, Math.floor((x.length - N) / H) + 1), env = new Float32Array(nFr + 6);
    for (let f = 0; f < nFr; f += 2) { // 2フレームを1回のFFTで
      const s = f * H, two = f + 1 < nFr;
      for (let i = 0; i < N; i++) { re[i] = x[s + i] * win[i]; im[i] = two ? x[s + H + i] * win[i] : 0; }
      fft.forward(re, im);
      for (let q = 0; q < (two ? 2 : 1); q++) {
        let fl = 0;
        for (let k = 1; k < N / 2; k++) {
          const j = N - k, mr = q ? 0.5 * (im[k] + im[j]) : 0.5 * (re[k] + re[j]), mi = q ? -0.5 * (re[k] - re[j]) : 0.5 * (im[k] - im[j]);
          const L = Math.log(1 + 100 * Math.sqrt(mr * mr + mi * mi)), d = L - prev[k];
          if (d > 0) fl += d;
          prev[k] = L;
        }
        env[f + q + 6] = f + q ? fl : 0;
      }
      if (progress && (f & 1022) === 0) progress(f / nFr);
    }
    let m = 0; for (let i = 0; i < env.length; i++) m += env[i];
    m /= env.length || 1;
    if (m > 0) for (let i = 0; i < env.length; i++) env[i] /= m;
    return env;
  }
  function beatGrid(env) {
    const n = env.length, e = new Float32Array(n);
    for (let i = 0; i < n; i++) e[i] = env[i] - 1;
    const lmin = Math.round(60 / 200 / ENV_DT), lmax = Math.round(60 / 60 / ENV_DT);
    const ac = l => { let s = 0; for (let i = 0; i + l < n; i++) s += e[i] * e[i + l]; return s / Math.max(1, n - l); };
    let best = lmin, bs = -1e30;
    const sc = new Float64Array(lmax + 2);
    for (let l = lmin; l <= lmax; l++) {
      const bpm = 60 / (l * ENV_DT), w = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120) / 0.7, 2));
      sc[l] = (ac(l) + 0.5 * ac(2 * l)) * w;
      if (sc[l] > bs) { bs = sc[l]; best = l; }
    }
    const P0 = best, at = p => { const i = Math.floor(p), f = p - i; return i >= 0 && i + 1 < n ? env[i] * (1 - f) + env[i + 1] * f : 0; };
    const WL = 12, WH = 6, wins = [];
    const dur = n * ENV_DT, nW = Math.max(1, Math.ceil(dur / WH) - 1);
    for (let w = 0; w < nW; w++) {
      const w0 = w * WH / ENV_DT, w1 = Math.min(n - 1, w0 + WL / ENV_DT);
      let bp = P0, bph = 0, bsc = -1;
      for (let r = -20; r <= 20; r++) {
        const p = P0 * (1 + r * 0.001);
        for (let ph = 0; ph < p; ph += 0.5) {
          let s = 0, c = 0;
          for (let t = w0 + ph; t < w1; t += p) { s += at(t) + 0.5 * at(t + p / 2); c++; }
          s /= c || 1;
          if (s > bsc) { bsc = s; bp = p; bph = ph; }
        }
      }
      wins.push({ period: bp * ENV_DT, phase: (w0 + bph) * ENV_DT, score: bsc });
    }
    const ss = wins.map(w => w.score).sort((a, b) => a - b), med = ss[ss.length >> 1];
    for (let w = 0; w < wins.length; w++) {
      if (wins[w].score < 0.4 * med) { // リズムが弱い区間は近くの区間から引き継ぐ
        let j = -1;
        for (let d = 1; d < wins.length && j < 0; d++) {
          if (w - d >= 0 && wins[w - d].score >= 0.4 * med) j = w - d;
          else if (w + d < wins.length && wins[w + d].score >= 0.4 * med) j = w + d;
        }
        if (j >= 0) wins[w] = { period: wins[j].period, phase: wins[j].phase, score: wins[w].score };
      }
    }
    return { bpm: 60 / (P0 * ENV_DT), wins, WH };
  }
  // 拍の位置の一覧(テンポの揺れに追従)
  function beatTimes(grid, dur) {
    const out = [], W = grid.wins, wi = t => W[Math.max(0, Math.min(W.length - 1, Math.round((t - grid.WH) / grid.WH)))];
    let t = W[0].phase;
    while (t - W[0].period >= 0) t -= W[0].period;
    while (t < dur && out.length < 20000) {
      out.push(t);
      const w = wi(t), s = w.period;
      let nx = t + s;
      const g = Math.round((nx - w.phase) / s) * s + w.phase;
      if (Math.abs(g - nx) < 0.25 * s) nx = g;
      if (nx <= t + 0.5 * s) nx = t + s;
      t = nx;
    }
    return Float32Array.from(out);
  }
  // 時刻t(曲の時間)に一番近い16分音符グリッドまでの差
  function gridFromBeats(b, base) { // 拍の時刻の並び → beatGrid と同じ形(窓ごとの拍の長さと位置)
    const WH = base.WH, n = base.wins.length, wins = [];
    for (let w = 0; w < n; w++) {
      const c = (w + 1) * WH; let lo = 0, hi = b.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (b[m] <= c) lo = m; else hi = m; }
      const i0 = Math.max(1, lo - 4), i1 = Math.min(b.length - 1, lo + 5), d = []; for (let i = i0; i <= i1; i++) d.push(b[i] - b[i - 1]); d.sort((p, q) => p - q);
      wins.push({ period: d[d.length >> 1], phase: b[lo], score: 1 });
    }
    return { bpm: 60 / wins[0].period, wins, WH };
  }
  function gridDelta(grid, t) {
    const w = grid.wins[Math.max(0, Math.min(grid.wins.length - 1, Math.round((t - grid.WH) / grid.WH)))];
    const s = w.period / 4, d = t - w.phase;
    return { d: Math.round(d / s) * s - d, s };
  }
  function vocalOnsets(f0, target, amp, segStarts) {
    const nF = f0.length, on = [], MINL = 10;
    // 音程が安定して検出されるのは母音の途中からなので、音量の立ち上がりまでさかのぼる
    const refine = t0 => {
      if (!amp) return t0;
      const i0 = Math.max(0, Math.round((t0 - 0.09) / ENV_DT)), i1 = Math.min(amp.length - 1, Math.round((t0 + 0.02) / ENV_DT));
      const j0 = Math.min(amp.length - 1, Math.round((t0 + 0.02) / ENV_DT)), j1 = Math.min(amp.length - 1, Math.round((t0 + 0.12) / ENV_DT));
      const lv = []; for (let i = j0; i <= j1; i++) lv.push(amp[i]);
      if (!lv.length) return t0;
      lv.sort((a, b) => a - b);
      const th = 0.3 * lv[lv.length >> 1];
      let im = i0; for (let i = i0; i <= i1; i++) if (amp[i] < amp[im]) im = i;
      if (amp[im] >= th) return t0;
      for (let i = im + 1; i <= i1 + 4 && i < amp.length; i++) if (amp[i] >= th) {
        const fr = (th - amp[i - 1]) / Math.max(1e-12, amp[i] - amp[i - 1]);
        return (i - 1 + fr + 0.5) * ENV_DT;
      }
      return t0;
    };
    let a = -1, lastEnd = -100;
    for (let k = 0; k <= nF; k++) {
      const v = k < nF && f0[k] > 0;
      if (v && a < 0) a = k;
      if (v || a < 0) continue;
      const run = { a, b: k, gapBefore: (a - lastEnd) * HOP_SEC, on: [] };
      if (k - a >= MINL) {
        run.on.push(Math.max(0, refine(a * HOP_SEC)));
        if (segStarts) { // ノートの区切り(言葉の区切り・音の変わり目)をそのまま歌い出しにする
          for (const ks of segStarts) if (ks > a + 3 && ks < k - 2) { const t = Math.max(0, refine(ks * HOP_SEC)); if (t - run.on[run.on.length - 1] > 0.04) run.on.push(t); }
        } else if (target) {
          let p = a;
          while (p < k) {
            let q = p + 1;
            while (q < k && target[q] === target[p]) q++;
            if (p > a && q - p >= MINL) {
              let pp = p - 1; while (pp > a && target[pp - 1] === target[p - 1]) pp--;
              if (p - pp >= MINL) run.on.push(p * HOP_SEC);
            }
            p = q;
          }
        }
      }
      on.push(run); lastEnd = k; a = -1;
    }
    return on;
  }
  // 歌全体のズレを推定する。お手本の歌があれば、その歌い出しと重ね合わせて決める(いちばん正確)。
  // なければ、歌い出しの8分音符の拍からのズレの平均(円周平均)を出し、どの拍に合わせるかはカラオケのアタックと「遅れの方が多い」ことから決める
  function autoOffset(f0, instEnv, grid, amp, segStarts, refOnsets) {
    const runs = vocalOnsets(f0, null, amp, segStarts || null), on = [];
    runs.forEach(r => r.on.forEach(t => on.push(t)));
    const cnt = on.length;
    if (cnt < 8) return { offsetSec: 0, conf: 0, count: cnt };
    if (refOnsets && refOnsets.length >= 8) { // お手本とのズレの分布の山
      const B = 0.003, W = 0.6, NB = Math.round(2 * W / B) + 1, h = new Float64Array(NB);
      let j0 = 0;
      for (const t of on) {
        while (j0 < refOnsets.length && refOnsets[j0] < t - W) j0++;
        for (let j = j0; j < refOnsets.length && refOnsets[j] <= t + W; j++) { const d = refOnsets[j] - t, b = Math.round((d + W) / B); for (let q = -5; q <= 5; q++) if (b + q >= 0 && b + q < NB) h[b + q] += Math.exp(-q * q / 8); }
      }
      let bi = 0; for (let i = 1; i < NB; i++) if (h[i] > h[bi]) bi = i;
      let tot = 0; for (let i = 0; i < NB; i++) tot += h[i];
      let near = 0; for (let i = Math.max(0, bi - 8); i <= Math.min(NB - 1, bi + 8); i++) near += h[i];
      return { offsetSec: bi * B - W, conf: Math.min(1, 3 * near / tot), count: cnt, byRef: true };
    }
    const BIAS = 0.008; // 音量の立ち上がりで見る歌い出しは、実際よりわずかに遅く出る
    const w8 = t => grid.wins[Math.max(0, Math.min(grid.wins.length - 1, Math.round((t - grid.WH) / grid.WH)))];
    let zr = 0, zi = 0, s8sum = 0;
    for (const t0 of on) { const t = t0 - BIAS, w = w8(t), s8 = w.period / 2, d = t - w.phase, dev = d - Math.round(d / s8) * s8; zr += Math.cos(2 * Math.PI * dev / s8); zi += Math.sin(2 * Math.PI * dev / s8); s8sum += s8; }
    const s8 = s8sum / cnt, late = Math.atan2(zi, zr) / (2 * Math.PI) * s8, R = Math.hypot(zr, zi) / cnt; // late: 8分の拍に対する遅れ(-s8/2〜+s8/2)
    // カラオケのアタックとの重なり
    const n = instEnv.length, sm = new Float32Array(n);
    for (let i = 2; i < n - 2; i++) sm[i] = (instEnv[i - 2] + 2 * instEnv[i - 1] + 3 * instEnv[i] + 2 * instEnv[i + 1] + instEnv[i + 2]) / 9;
    let mean = 0; for (let i = 0; i < n; i++) mean += sm[i]; mean /= n || 1;
    const att = o => { let a = 0; for (const t of on) { const i = Math.round((t - BIAS + o) / ENV_DT); if (i >= 0 && i < n) a += sm[i]; } return a / cnt / (mean || 1); };
    let best = -late, bv = -1e30;
    for (let k = -2; k <= 1; k++) {
      const o = -late + k * s8; // offsetは「歌をずらす量」(遅れている歌はマイナスへ)
      if (o < -0.45 || o > 0.2) continue;
      const val = 0.25 * att(o) - (o < 0 ? 0.45 : 0.9) * Math.abs(o) / 0.3; // 録音は遅れがち。ただし同程度なら小さいズレを選ぶ
      if (val > bv) { bv = val; best = o; }
    }
    return { offsetSec: best, conf: R, count: cnt };
  }
  // リズム補正: 歌い出し/音の変わり目を16分グリッドへ寄せる時間ワープを作る
  function planTiming(f0, target, grid, opts) {
    const nF = f0.length, dur = nF * HOP_SEC, str = opts.strength, off = opts.offsetSec || 0;
    const ain = [0], aout = [0];
    let sum = 0, cnt = 0, total = 0;
    const push = (ti, to, loose) => { // loose: 歌っていない(すき間の)区間なので、大きく縮め/伸ばしてもよい
      const li = ain[ain.length - 1], lo = aout[aout.length - 1];
      if (ti <= li + 0.004) return;
      const di = ti - li, a = loose === 'user' ? 0.35 : loose ? 0.15 : 0.8, b = loose === 'user' ? 3 : loose ? 4 : 1.25; // 声の部分は 0.8〜1.25倍まで(言葉の長さが不自然にならないように)
      ain.push(ti); aout.push(Math.max(lo + a * di, Math.min(lo + b * di, to)));
    };
    const userDt = opts.noteDt || [], refOn = opts.refOnsets || null, RW = opts.refWin || 0.22; let lo0 = 0;
    if (grid && (str > 0 || userDt.length)) {
      // 伸ばしている母音の部分で伸縮を吸収する。長い休符の前後だけは休符側で吸収して、
      // 歌い出しの子音と語尾はそのまま(同じだけ)動かす
      const runs = vocalOnsets(f0, target, opts.amp, opts.segStarts || null), GAP = 0.15;
      // フレーズ(0.35秒以上の休みで区切る)ごとに、全体がどれだけ遅れて/早く歌っているか(8分音符の拍に対して)を求める
      const s8 = t => gridDelta(grid, t).s * 2; // 8分音符の長さ
      const circ = ts => { let zr = 0, zi = 0, c = 0; for (const t of ts) { const ph = 2 * Math.PI * gridDelta8(t) / (s8(t) / 2 * 2); zr += Math.cos(ph); zi += Math.sin(ph); c++; } return c ? { mu: Math.atan2(zi, zr) / (2 * Math.PI), R: Math.hypot(zr, zi) / c, n: c } : { mu: 0, R: 0, n: 0 }; };
      function gridDelta8(t) { const w = grid.wins[Math.max(0, Math.min(grid.wins.length - 1, Math.round((t - grid.WH) / grid.WH)))], s2 = w.period / 2, d = t - w.phase; return d - Math.round(d / s2) * s2; } // 8分の拍からのズレ(+は遅れ)
      const allOn = []; runs.forEach(r => r.on.forEach(t => allOn.push(t + off)));
      const gl = circ(allOn), sysOf = new Map();
      let ph = [], phStart = 0;
      const flush = () => { if (!ph.length) return; const cc = circ(ph.map(t => t + off)); const per = s8(ph[0] + off); let mu = (cc.n >= 3 && cc.R > 0.35 ? cc.mu : gl.mu) * per; if (mu < -0.3 * per) mu += per; /* 大きく早く見えるときは、8分音符1つ分おくれていると考える(遅れの方がずっと多い) */ ph.forEach(t => sysOf.set(t, Math.abs(mu) > 0.006 ? mu : 0)); ph = []; };
      for (let ri = 0; ri < runs.length; ri++) { const r = runs[ri]; if (ri && r.a * HOP_SEC - runs[ri - 1].b * HOP_SEC > 0.35) flush(); r.on.forEach(t => ph.push(t)); }
      flush();
      let prevEnd = 0;
      // 1回目: 歌い出しごとにずらす量を決める / 2回目: 前の音の終わりが次の音にぶつからないようにして並べる
      const plan2 = runs.map(() => []);
      for (let ri = 0; ri < runs.length; ri++) {
        const r = runs[ri];
        let sh = null;
        for (let i = 0; i < r.on.length; i++) {
          const t = r.on[i], sys = sysOf.get(t) || 0, g = gridDelta(grid, t + off - sys), tol = Math.min(0.42 * g.s, 0.08);
          const ue = userDt.find(e => t >= e.t0 - 0.06 && t < e.t1); // 手で動かしたノート
          if (!ue && refOn) { // お手本の歌い出しが近くにあれば、そこに合わせる
            let best = null;
            if (opts.refMap) { // 時間合わせの結果: この歌い出しはお手本のどこか → そこにいちばん近いお手本の歌い出しへ
              const kf = Math.max(0, Math.min(opts.refMap.length - 1, Math.round(t / HOP_SEC))), tg = opts.refMap[kf] * HOP_SEC;
              let bo = null; for (let q = 0; q < refOn.length; q++) { const d = refOn[q] - tg; if (Math.abs(d) < 0.12 && (bo == null || Math.abs(d) < Math.abs(bo - tg))) bo = refOn[q]; }
              if (opts.refNotes) { // 1音ずつ: 同じ音で始まるお手本の歌い出し(±0.35秒)を探す → フレーズの途中で走った/もたった音も合わせる
                const kk = Math.min(target.length - 1, kf + 4), n = target[kk];
                if (n === n) { let bm = null; for (let q = 0; q < refOn.length; q++) { const d = refOn[q] - tg; if (Math.abs(d) > 0.35) continue; const j = Math.min(opts.refNotes.length - 1, Math.round(refOn[q] / HOP_SEC) + 4), rn = opts.refNotes[j]; if (!(rn === rn)) continue; let dd = (n - rn) % 12; if (dd > 6) dd -= 12; if (dd < -6) dd += 12; if (Math.abs(dd) < 0.5 && (bm == null || Math.abs(d) < Math.abs(bm - tg))) bm = refOn[q]; }
                  if (bm != null) bo = bm; }
              }
              const cand = (bo != null ? bo : tg) - (t + off); if (Math.abs(cand) < 1.5) best = cand;
            } else for (let q = lo0; q < refOn.length; q++) { const d = refOn[q] - (t + off); if (d < -RW) { lo0 = q; continue; } if (d > RW) break; if (best == null || Math.abs(d) < Math.abs(best)) best = d; }
            total++;
            if (best != null) { sum += Math.abs(best); cnt++; sh = best * str; plan2[ri].push([t, sh, i, null]); continue; }
            total--;
          }
          total++;
          // まとめての遅れ(-sys) + 一番近い16分音符へ
          const d = -sys + (Math.abs(g.d) <= tol ? g.d : 0);
          if (ue) sh = ue.dt + d * str;
          else { sum += Math.abs(d); cnt++; sh = d * str; }
          plan2[ri].push([t, sh, i, ue || null]);
        }
      }
      for (let ri = 0; ri < runs.length; ri++) {
        const r = runs[ri], pl = plan2[ri];
        for (let pi = 0; pi < pl.length; pi++) {
          const [t, sh, i, ue] = pl[pi];
          if (i === 0 && t - prevEnd >= GAP) { const pre = Math.max(t - 0.08, (prevEnd + t) / 2); push(pre, pre + sh, true); }
          push(t, t + sh);
          if (ue && ue.dl) { const tend = Math.min(ue.t1, pi + 1 < pl.length ? pl[pi + 1][0] - 0.01 : ue.t1) - 0.01; if (tend > t + 0.03) push(tend, tend + sh + ue.dl, 'user'); } // ノートの長さ(手で伸ばした/縮めた)
        }
        const te = r.b * HOP_SEC, gapAfter = ri + 1 < runs.length ? runs[ri + 1].a * HOP_SEC - te : 1e9;
        if (pl.length && gapAfter >= GAP) {
          let she = pl[pl.length - 1][1];
          const nx = ri + 1 < runs.length ? plan2[ri + 1][0] : null;
          if (nx) { const lim = nx[0] + nx[1] - 0.05 - (te + 0.03); if (she > lim) she = Math.max(lim, pl[pl.length - 1][1] - 0.4); } // 次の音が早まったら、この音の終わりも早める(重ならない)
          push(te + 0.03, te + 0.03 + she);
        }
        prevEnd = te;
      }
    }
    push(dur + 1, dur + 1);
    const tOut = new Float32Array(nF);
    let j = 0;
    for (let k = 0; k < nF; k++) {
      const t = k * HOP_SEC;
      while (j + 2 < ain.length && t >= ain[j + 1]) j++;
      tOut[k] = aout[j] + (t - ain[j]) * (aout[j + 1] - aout[j]) / (ain[j + 1] - ain[j]);
    }
    const before = cnt ? sum / cnt * 1000 : 0;
    return { warp: ain.length > 2 ? { ain, aout } : null, tOut, msBefore: before, msAfter: before * (1 - str), matched: cnt, total };
  }


  // ---------- ノイズ除去 ----------
  // 歌っていない静かな部分からノイズの「指紋」を取り、その成分だけを全体から引く(スペクトルゲート)
  function denoise(x, amount) {
    const N = 2048, H = 512, n = x.length, nFr = Math.floor((n - N) / H) + 1;
    const info = { ok: false, snrDb: null };
    if (nFr < 50) return { y: x, info };
    const ps = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) ps[i + 1] = ps[i] + x[i] * x[i];
    const en = new Float64Array(nFr), valid = [];
    for (let f = 0; f < nFr; f++) { en[f] = ps[f * H + N] - ps[f * H]; if (en[f] > 1e-9 * N) valid.push(en[f]); }
    if (valid.length < 50) return { y: x, info };
    valid.sort((a, b) => a - b);
    const p10 = valid[Math.floor(valid.length * 0.1)], p90 = valid[Math.floor(valid.length * 0.9)];
    info.snrDb = 10 * Math.log10(p90 / p10);
    if (amount <= 0) return { y: x, info };
    const quiet = p10 < 0.02 * p90; // 歌の合間の静かな所があるか。なければ「各帯域でいちばん小さかった時の音」をノイズとみなす(ずっと歌っている・切り詰めた録音でも効く)
    info.mode = quiet ? 'gaps' : 'minstat';
    const fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), win = new Float64Array(N), K = N / 2 + 1;
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
    const Np = new Float64Array(K);
    let cnt = 0;
    const step = Math.max(1, Math.floor(valid.length * 0.1 / 400));
    for (let f = 0, seen = 0; f < nFr; f++) {
      if (!(en[f] > 1e-9 * N && en[f] <= p10)) continue;
      if (seen++ % step) continue;
      for (let i = 0; i < N; i++) { re[i] = x[f * H + i] * win[i]; im[i] = 0; }
      fft.forward(re, im);
      for (let k = 0; k < K; k++) Np[k] += re[k] * re[k] + im[k] * im[k];
      cnt++;
    }
    for (let k = 0; k < K; k++) Np[k] /= Math.max(1, cnt);
    if (!quiet) { // 最小値統計: 帯域ごとに、全フレームの中で下から5%の大きさ
      const fs = [], stepF = Math.max(1, Math.floor(nFr / 600)); for (let f = 0; f < nFr; f += stepF) if (en[f] > 1e-9 * N) fs.push(f);
      const P2 = fs.map(f => { for (let i = 0; i < N; i++) { re[i] = x[f * H + i] * win[i]; im[i] = 0; } fft.forward(re, im); const a = new Float32Array(K); for (let k = 0; k < K; k++) a[k] = re[k] * re[k] + im[k] * im[k]; return a; });
      const col = new Float32Array(P2.length); let tot = 0, nz = 0;
      for (let k = 0; k < K; k++) { for (let j = 0; j < P2.length; j++) col[j] = P2[j][k]; col.sort(); Np[k] = col[Math.floor(col.length * 0.05)] * 10; /* ノイズの強さはばらつくので下から5%の値は平均の約1/20。声も混ざるぶん控えめに×10 */ nz += Np[k]; let m = 0; for (let j = 0; j < col.length; j++) m += col[j]; tot += m / col.length; }
      info.snrDb = 10 * Math.log10(tot / (nz + 1e-20));
      if (info.snrDb > 35) { info.ok = false; info.clean = true; return { y: x, info }; } // ノイズがほとんど無い
      amount *= 0.6; // 声と重なっている所なので控えめに
    }
    { const sn = info.snrDb == null ? 30 : info.snrDb, k = Math.max(0, Math.min(1, (40 - sn) / 22)); info.applied = amount * k; amount *= k; if (amount < 0.03) { info.ok = false; info.clean = true; return { y: x, info }; } } // ノイズの量に合わせて強さを決める(静かな録音はさわらない)
    const beta = 1.2 + 2 * amount, gmin = Math.pow(10, -(8 + 22 * amount) / 20);
    const y = new Float32Array(n), ws = new Float32Array(n), g = new Float64Array(K), gp = new Float64Array(K).fill(1), g2 = new Float64Array(K);
    // 2フレームを1回のFFTで(実部=前のフレーム、虚部=次のフレーム)
    const gA = new Float64Array(K), gB = new Float64Array(K), Ar = new Float64Array(K), Ai = new Float64Array(K), Br = new Float64Array(K), Bi = new Float64Array(K);
    const gain = (Pr, Pi, gout) => {
      for (let k = 0; k < K; k++) { const P = Pr[k] * Pr[k] + Pi[k] * Pi[k]; g[k] = P > 0 ? Math.max(gmin, 1 - beta * Np[k] / P) : gmin; }
      for (let k = 0; k < K; k++) {
        let v = (g[Math.max(0, k - 1)] + 2 * g[k] + g[Math.min(K - 1, k + 1)]) / 4;
        if (v < gp[k]) v = 0.5 * v + 0.5 * gp[k]; // 閉じるときはゆっくり(シュワシュワ音を防ぐ)
        gout[k] = v; gp[k] = v;
      }
    };
    for (let f = 0; f < nFr; f += 2) {
      const s = f * H, two = f + 1 < nFr, s2 = s + H;
      for (let i = 0; i < N; i++) { re[i] = x[s + i] * win[i]; im[i] = two ? x[s2 + i] * win[i] : 0; }
      fft.forward(re, im);
      for (let k = 0; k < K; k++) { const j = (N - k) % N; Ar[k] = 0.5 * (re[k] + re[j]); Ai[k] = 0.5 * (im[k] - im[j]); Br[k] = 0.5 * (im[k] + im[j]); Bi[k] = -0.5 * (re[k] - re[j]); }
      gain(Ar, Ai, gA); if (two) gain(Br, Bi, gB); else gB.fill(0);
      // Y = Ya + i*Yb (どちらもエルミート) → 逆変換で 実部=前, 虚部=次
      for (let k = 0; k < K; k++) {
        const ar = Ar[k] * gA[k], ai = Ai[k] * gA[k], br = Br[k] * gB[k], bi = Bi[k] * gB[k];
        re[k] = ar - bi; im[k] = ai + br;
        if (k > 0 && k < N / 2) { re[N - k] = ar + bi; im[N - k] = -ai + br; }
      }
      fft.forward(im, re); // 逆変換(実部・虚部を入れ替えて順変換)
      for (let i = 0; i < N; i++) { y[s + i] += re[i] / N * win[i]; ws[s + i] += win[i] * win[i]; }
      if (two) for (let i = 0; i < N; i++) { y[s2 + i] += im[i] / N * win[i]; ws[s2 + i] += win[i] * win[i]; }
    }
    for (let i = 0; i < n; i++) y[i] = ws[i] > 0.05 ? y[i] / ws[i] : x[i];
    info.ok = true;
    return { y, info };
  }
  // 電源ハム(50/60Hzのブーン)を検出。静かな区間で、ハムの周波数だけ飛び抜けて強いかを見る
  function detectHum(x) {
    const B = Math.round(0.5 * SR), nB = Math.floor(x.length / B);
    if (nB < 8) return 0;
    const en = [];
    for (let b = 0; b < nB; b++) { let e = 0; for (let i = b * B; i < (b + 1) * B; i += 4) e += x[i] * x[i]; if (e > 1e-9 * B) en.push([e, b]); }
    if (en.length < 8) return 0;
    en.sort((a, c) => a[0] - c[0]);
    const use = en.slice(0, Math.max(4, Math.min(24, Math.floor(en.length / 4))));
    const pw = fr => { // 静かな区間ごとの強さの中央値(一部の区間にだけ入った歌声に引っぱられないように)
      const w = 2 * Math.PI * fr / SR, c = 2 * Math.cos(w), vals = [];
      for (const [, b] of use) {
        let s1 = 0, s2 = 0;
        for (let i = 0; i < B; i++) { const v = x[b * B + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / B)), s0 = v + c * s1 - s2; s2 = s1; s1 = s0; }
        vals.push(s1 * s1 + s2 * s2 - c * s1 * s2);
      }
      vals.sort((p, q) => p - q);
      return vals[vals.length >> 1];
    };
    const refs = [40, 70, 80, 90, 110, 130, 140, 160, 170].map(pw).sort((a, c) => a - c), ref = refs[4] + 1e-20;
    const strong = f0 => [1, 2, 3].filter(h => pw(f0 * h) > 8 * ref).length;
    const s50 = strong(50), s60 = strong(60);
    if (s50 >= 2 && s50 > s60) return 50;
    if (s60 >= 2 && s60 > s50) return 60;
    return 0;
  }

  // 声質の測定: 歌っている部分の平均スペクトルから、こもり/抜け/空気感の帯域が基準(400-900Hz)に対して何dBかを出す
  function toneProfile(x) {
    const N = 2048, nFr = Math.floor(x.length / N);
    if (nFr < 20) return null;
    const en = new Float64Array(nFr);
    for (let f = 0; f < nFr; f++) { let e = 0; for (let i = f * N; i < (f + 1) * N; i += 2) e += x[i] * x[i]; en[f] = e; }
    const so = Float64Array.from(en).sort(), thr = so[Math.floor(nFr * 0.9)] * 0.03;
    const fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), acc = new Float64Array(N / 2);
    let c = 0;
    for (let f = 0; f < nFr; f++) {
      if (en[f] < thr || en[f] <= 0) continue;
      for (let i = 0; i < N; i++) { re[i] = x[f * N + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)); im[i] = 0; }
      fft.forward(re, im);
      for (let k = 0; k < N / 2; k++) acc[k] += re[k] * re[k] + im[k] * im[k];
      c++;
    }
    if (c < 10) return null;
    const band = (f0, f1) => { const k0 = Math.round(f0 * N / SR), k1 = Math.round(f1 * N / SR); let s = 0; for (let k = k0; k < k1; k++) s += acc[k]; return 10 * Math.log10(s / (k1 - k0) + 1e-30); };
    const ref = band(400, 900);
    return { low: band(180, 350) - ref, pres: band(2500, 5000) - ref, air: band(7000, 14000) - ref };
  }

  // ---------- 声の下ごしらえ(ノイズ除去・音量そろえ・ディエッサー・ノイズゲート) ----------
  // ブレス(息を吸う音)を見つけて小さくする。voiced: 出力の時間でフレーム(HOP)ごとに 声あり=1
  function breathDb(a) { return a <= 0.8 ? a * 22 : 17.6 + (Math.min(1, a) - 0.8) / 0.2 * 27.4; }
  function breathCtl(y, voiced, amount) {
    const n = y.length, nF = Math.floor(n / HOP), info = { count: 0, sec: 0, spans: [] };
    if (!(amount > 0) || nF < 20) return info;
    const hf = new Float32Array(y), hp = biquad('hp', 4500, 0.707); filt(hf, hp); filt(hf, hp);
    const lf = new Float32Array(y), bp1 = biquad('hp', 250, 0.707); filt(lf, bp1);
    const e = new Float32Array(nF), eh = new Float32Array(nF);
    for (let f = 0; f < nF; f++) { let a = 0, b = 0; for (let i = f * HOP; i < (f + 1) * HOP; i++) { a += lf[i] * lf[i]; b += hf[i] * hf[i]; } e[f] = a / HOP; eh[f] = b / HOP; }
    const isV = f => voiced[Math.min(voiced.length - 1, f)] === 1;
    // 静かな所の大きさ(ノイズ)と、歌っている所の大きさ
    const un = [], vo = []; for (let f = 0; f < nF; f++) (isV(f) ? vo : un).push(e[f]);
    un.sort((p, q) => p - q); vo.sort((p, q) => p - q);
    if (vo.length < 10) return info;
    const floor = un.length ? un[Math.floor(un.length * 0.2)] : 0, vlev = vo[Math.floor(vo.length * 0.5)];
    const PRE = Math.round(0.07 / HOP_SEC), POST = Math.round(0.05 / HOP_SEC), MINL = Math.round(0.1 / HOP_SEC);
    const cand = new Uint8Array(nF);
    for (let f = 0; f < nF; f++) {
      if (isV(f)) continue;
      let near = false; for (let d = 1; d <= PRE && !near; d++) if (f + d < nF && isV(f + d)) near = true; for (let d = 1; d <= POST && !near; d++) if (f - d >= 0 && isV(f - d)) near = true;
      if (near) continue; // 歌い出し直前の子音・語尾は守る
      const hfShare = eh[f] / (e[f] + 1e-12);
      if (e[f] > Math.max(floor * 4, vlev * 0.0015) && e[f] < vlev * 0.9 && hfShare < 0.5) cand[f] = 1; // サ行(高い音が主役)は残す
    }
    const gain = new Float32Array(nF).fill(1), gdb = Math.pow(10, -breathDb(amount) / 20); // 0.8までは自然に小さく(最大-17.6dB)、それより上は「消す」(1.0で-45dB)
    for (let f = 0; f < nF;) {
      if (!cand[f]) { f++; continue; }
      let g2 = f; while (g2 < nF && (cand[g2] || (g2 + 1 < nF && cand[g2 + 1]))) g2++;
      if (g2 - f >= MINL) { // 息の出だし・終わり(小さい所)も含める
        let a0 = f, b0 = g2; const ok = k => k >= 0 && k < nF && !isV(k) && e[k] > floor * 1.5 && eh[k] / (e[k] + 1e-12) < 0.5;
        for (let d = 0; d < 8 && ok(a0 - 1); d++) a0--; for (let d = 0; d < 8 && ok(b0); d++) b0++;
        let pv = false; for (let d = 1; d <= PRE; d++) if (b0 - 1 + d < nF && isV(b0 - 1 + d)) pv = true; if (pv) b0 = Math.max(g2, b0 - PRE);
        let be = 0; for (let k = f; k < g2; k++) be += e[k]; be /= Math.max(1, g2 - f); // この息の大きさ(歌の大きさとくらべて)
        const rel = 10 * Math.log10(be / (vlev + 1e-12) + 1e-12), sc = amount >= 0.95 ? 1 : Math.max(0.5, Math.min(1.5, 1 + (rel + 20) / 20)); // −30dB の息は半分、−10dB の息は1.5倍
        const gk = amount >= 0.95 ? gdb : Math.pow(10, -breathDb(amount) * sc / 20); info.dbs = info.dbs || []; if (info.dbs.length < 200) info.dbs.push(Math.round(breathDb(amount) * sc * 10) / 10);
        for (let k = a0; k < b0; k++) gain[k] = gk; info.count++; info.sec += (g2 - f) * HOP_SEC; info.db = Math.round(breathDb(amount)); if (info.spans.length < 200) info.spans.push([f * HOP_SEC, g2 * HOP_SEC]); }
      f = g2;
    }
    // なめらかにつなぐ(15ms)
    const a = Math.exp(-1 / (0.015 * SR)); let gs = 1;
    for (let i = 0; i < n; i++) { const tg = gain[Math.min(nF - 1, Math.floor(i / HOP))]; gs = a * gs + (1 - a) * tg; y[i] *= gs; }
    return info;
  }
  // 耳につく響き(マイク・部屋のクセで出る、細い山の帯域)を、出ている瞬間だけ自動で抑える
  function tameResonance(x, amount) {
    if (!(amount > 0)) return { y: x, info: { avgDb: 0 } };
    const N = 2048, H = 512, n = x.length, nb = N / 2 + 1, fft = new FFT(N), win = new Float64Array(N), y = new Float32Array(n), ws = new Float32Array(n);
    for (let i = 0; i < N; i++) win[i] = Math.sqrt(0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
    const re = new Float64Array(N), im = new Float64Array(N), mag = new Float64Array(nb), e1 = new Float64Array(nb), e2 = new Float64Array(nb), gr = new Float64Array(nb), gs = new Float64Array(nb), ps = new Float64Array(nb + 1), g = new Float64Array(nb).fill(1);
    const bHz = SR / N, kLo = Math.round(1200 / bHz), kHi = Math.min(nb - 1, Math.round(11000 / bHz)), maxRed = 3 + 6 * amount, thr = 1.8 - amount, RW = Math.round(550 / bHz); // RW: ±550Hz(v3.4) より細い山だけを「響きのクセ」とみなす(声のフォルマントはもっと太い)
    let sumRed = 0, cntRed = 0;
    const excess = () => { // 今のフレームの「細い山」の出っぱり(dB)を gr に
      for (let k = 0; k < nb; k++) mag[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-20);
      for (let k = 0; k < nb; k++) { let m = -400; for (let j = Math.max(0, k - 14); j <= Math.min(nb - 1, k + 14); j++) if (mag[j] > m) m = mag[j]; e1[k] = m; } /* v3.4: ±300Hz。高い声(倍音の間隔が広い)で倍音そのものを「響きの山」とまちがえて、倍音ごとに音量がゆれていた */
      for (let k = 0; k < nb; k++) { let a = 0, c = 0; for (let j = Math.max(0, k - 5); j <= Math.min(nb - 1, k + 5); j++) { a += e1[j]; c++; } gs[k] = a / c; }
      ps[0] = 0; for (let k = 0; k < nb; k++) ps[k + 1] = ps[k] + gs[k];
      for (let k = 0; k < nb; k++) { const a = Math.max(0, k - RW), b = Math.min(nb - 1, k + RW); e2[k] = gs[k] - (ps[b + 1] - ps[a]) / (b - a + 1); }
    };
    // 1回目(4フレームに1回): 曲を通していつも同じ所に出る山 = マイク・部屋のクセ。母音で動くフォルマントは平均すると目立たない
    const LT = new Float64Array(nb); let ltc = 0;
    for (let s0 = 0; s0 + N < n; s0 += H * 4) { let en = 0; for (let i = 0; i < N; i++) { const v = x[s0 + i]; re[i] = v * win[i]; im[i] = 0; en += v * v; } if (en / N < 1e-5) continue; fft.forward(re, im); excess(); for (let k = 0; k < nb; k++) LT[k] += e2[k]; ltc++; }
    const wLT = new Float64Array(nb); for (let k = 0; k < nb; k++) wLT[k] = ltc ? Math.max(0.35, Math.min(1, (LT[k] / ltc - 0.5) / 2)) : 1;
    for (let s0 = -N + H; s0 < n; s0 += H) {
      let en = 0;
      for (let i = 0; i < N; i++) { const j = s0 + i, v = j >= 0 && j < n ? x[j] : 0; re[i] = v * win[i]; im[i] = 0; en += v * v; }
      fft.forward(re, im);
      if (en / N < 1e-7) { for (let k = 0; k < nb; k++) g[k] += (1 - g[k]) * 0.3; }
      else {
        excess(); // 倍音の山をなぞった線(声の形)が、まわり±550Hzの平均より出ている所 = 細い響きの山
        for (let k = 0; k < nb; k++) { const ex = k < kLo || k > kHi ? 0 : (e2[k] - thr) * wLT[k], fade = Math.min(1, (k - kLo) / 20, (kHi - k) / 40); gr[k] = ex > 0 ? Math.min(maxRed, ex * 1.8) * Math.max(0, fade) : 0; }
        for (let k = 0; k < nb; k++) { const tg = Math.pow(10, -gr[k] / 20); g[k] += (tg - g[k]) * (tg < g[k] ? 0.25 : 0.08); } /* 速く動かすと「シュワシュワ」した金属っぽい音(ミュージカルノイズ)になるので、ゆっくり */
        for (let k = kLo; k <= kHi; k++) { sumRed += -20 * Math.log10(g[k]); cntRed++; }
      }
      for (let k = 0; k < nb; k++) { re[k] *= g[k]; im[k] *= g[k]; if (k > 0 && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; } }
      fft.forward(im, re); // 逆変換
      for (let i = 0; i < N; i++) { const j = s0 + i; if (j >= 0 && j < n) { y[j] += re[i] / N * win[i]; ws[j] += win[i] * win[i]; } }
    }
    for (let i = 0; i < n; i++) y[i] = ws[i] > 1e-6 ? y[i] / ws[i] : 0;
    return { y, info: { avgDb: cntRed ? sumRed / cntRed : 0 } };
  }
  // 口の「プチッ」(クリック)と、マイクに息が当たる「ボフッ」(ポップ)を、見つけた所だけ軽くする
  function declick(y) {
    const n = y.length, d2 = new Float32Array(n); let cnt = 0;
    for (let i = 1; i < n - 1; i++) d2[i] = Math.abs(y[i + 1] - 2 * y[i] + y[i - 1]);
    const B = 1024, loc = new Float32Array(Math.ceil(n / B)); for (let b = 0; b < loc.length; b++) { const a = []; for (let i = b * B; i < Math.min(n, (b + 1) * B); i += 4) a.push(d2[i]); a.sort((p, q) => p - q); loc[b] = a[Math.floor(a.length * 0.9)] || 0; }
    for (let i = 4; i < n - 70; i++) {
      const th = 10 * Math.max(loc[Math.floor(i / B)], 1e-5); if (d2[i] < th) continue;
      let j = i; while (j < n - 4 && j - i < 66 && d2[j] > th * 0.3) j++; // 1.5ms以内の、とがった短い所だけ
      if (j - i >= 66) { i = j; continue; }
      const a = i - 3, b = j + 3, ya = y[a], yb = y[b], sa = y[a] - y[a - 1], sb = y[b + 1] - y[b], L2 = b - a; // 前後をなめらかにつなぐ(3次)
      for (let k = a + 1; k < b; k++) { const t = (k - a) / L2, h00 = 2 * t ** 3 - 3 * t * t + 1, h10 = t ** 3 - 2 * t * t + t, h01 = -2 * t ** 3 + 3 * t * t, h11 = t ** 3 - t * t; y[k] = h00 * ya + h10 * sa * L2 + h01 * yb + h11 * sb * L2; }
      cnt++; i = b;
    }
    return cnt;
  }
  function depop(y) {
    const n = y.length, lf = new Float32Array(y); filt(lf, biquad('lp', 120, 0.707)); lf.reverse(); filt(lf, biquad('lp', 120, 0.707)); lf.reverse(); // 前後から2回かけて、ずれ(位相)の無い低音だけを取り出す
    const W = 256, nb = Math.floor(n / W), el = new Float32Array(nb), ea = new Float32Array(nb);
    for (let b = 0; b < nb; b++) { let a = 0, c = 0; for (let i = b * W; i < (b + 1) * W; i++) { a += lf[i] * lf[i]; c += y[i] * y[i]; } el[b] = a / W; ea[b] = c / W; }
    const so = Array.from(el).filter(v => v > 0).sort((p, q) => p - q), bg = so[Math.floor(so.length * 0.6)] || 0; let cnt = 0;
    const g = new Float32Array(nb);
    for (let b = 1; b < nb - 1; b++) if (el[b] > bg * 30 && el[b] > 0.5 * ea[b] && el[b] > 1e-5) { g[b] = 1; } // 低音だけが急に大きい = ポップ
    for (let b = 0; b < nb; b++) if (g[b] && !g[b - 1]) cnt++;
    const sm = new Float32Array(nb); for (let b = 0; b < nb; b++) { let m = 0; for (let k = -3; k <= 3; k++) if (g[b + k]) m = Math.max(m, 1 - Math.abs(k) / 4); sm[b] = m; }
    for (let i = 0; i < nb * W; i++) { const b = Math.floor(i / W), f = (i % W) / W, w = sm[b] * (1 - f) + (sm[Math.min(nb - 1, b + 1)] || 0) * f; if (w > 0) y[i] -= 0.85 * w * lf[i]; }
    return cnt;
  }
  // ダイナミックEQ: 決めた帯域を、問題が出ている瞬間だけ下げる(いつも削るのではない)
  // 帯域の音は「前後から2回かけたバンドパス」(位相のずれ無し)で取り出し、y から (1-g) 倍を引く = その帯域だけ g 倍になる
  function zpBand(x, f, q) { const b = new Float32Array(x); filt(b, biquad('bp', f, q)); b.reverse(); filt(b, biquad('bp', f, q)); b.reverse(); return b; }
  function envOf(x, atk, rel) { const n = x.length, e = new Float32Array(n), aA = Math.exp(-1 / (atk * SR)), aR = Math.exp(-1 / (rel * SR)); let v = 0; for (let i = 0; i < n; i++) { const p = x[i] * x[i]; v = p > v ? aA * v + (1 - aA) * p : aR * v + (1 - aR) * p; e[i] = v; } return e; }
  function dynBand(y, band, ref, opt) { // opt: {over: 中央値より何dB上から, maxCut, atk, rel, needRatio}
    const n = y.length, eb = envOf(band, opt.atk || 0.005, opt.rel || 0.08), er = envOf(ref, 0.01, 0.1);
    const step = 256, rs = []; for (let i = 0; i < n; i += step) if (er[i] > 1e-7) rs.push(10 * Math.log10(eb[i] / er[i] + 1e-12)); rs.sort((a, b) => a - b);
    if (rs.length < 20) return { hits: 0, maxDb: 0 }; const med = rs[rs.length >> 1];
    let hits = 0, maxDb = 0, on = false, g = 1; const aA = Math.exp(-1 / (0.003 * SR)), aR = Math.exp(-1 / (0.06 * SR));
    for (let i = 0; i < n; i++) {
      let cut = 0; if (er[i] > 1e-7 && eb[i] > 1e-9) { const r = 10 * Math.log10(eb[i] / er[i] + 1e-12) - med - opt.over; if (r > 0 && (!opt.needRatio || eb[i] > opt.needRatio * er[i])) cut = Math.min(opt.maxCut, r * (opt.k || 0.7)); }
      const tg = Math.pow(10, -cut / 20); g = tg < g ? aA * g + (1 - aA) * tg : aR * g + (1 - aR) * tg;
      if (g < 0.999) y[i] -= (1 - g) * band[i];
      const gd = -20 * Math.log10(g); if (gd > 1.5) { if (!on) hits++; on = true; if (gd > maxDb) maxDb = gd; } else if (gd < 0.5) on = false;
    }
    return { hits, maxDb: Math.round(maxDb * 10) / 10 };
  }
  function vocalDynEq(y, amt) { // こもり(320Hz)・刺さり(3.5kHz)を、出ている瞬間だけ
    if (!(amt > 0)) return null;
    const mid = zpBand(y, 1600, 0.5), mud = zpBand(y, 320, 1.1), harsh = zpBand(y, 3500, 1.4);
    const a = dynBand(y, mud, mid, { over: 5, maxCut: 1.5 + 2.5 * amt }), b = dynBand(y, harsh, mid, { over: 3.5, k: 1, maxCut: 1.5 + 3 * amt });
    return { mud: a, harsh: b };
  }
  function deessMulti(y, amt) { // マルチバンドのディエッサー: サ行の帯域(5〜10kHz)だけを、刺さった瞬間だけ下げる(声全体は揺らさない)
    const sib = new Float32Array(y); { const hp = biquad('hp', 5000, 0.707), lp = biquad('lp', 10500, 0.707); filt(sib, hp); filt(sib, lp); sib.reverse(); filt(sib, biquad('hp', 5000, 0.707)); filt(sib, biquad('lp', 10500, 0.707)); sib.reverse(); }
    return dynBand(y, sib, y, { over: 6 - 2 * amt, maxCut: 4 + 8 * amt, atk: 0.001, rel: 0.05, needRatio: 0.12 });
  }
  function prepVocal(x, o) {
    const dn = denoise(x, o.denoise || 0), info = dn.info;
    const y = dn.y === x ? new Float32Array(x) : dn.y, n = y.length;
    info.hum = 0;
    if (o.denoise > 0) {
      info.hum = detectHum(x);
      if (info.hum) for (let h = 1; h <= 4; h++) filt(y, biquad('notch', info.hum * h, 35));
    }
    if (o.reso > 0) { const t2 = tameResonance(y, o.reso); y.set(t2.y); info.reso = t2.info.avgDb; }
    if (o.declick !== false && (o.denoise > 0 || o.declick)) { info.clicks = declick(y); info.pops = depop(y); }
    const r = activeRms(y);
    if (r > 0) { const g = 0.1 / r; for (let i = 0; i < n; i++) y[i] *= g; }
    if (o.phrase > 0) info.phrase = phraseLevel(y, o.phrase); // v4.0: コンプの前に、フレーズごとの大きさをゆるやかにそろえる
    if (o.dyneq > 0) info.dyneq = vocalDynEq(y, o.dyneq);
    if (o.deess > 0) { const d = deessMulti(y, o.deess); info.deess = { hits: d.hits, maxDb: d.maxDb, multi: true }; }
    if (o.gate > 0) {
      const T = -52 + 14 * o.gate, maxRed = 6 + 18 * o.gate;
      const aE = Math.exp(-1 / (0.01 * SR)), aO = Math.exp(-1 / (0.003 * SR)), aC = Math.exp(-1 / (0.15 * SR));
      let env = 0, gr = 0;
      for (let i = 0; i < n; i++) {
        const v = y[i];
        env = aE * env + (1 - aE) * v * v;
        const L = 10 * Math.log10(env + 1e-12), red = L < T ? Math.min(maxRed, T - L) : 0;
        gr = red < gr ? aO * gr + (1 - aO) * red : aC * gr + (1 - aC) * red;
        if (gr > 0.01) y[i] = v * Math.pow(10, -gr / 20);
      }
    }
    return { y, info };
  }


  // v4.0 録音品質の診断(加工はしない。結果は「録音のアドバイス」と処理計画に使う)
  //   level: 歌っている所の大きさ(dBFS) / clip: 音割れの数 / floorDb: 静かな所の大きさ / snr / hum(Hz) / pops: 吹かれ
  //   dark: 明瞭さの目安(2〜5kHz と 200〜800Hz の差)
  function recordingCheck(x, f0) {
    const n = x.length, W = 441, nW = Math.floor(n / W), e = new Float32Array(nW), r = { level: null, clip: 0, floorDb: null, snr: null, hum: 0, pops: 0, dark: null };
    if (nW < 50) return r;
    for (let w = 0; w < nW; w++) { let a = 0; for (let i = w * W; i < (w + 1) * W; i++) a += x[i] * x[i]; e[w] = 10 * Math.log10(a / W + 1e-14); }
    let run = 0; for (let i = 0; i < n; i++) { if (Math.abs(x[i]) >= 0.995) { run++; if (run === 3) r.clip++; } else run = 0; }
    const so = Float32Array.from(e).sort(), top = so[Math.floor(nW * 0.95)], nz = so.filter(v => v > -100); r.level = Math.round(top * 10) / 10; if (nz.length > 20 && nz[Math.floor(nz.length * 0.1)] < top - 25) { r.floorDb = Math.round(nz[Math.floor(nz.length * 0.1)] * 10) / 10; r.snr = Math.round((top - r.floorDb) * 10) / 10; } // 完全な無音(編集で切った所)は数えない
    r.hum = detectHum(x) || 0;
    { const lp = Float32Array.from(x), c = biquad('lp', 120, 0.707); filt(lp, c); filt(lp, c); let k = 0; const WL = 220; for (let w = 0; w + WL < n; w += WL) { let a = 0, b = 0; for (let i = w; i < w + WL; i++) { a += lp[i] * lp[i]; b += x[i] * x[i]; } if (a > 0.6 * b && b / WL > Math.pow(10, (top - 6) / 10)) { if (w - k > SR * 0.1) r.pops++; k = w; } } }
    /* 部屋の響きの長さ(tailMs)は、合成した響きで試したところ歌の切れ目と響きを区別できなかったため、v4.0 では調べない */
    { const N = 4096, fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), acc = new Float64Array(N / 2); let c = 0;
      for (let s0 = 0; s0 + N < n && c < 200; s0 += N * 2) { let a = 0; for (let i = s0; i < s0 + N; i += 4) a += x[i] * x[i]; if (10 * Math.log10(a / (N / 4) + 1e-14) < top - 12) continue; for (let i = 0; i < N; i++) { re[i] = x[s0 + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)); im[i] = 0; } fft.forward(re, im); for (let q = 0; q < N / 2; q++) acc[q] += re[q] * re[q] + im[q] * im[q]; c++; }
      if (c >= 5) { const bin = SR / N, bandE = (a, b) => { let v = 0; for (let q = Math.round(a / bin); q < Math.round(b / bin); q++) v += acc[q]; return v / Math.max(1, Math.round(b / bin) - Math.round(a / bin)); };
        r.dark = Math.round(10 * Math.log10(bandE(2000, 5000) / bandE(200, 800)) * 10) / 10;
        /* 細い山(peaks)は、のばした音の倍音と区別できないため調べない */ } }
    return r;
  }

  // v4.0 自動ハモリの仕上げ
  //  1) 音程の跳び(となりのフレームで0.5半音以上)がある所は、その前後50msを −6dB(機械っぽい音が目立たないように)
  //  2) メインがいつもより小さく歌っているフレーズでは、ハモリも少し下げる(最大 −4dB)
  //  3) take=true のときだけ: フレーズごとに 6〜13ms 遅らせる(別テイク風)。切りかえは声のない所だけ
  function harmPolish(y, shift, f0, tO, take) {
    const n = y.length, q = { jumps: 0, quiet: 0, take: 0 }, g = new Float32Array(Math.ceil(n / HOP) + 2).fill(1), at = k => Math.round((tO ? tO[Math.min(tO.length - 1, k)] : k * HOP_SEC) * SR / HOP);
    for (let k = 1; k < f0.length; k++) if (f0[k] > 0 && f0[k - 1] > 0 && Math.abs(shift[k] - shift[k - 1]) > 0.5) { q.jumps++; const j = at(k); for (let d = -4; d <= 4; d++) if (j + d >= 0 && j + d < g.length) g[j + d] = Math.min(g[j + d], 0.5); }
    const F = HOP, nF = Math.floor(n / F), e = new Float32Array(nF); for (let f = 0; f < nF; f++) { let a = 0; for (let i = f * F; i < (f + 1) * F; i++) a += y[i] * y[i]; e[f] = 10 * Math.log10(a / F + 1e-12); }
    const so = Float32Array.from(e).sort(), top = so[Math.floor(nF * 0.95)] || -60, th = top - 30, gap = Math.ceil(0.25 * SR / F), ph = []; let a0 = -1, last = -1;
    for (let f = 0; f <= nF; f++) { const on = f < nF && e[f] > th; if (on) { if (a0 < 0) a0 = f; last = f; } else if (a0 >= 0 && f - last > gap) { if (last - a0 >= 4) ph.push([a0, last + 1]); a0 = -1; } }
    if (a0 >= 0 && last - a0 >= 4) ph.push([a0, last + 1]);
    if (ph.length >= 3) { const lv = ph.map(([a, b]) => { const v = Array.from(e.subarray(a, b)).sort((p, r) => p - r); return v[Math.floor(v.length * 0.75)]; }), sl = lv.slice().sort((p, r) => p - r), med = sl[sl.length >> 1];
      ph.forEach(([a, b], i) => { const d = lv[i] - med; if (d < -3) { q.quiet++; const gg = Math.pow(10, -Math.min(4, 0.5 * (-d)) / 20); for (let f = a; f < b; f++) g[f] = Math.min(g[f], gg); } }); }
    const gs = Float32Array.from(g); for (let j = 0; j < g.length; j++) { let a = 0, c = 0; for (let r = j - 3; r <= j + 3; r++) if (r >= 0 && r < g.length) { a += g[r]; c++; } gs[j] = a / c; }
    for (let i = 0; i < n; i++) { const fj = i / HOP, j0 = Math.floor(fj), w = fj - j0; y[i] *= gs[j0] * (1 - w) + (gs[Math.min(gs.length - 1, j0 + 1)] || 1) * w; }
    if (take && ph.length) { const src = Float32Array.from(y), D = [8, 12, 6, 13, 9, 11]; let prevEnd = 0;
      ph.forEach(([a, b], i) => { const d = Math.round(D[i % D.length] * SR / 1000), s0 = Math.round((prevEnd + a) / 2) * F, s1 = Math.min(n, (i + 1 < ph.length ? Math.round((b + ph[i + 1][0]) / 2) : nF) * F); for (let k = s0; k < s1; k++) y[k] = k - d >= 0 ? src[k - d] : 0; prevEnd = b; q.take++; }); }
    return q;
  }

  // v4.0 フレーズの音量合わせ: 0.3秒以上の切れ目で区切った「フレーズ」ごとの大きさを、全体の中央へ amount の割合だけ寄せる(最大±4dB)。
  //   ばらつきが2dB未満なら何もしない。音量は切れ目(声のない所)の中だけで切りかえるので、歌っている途中で音量が動かない
  function phraseLevel(y, amount) {
    const n = y.length, F = 1024, nF = Math.floor(n / F), e = new Float32Array(nF), info = { phrases: 0, maxDb: 0, spread: 0, applied: false };
    for (let f = 0; f < nF; f++) { let a = 0; for (let i = f * F; i < (f + 1) * F; i++) a += y[i] * y[i]; e[f] = 10 * Math.log10(a / F + 1e-12); }
    const sorted = Float32Array.from(e).sort(), top = sorted[Math.floor(nF * 0.95)] || -60, th = top - 30, gapF = Math.ceil(0.3 * SR / F);
    const ph = []; let a0 = -1, last = -1;
    for (let f = 0; f <= nF; f++) { const on = f < nF && e[f] > th; if (on) { if (a0 < 0) a0 = f; last = f; } else if (a0 >= 0 && f - last > gapF) { if (last - a0 >= 4) ph.push([a0, last + 1]); a0 = -1; } }
    if (a0 >= 0 && last - a0 >= 4) ph.push([a0, last + 1]);
    info.phrases = ph.length; if (ph.length < 3) return info;
    const lv = ph.map(([a, b]) => { const v = Array.from(e.subarray(a, b)).sort((p, q) => p - q); return v[Math.floor(v.length * 0.75)]; }); // 大きめの所(上位25%)の大きさ
    const sl = lv.slice().sort((p, q) => p - q), med = sl[sl.length >> 1]; info.spread = Math.round((sl[Math.floor(sl.length * 0.9)] - sl[Math.floor(sl.length * 0.1)]) * 10) / 10;
    if (info.spread < 2) return info;
    const g = new Float32Array(nF + 1).fill(NaN);
    ph.forEach(([a, b], i) => { const d = Math.max(-4, Math.min(4, -amount * (lv[i] - med))); for (let f = a; f < b; f++) g[f] = d; info.maxDb = Math.max(info.maxDb, Math.abs(d)); });
    // 切れ目は前後のフレーズの間をなめらかに移す
    let prev = 0; for (let f = 0; f <= nF; f++) { if (g[f] === g[f]) { prev = g[f]; continue; } let j = f; while (j <= nF && !(g[j] === g[j])) j++; const nx = j <= nF ? g[j] : prev; for (let q = f; q < j; q++) g[q] = prev + (nx - prev) * (q - f + 1) / (j - f + 1); f = j - 1; }
    for (let i = 0; i < n; i++) { const fp = i / F - 0.5, f0 = Math.max(0, Math.floor(fp)), w = Math.max(0, Math.min(1, fp - f0)), d = g[Math.min(nF, f0)] * (1 - w) + g[Math.min(nF, f0 + 1)] * w; y[i] *= Math.pow(10, d / 20); }
    info.maxDb = Math.round(info.maxDb * 10) / 10; info.applied = true; return info;
  }

  // お手本の音源の位置をカラオケに合わせ、歌入りなら引き算で歌だけを取り出す
  function alignRef(x0, inst, instEnv, kind, alignAca) {
    let x = x0, lag = 0;
    const ea = onsetEnv(x0), eb = instEnv, L = Math.round(30 / ENV_DT), sc = new Float64Array(2 * L + 1); // ±30秒まで探す(イントロの長さが違う音源など)
    for (let l = -L; l <= L; l += 2) { let sm = 0; for (let i = Math.max(0, -l); i < Math.min(ea.length, eb.length - l); i += 4) sm += ea[i] * eb[i + l]; sc[l + L] = sm; }
    for (let l = -L + 1; l < L; l += 2) sc[l + L] = 0.5 * (sc[l + L - 1] + sc[l + L + 1]);
    // 候補(山の上位)を波形で確かめて、一番よく重なるずれを選ぶ(拍の周期で間違えないように)
    const cand = [];
    for (let i = 1; i < sc.length - 1; i++) if (sc[i] >= sc[i - 1] && sc[i] >= sc[i + 1]) cand.push(i);
    cand.sort((a, b) => sc[b] - sc[a]);
    if (kind === 'full') {
      const c0 = Math.floor(Math.min(inst.length, x0.length) * 0.3), N = Math.max(SR, Math.min(SR * 6, inst.length - c0 - 30000, x0.length - c0 - 30000));
      let bestLag = 0, bestC = -1e30;
      const corr = l => { let sm = 0, e = 0; for (let i = 0; i < N; i += 2) { const j = c0 + i - l; if (j >= 0 && j < x0.length) { sm += inst[c0 + i] * x0[j]; e += x0[j] * x0[j]; } } return e > 0 ? sm / Math.sqrt(e) : -1e30; };
      for (const ci of cand.slice(0, 10)) {
        const l0 = (ci - L) * 256;
        for (let l = l0 - 300; l <= l0 + 300; l += 3) { const v = corr(l); if (v > bestC) { bestC = v; bestLag = l; } }
      }
      for (let l = bestLag - 3; l <= bestLag + 3; l++) { const v = corr(l); if (v > bestC) { bestC = v; bestLag = l; } }
      lag = bestLag;
      const y = new Float32Array(inst.length); let xi = 0, ii = 0, xx = 0;
      for (let i = 0; i < inst.length; i++) { const j = i - lag; const v = j >= 0 && j < x0.length ? x0[j] : 0; y[i] = v; xi += v * inst[i]; ii += inst[i] * inst[i]; xx += v * v; }
      const g = ii > 0 ? xi / ii : 0;
      const r = new Float32Array(y.length); for (let i = 0; i < y.length; i++) r[i] = y[i] - g * inst[i];
      // 同じ音源かどうか: 歌の少ない低音(150Hz以下)が引き算で消えていれば同じ音源
      const lp = biquad('lp', 150, 0.707), yl = Float32Array.from(y), rl = Float32Array.from(r); filt(yl, lp); filt(yl, lp); filt(rl, lp); filt(rl, lp);
      let ey = 0, er = 0; for (let i = 0; i < yl.length; i += 2) { ey += yl[i] * yl[i]; er += rl[i] * rl[i]; }
      const ratio = ey > 0 ? er / ey : 1;
      if (ratio < 0.3) return { x: r, mode: 'subtract', ratio, lag };
      return { x: y, mode: 'mix', ratio, lag }; // 同じ音源でなければ、歌入りのまま使う(精度は下がる)
    }
    if (alignAca && cand.length) { lag = (cand[0] - L) * 256; const y = new Float32Array(x0.length); for (let i = 0; i < x0.length; i++) { const j = i - lag; y[i] = j >= 0 && j < x0.length ? x0[j] : 0; } x = y; }
    return { x, mode: kind, ratio: null, lag };
  }

  // お手本(歌だけ)とカラオケの頭合わせ + キーのちがい: お手本の音名が、カラオケの和音といちばん合うずれ・移調を探す
  function alignByHarmony(f0, key, instEnv, xEnv, maxSec, opts2) {
    const HC = key && key.hc, hcN = HC ? HC.length / 12 : 0, M = maxSec || 30;
    if (!HC || !hcN) return null;
    const pts = []; for (let k = 0; k < f0.length; k += 2) if (f0[k] > 0) { const m = 69 + 12 * Math.log2(f0[k] / 440) - key.tuning / 100; pts.push(k * HOP_SEC, ((Math.round(m) % 12) + 12) % 12); }
    if (pts.length < 200) return null;
    const step = 0.05, nl = Math.round(2 * M / step) + 1, scoreH = new Float64Array(nl * 12);
    for (let li = 0; li < nl; li++) { const lag = -M + li * step;
      for (let q = 0; q < pts.length; q += 2) { const f = Math.round((pts[q] + lag - key.hcT0) / key.hcDt); if (f < 0 || f >= hcN) continue; const b = f * 12, pc = pts[q + 1]; for (let st = 0; st < 12; st++) scoreH[li * 12 + st] += HC[b + ((pc + st) % 12)]; } }
    // 拍の重なり(歌い出しとカラオケのアタック)も少し足す
    const scoreO = new Float64Array(nl);
    if (xEnv) for (let li = 0; li < nl; li++) { const l = Math.round((-M + li * step) / ENV_DT); let sm = 0; for (let i = Math.max(0, -l); i < Math.min(xEnv.length, instEnv.length - l); i += 4) sm += xEnv[i] * instEnv[i + l]; scoreO[li] = sm; }
    const z = a => { let m = 0, v = 0; for (const x of a) m += x; m /= a.length; for (const x of a) v += (x - m) ** 2; v = Math.sqrt(v / a.length) || 1; return a.map(x => (x - m) / v); };
    const hcAll = new Float64Array(nl * 12); { const zz = z(Array.from(scoreH)); for (let i = 0; i < zz.length; i++) hcAll[i] = zz[i]; }
    const zo = xEnv ? z(Array.from(scoreO)) : null;
    let best = 0, bv = -1e9;
    for (let li = 0; li < nl; li++) for (let st = 0; st < (opts2 && opts2.keySearch ? 12 : 1); st++) { const v = hcAll[li * 12 + st] + (zo ? 2.0 * zo[li] : 0) - (st ? 1.0 : 0); if (v > bv) { bv = v; best = li * 12 + st; } }
    const li = Math.floor(best / 12); let st = best % 12; if (st > 6) st -= 12;
    // 移調なしとの差(はっきり良いときだけキーを変える)
    const v0 = hcAll[li * 12], vb = hcAll[best];
    return { lagSec: -M + li * step, keyShift: vb - v0 > 1.5 ? st : 0 };
  }

  // 自分の歌とお手本のメロディを重ねて、ずれ(フレーム)とキーのちがい(半音)を探す。いちばん確実
  function lagByMelody(sungF0, offF, refNotes, maxSec) {
    const ms = []; for (let k = 0; k < sungF0.length; k += 3) if (sungF0[k] > 0) ms.push(k + offF, 69 + 12 * Math.log2(sungF0[k] / 440));
    if (ms.length < 300) return null;
    const M = Math.round((maxSec || 30) / HOP_SEC), n = refNotes.length;
    const score = (L, st) => { let ok = 0, both = 0; for (let q = 0; q < ms.length; q += 2) { const j = ms[q] + L; if (j < 0 || j >= n) continue; const r = refNotes[j]; if (!(r === r)) continue; both++; let d = (ms[q + 1] - r - st) % 12; if (d > 6) d -= 12; if (d < -6) d += 12; if (Math.abs(d) < 0.75) ok++; } return both > 100 ? ok / (ms.length / 2) : 0; };
    let bL = 0, bS = 0, bv = -1;
    for (let L = -M; L <= M; L += 4) for (let st = 0; st < 12; st++) { const v = score(L, st) - (st ? 0.04 : 0); if (v > bv) { bv = v; bL = L; bS = st; } }
    for (let L = bL - 4; L <= bL + 4; L++) { const v = score(L, bS) - (bS ? 0.04 : 0); if (v > bv) { bv = v; bL = L; } }
    if (bS > 6) bS -= 12;
    return { lagF: bL, st: bS, score: bv + (bS ? 0.04 : 0), score0: score(0, 0) };
  }

  // ---------- ラウドネス(簡易 BS.1770) ----------
  function loudness(L, R, sr) {
    sr = sr || SR;
    const c1 = biquad('hs', 1500, 0.707, 4, sr), c2 = biquad('hp', 38, 0.5, sr);
    const CH = Math.round(0.1 * sr), chans = R ? [L, R] : [L];
    const nC = Math.floor(L.length / CH);
    if (nC < 4) return -70;
    const e = new Float64Array(nC);
    for (const x of chans) {
      let a1 = 0, a2 = 0, b1 = 0, b2 = 0;
      for (let c = 0; c < nC; c++) {
        let sm = 0;
        for (let i = c * CH; i < (c + 1) * CH; i++) {
          const v = x[i];
          let y = c1.b0 * v + a1; a1 = c1.b1 * v - c1.a1 * y + a2; a2 = c1.b2 * v - c1.a2 * y;
          const z = c2.b0 * y + b1; b1 = c2.b1 * y - c2.a1 * z + b2; b2 = c2.b2 * y - c2.a2 * z;
          sm += z * z;
        }
        e[c] += sm / CH;
      }
    }
    const blk = [];
    for (let c = 0; c + 4 <= nC; c++) blk.push((e[c] + e[c + 1] + e[c + 2] + e[c + 3]) / 4);
    const db = v => -0.691 + 10 * Math.log10(v + 1e-12);
    let s = 0, k = 0;
    for (const v of blk) if (db(v) > -70) { s += v; k++; }
    if (!k) return -70;
    const rel = db(s / k) - 10;
    s = 0; k = 0;
    for (const v of blk) if (db(v) > rel) { s += v; k++; }
    return k ? db(s / k) : -70;
  }

  // ---------- リミッター ----------
  function limiter(L, R, ceil) {
    const n = L.length, LA = Math.round(0.005 * SR);
    const g = new Float32Array(n).fill(1);
    for (let i = 0; i < n; i++) {
      const p = Math.max(Math.abs(L[i]), Math.abs(R[i]));
      if (p > ceil) {
        const need = ceil / p;
        for (let j = Math.max(0, i - LA); j <= i; j++) if (g[j] > need) g[j] = need;
      }
    }
    const k = 1 - Math.exp(-1 / (0.08 * SR));
    let rel = 1;
    for (let i = 0; i < n; i++) { rel = Math.min(g[i], rel + (1 - rel) * k); g[i] = rel; }
    const ring = new Float32Array(LA + 1).fill(n ? g[0] : 1);
    let sum = (LA + 1) * (n ? g[0] : 1), ri = 0;
    for (let i = 0; i < n; i++) {
      sum += g[i] - ring[ri]; ring[ri] = g[i]; ri = ri === LA ? 0 : ri + 1;
      const s = Math.min(1, sum / (LA + 1));
      L[i] *= s; R[i] *= s;
    }
  }

  // ---------- ミックス ----------
  // o: {instL, instR, dry, wetL, wetR, raw, offsetSec, balanceDb, reverbDb(null=なし)}
  function mix(o) {
    const n = o.instL.length, off = Math.round((o.offsetSec || 0) * SR), CEIL = 0.891;
    const Li = loudness(o.instL, o.instR);
    const gv = Math.pow(10, (Li + o.balanceDb - loudness(o.dry)) / 20);
    const gRaw = Math.pow(10, (Li + o.balanceDb - loudness(o.raw)) / 20);
    let gw = 0;
    if (o.reverbDb != null) {
      let ed = 0, ew = 0;
      for (let i = 0; i < o.dry.length; i++) ed += o.dry[i] * o.dry[i];
      for (let i = 0; i < o.wetL.length; i++) ew += o.wetL[i] * o.wetL[i];
      gw = ew > 0 ? gv * Math.pow(10, o.reverbDb / 20) * Math.sqrt(ed / ew) : 0;
    }
    const L = new Float32Array(o.instL), R = new Float32Array(o.instR);
    const pocket = biquad('peak', 3000, 0.8, -1.5);
    filt(L, pocket); filt(R, pocket);
    let peakB = 0;
    for (let i = 0; i < n; i++) {
      const j = i - off;
      if (j >= 0 && j < o.raw.length) {
        const b = Math.max(Math.abs(o.instL[i] + gRaw * o.raw[j]), Math.abs(o.instR[i] + gRaw * o.raw[j]));
        if (b > peakB) peakB = b;
      }
      if (j >= 0 && j < o.dry.length) { const d = gv * o.dry[j]; L[i] += d; R[i] += d; }
      if (gw > 0 && j >= 0 && j < o.wetL.length) { L[i] += gw * o.wetL[j]; R[i] += gw * o.wetR[j]; }
    }
    let peak = 0;
    for (let i = 0; i < n; i++) { const p = Math.max(Math.abs(L[i]), Math.abs(R[i])); if (p > peak) peak = p; }
    const master = peak > CEIL ? Math.min(1, CEIL * 1.41 / peak) : 1;
    if (master < 1) for (let i = 0; i < n; i++) { L[i] *= master; R[i] *= master; }
    limiter(L, R, CEIL);
    const gBefore = peakB > 0 ? Math.min(master, 0.97 / peakB) : master;
    return { L, R, before: { inst: gBefore, vocal: gBefore * gRaw }, lufsInst: Li };
  }

  // v4.2 品質チェック(AI MIX の音質ガード)。元の声とくらべて、処理で増えた問題だけを数える(歌い方そのものは問題にしない)
  //  金属音: 倍音と倍音の間の雑音の比(dB) / ジー音: 声のない所が周期的になった度合い / 音程の跳び: となりのフレームで0.8半音以上 / 音割れ / 急な音量の段差
  function qualityGuard(raw, out, f0raw) {
    const N = 2048, fft = new FFT(N), re = new Float64Array(N), im = new Float64Array(N), bin = SR / N;
    function metal(x, f0) { const r = []; for (let k = 0; k < f0.length; k += 3) { const fz = f0[k]; if (!(fz > 90)) continue; const s0 = k * HOP - N / 2; if (s0 < 0 || s0 + N > x.length) continue;
        let e0 = 0; for (let i = 0; i < N; i++) { const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N); re[i] = x[s0 + i] * w; im[i] = 0; e0 += re[i] * re[i]; } if (e0 < 1e-6) continue; fft.forward(re, im);
        let h = 0, b = 0; for (let q = 1; q * fz < 5000; q++) { const c = q * fz / bin, w = Math.max(2, fz / bin * 0.15); for (let j = Math.ceil(c - w); j <= Math.floor(c + w); j++) h += re[j] ** 2 + im[j] ** 2;
          const c2 = (q + 0.5) * fz / bin; for (let j = Math.ceil(c2 - w); j <= Math.floor(c2 + w); j++) b += re[j] ** 2 + im[j] ** 2; }
        if (h > 0) r.push(10 * Math.log10(b / h + 1e-12)); }
      r.sort((a, b) => a - b); return r.length ? r[r.length >> 1] : null; }
    function buzz(x, f0) { const r = []; for (let k = 2; k < f0.length - 2; k += 2) { if (f0[k] > 0 || f0[k - 2] > 0 || f0[k + 2] > 0) continue; const s0 = k * HOP; if (s0 + 1500 > x.length) break;
        let e0 = 0; for (let i = 0; i < 1024; i++) e0 += x[s0 + i] ** 2; if (e0 < 1e-5) continue; let best = 0; for (let L = 40; L < 400; L += 2) { let c = 0, e1 = 0; for (let i = 0; i < 1024; i++) { c += x[s0 + i] * x[s0 + i + L]; e1 += x[s0 + i + L] ** 2; } best = Math.max(best, c / Math.sqrt(e0 * e1 + 1e-12)); } r.push(best); }
      r.sort((a, b) => a - b); return r.length ? r[r.length >> 1] : 0; }
    const jumps = f0 => { let n = 0; for (let k = 1; k < f0.length; k++) if (f0[k] > 0 && f0[k - 1] > 0 && Math.abs(12 * Math.log2(f0[k] / f0[k - 1])) > 0.8) n++; return n; };
    const steps = x => { let n = 0, prev = null; for (let i = 0; i + HOP <= x.length; i += HOP) { let e = 0; for (let j = i; j < i + HOP; j++) e += x[j] * x[j]; const d = 10 * Math.log10(e / HOP + 1e-12); if (prev != null && d > -45 && prev > -45 && Math.abs(d - prev) > 12) n++; prev = d; } return n; };
    let clip = 0; for (let i = 0; i < out.length; i++) if (Math.abs(out[i]) > 0.999) clip++;
    const f0o = trackPitch(out).f0;
    return { metal: [metal(raw, f0raw), metal(out, f0o)], buzz: [buzz(raw, f0raw), buzz(out, f0o)], jumps: [jumps(f0raw), jumps(f0o)], steps: [steps(raw), steps(out)], clip };
  }

  const DSP = { recordingCheck, qualityGuard, phraseLevel, voiceStability, pitchConf, harmonyPlan, vocalDynEq, deessMulti, declick, depop, tameResonance, lagByMelody, alignByHarmony, alignRef, alignToRef, pitchShiftMusic, breathCtl, SR, HOP, HOP_SEC, ENV_DT, NOTE_NAMES, FFT, biquad, filt, detectKey, keyName, trackPitch, planCorrection, psola, processVocal, reverb, loudness, limiter, mix,
    onsetEnv, beatGrid, beatTimes, gridDelta, alignRef, spectralVoice, toOutFrames, vocalOnsets, autoOffset, planTiming, prepVocal, denoise, detectHum, toneProfile, harmNote, harmonyPlan, robotShift, vocalProfile, origAnalyze, harmFromNotes, mergeHarm, arrangeVoices, aiHarmonyPlan, noteChroma };

  if (typeof module !== 'undefined' && module.exports) module.exports = DSP;
  else if (typeof importScripts !== 'function') root.DSP = DSP; // 画面側(保存時のリミッターなどで使う)
  else {
    // ---------- Worker (複数トラック対応) ----------
    const st = { key: null, grid: null, env: null, lufsInst: -70, tr: {} };
    const prog = (label, base, span) => f => root.postMessage({ type: 'progress', label, value: base + span * f });
    const pview = t => { // 音程チェック用: 歌った高さと、合わせるべき音
      if (!st.key || !t.pitch) return null;
      const p = planCorrection(t.pitch.f0, st.key, { strength: 1, mode: 'scale', tonic: null, offsetSec: 0, amp: t.pitch.amp });
      return { midi: p.midi, target: p.target };
    };
    const T = id => st.tr[id] || (st.tr[id] = { vocal: null, pitch: null, tuneKey: '', tuned: null, plan: null, timing: null, hTarget: null });
    root.onmessage = function (ev) {
      const m = ev.data;
      try {
        if (m.cmd === 'inst') {
          for (const id in st.tr) st.tr[id].tuneKey = '';
          const mono = new Float32Array(m.L.length);
          for (let i = 0; i < mono.length; i++) mono[i] = 0.5 * (m.L[i] + m.R[i]);
          st.key = detectKey(mono, prog('キーを調べています', 0, 0.4));
          st.env = onsetEnv(mono, prog('テンポを調べています', 0.4, 0.5));
          st.grid = st.gridAuto = beatGrid(st.env);
          st.lufsInst = loudness(m.L, m.R); st.instMono = mono; st.instLR = [m.L, m.R]; // v4.6 原曲解析(引き算)のためにステレオも持つ
          root.postMessage({ type: 'key', key: st.key, name: keyName(st.key.tonic, st.key.minor), bpm: st.grid.bpm, lufsInst: st.lufsInst, beats: beatTimes(st.grid, m.L.length / SR) });
        } else if (m.cmd === 'beats') { // 手で直したテンポ(拍の位置)をリズム補正にも使う
          if (!st.gridAuto) return;
          st.grid = m.beats && m.beats.length > 4 ? gridFromBeats(m.beats, st.gridAuto) : st.gridAuto;
          for (const id in st.tr) st.tr[id].tuneKey = '';
        } else if (m.cmd === 'score') { // テイクの良さ: 音程のズレ(セント)と声の入っている割合
          const tp = trackPitch(m.x), pl = st.key ? planCorrection(tp.f0, st.key, { strength: 1, mode: 'scale', tonic: null, offsetSec: m.offsetSec, amp: tp.amp }) : null;
          let v = 0; for (const f of tp.f0) if (f > 0) v++;
          root.postMessage({ type: 'score', key: m.key, err: pl ? pl.errBefore : 0, voiced: tp.f0.length ? v / tp.f0.length : 0 });
        } else if (m.cmd === 'instShift') { // カラオケのキー変更
          const [oL, oR] = pitchShiftMusic(m.L, m.R, m.st, prog('カラオケのキーを変えています', 0, 1));
          root.postMessage({ type: 'instShifted', st: m.st, seq: m.seq, L: oL, R: oR }, [oL.buffer, oR.buffer]);
        } else if (m.cmd === 'aiSet') { // v4.2 AI MIX: AIで整えた声・AIの有声の信頼度を受け取る(元の声は残す)
          const t = T(m.id);
          if (m.clean !== undefined) { t.vocalClean = m.clean || null; t.cleanVer = (t.cleanVer || 0) + 1; }
          if (m.conf !== undefined) { t.mlConf = m.conf || null; t.mlVer = (t.mlVer || 0) + 1; }
          for (const id in st.tr) if (st.tr[id].src === m.id) st.tr[id].tuneKey = '';
          t.tuneKey = ''; root.postMessage({ type: 'aiSet', id: m.id });
        } else if (m.cmd === 'guard') { // v4.2 AI MIX の品質チェック: 元の声と、補正・前処理のあとの声をくらべる
          const t = T(m.id), srcT = T(t.src || m.id);
          root.postMessage({ type: 'guard', id: m.id, seq: m.seq, r: t.tuned && srcT.vocal ? qualityGuard(srcT.vocal, t.tuned, srcT.pitch.f0) : null });
        } else if (m.cmd === 'vocal') {
          const t = T(m.id);
          t.vocal = m.x; t.tuneKey = ''; t.vocalClean = null; t.mlConf = null; // v4.2: 声が変わったら、AIの結果は使わない(もう一度診断する)
          for (const id in st.tr) if (st.tr[id].src === m.id) st.tr[id].tuneKey = '';
          t.pitch = trackPitch(m.x, prog('歌声の音程を調べています', 0, 1)); t.pitch.stab = voiceStability(m.x, t.pitch.f0);
          let v = 0; for (let i = 0; i < t.pitch.f0.length; i++) if (t.pitch.f0[i] > 0) v++;
          let rc = null; try { rc = recordingCheck(m.x, t.pitch.f0); } catch (e) { rc = null; }
          root.postMessage({ type: 'pitch', id: m.id, seq: m.seq, voicedSec: v * HOP_SEC, lufsRaw: loudness(m.x), pv: pview(t), rec: rc });
        } else if (m.cmd === 'notes') { // ノート編集画面用: 歌をノートに分けた結果
          const t = st.tr[m.id];
          if (t && t.pitch) {
            const p = planCorrection(t.pitch.f0, st.key, { strength: 1, mode: m.mode, tonic: m.tonic, offsetSec: m.offsetSec, ref: m.useRef ? st.ref : null, amp: t.pitch.amp, cuts: m.cuts, joins: m.joins, refShift: m.refShift || 0 });
            root.postMessage({ type: 'notes', id: m.id, seq: m.seq, segs: p.notes, midi: p.midi, slow: p.slow });
          }
        } else if (m.cmd === 'ref') { // お手本の歌(アカペラ / 本家の歌入り音源から歌を取り出す)
          let al = st.instMono ? alignRef(m.x, st.instMono, st.env, m.kind === 'full' ? 'full' : 'acapella', false) : { x: m.x, mode: m.kind, ratio: null, lag: 0 };
          let keyShift = 0;
          // 歌だけ(アカペラ)の頭合わせとキーのちがいは、自分の歌が入ったときにメロディで合わせる(和音や拍だけでは、くり返しの多い曲で間違えやすい)
          const x = al.x, mode = al.mode, ratio = al.ratio, lag = al.lag;
          const tp = trackPitch(x, prog('お手本の歌を調べています', 0.3, 0.9));
          const pc = planCorrection(tp.f0, st.key, { strength: 1, mode: 'chromatic', tonic: null, offsetSec: 0, amp: tp.amp });
          const notes = Float32Array.from(pc.target, v => v + keyShift), ons = [];
          for (const r of vocalOnsets(tp.f0, pc.target, tp.amp)) for (const t of r.on) ons.push(t);
          let v = 0; for (const f of tp.f0) if (f > 0) v++;
          st.ref = { notes, onsets: Float32Array.from(ons) }; st.refVer = (st.refVer || 0) + 1;
          for (const id in st.tr) st.tr[id].tuneKey = '';
          st.refBase = { notes: Float32Array.from(notes), onsets: Float32Array.from(ons) }; st.refAdj = { sec: 0, st: 0 }; st.refByVocal = !!m.noAlign;
          root.postMessage({ type: 'ref', mode, ratio, lagSec: lag / SR, keyShift, voicedSec: v * HOP_SEC, notes, x: m.wantAudio ? x : null });
        } else if (m.cmd === 'refMidi') { // ガイドメロディ(MIDI)をお手本として使う: 曲の頭 = MIDI の頭
          const nF = Math.ceil((m.dur || 1) / HOP_SEC) + 2, notes = new Float32Array(nF).fill(NaN), ons = [];
          for (const nt of m.notes) { const a = Math.max(0, Math.round(nt.t0 / HOP_SEC)), b = Math.min(nF, Math.round(nt.t1 / HOP_SEC)); for (let k = a; k < b; k++) notes[k] = nt.p; ons.push(nt.t0); }
          st.ref = { notes, onsets: Float32Array.from(ons) }; st.refVer = (st.refVer || 0) + 1;
          for (const id in st.tr) st.tr[id].tuneKey = '';
          st.refBase = { notes: Float32Array.from(notes), onsets: Float32Array.from(ons) }; st.refAdj = { sec: 0, st: 0 }; st.refByVocal = false;
          let v = 0; for (const x of notes) if (x === x) v++;
          root.postMessage({ type: 'ref', mode: 'midi', ratio: null, lagSec: 0, keyShift: 0, voicedSec: v * HOP_SEC, notes, x: null });
        } else if (m.cmd === 'refNudge') { // お手本の位置・キーを手で直す(秒, 半音)
          if (!st.refBase) return;
          st.refAdj = { sec: m.sec || 0, st: m.st || 0 }; st.refByVocal = true; // 手で直したら自動では動かさない
          const d = Math.round((m.sec || 0) / HOP_SEC), b = st.refBase, n = new Float32Array(b.notes.length).fill(NaN);
          for (let k = 0; k < n.length; k++) { const j = k - d; if (j >= 0 && j < b.notes.length) n[k] = b.notes[j] + (m.st || 0); }
          st.ref = { notes: n, onsets: Float32Array.from(b.onsets, t => t + d * HOP_SEC) }; st.refVer = (st.refVer || 0) + 1;
          for (const id in st.tr) st.tr[id].tuneKey = '';
          root.postMessage({ type: 'refNudged', notes: n });
        } else if (m.cmd === 'refClear') {
          st.ref = null; st.refVer = (st.refVer || 0) + 1; for (const id in st.tr) st.tr[id].tuneKey = '';
        } else if (m.cmd === 'pview') {
          const t = st.tr[m.id]; if (t && t.pitch) root.postMessage({ type: 'pview', id: m.id, pv: pview(t) });
        } else if (m.cmd === 'drop') {
          delete st.tr[m.id];
        } else if (m.cmd === 'align') {
          { const tt = T(m.id), pa = planCorrection(tt.pitch.f0, st.key, { strength: 1, mode: 'scale', tonic: null, offsetSec: 0, amp: tt.pitch.amp });
            root.postMessage(Object.assign({ type: 'align', id: m.id }, autoOffset(tt.pitch.f0, st.env, st.grid, tt.pitch.amp, pa.notes.map(x => x.k0), st.ref ? st.ref.onsets : null))); }
        } else if (m.cmd === 'orig') { // v4.6 原曲解析(歌入りの原曲を、このときだけ受け取る。結果だけ返し、音声は残さない)
          if (!st.instLR) { root.postMessage({ type: 'orig', reqId: m.reqId, err: '先にカラオケを入れてください' }); return; }
          const t0 = performance.now(), u = m.userId && st.tr[m.userId], r = origAnalyze(m.L, m.R, st.instLR[0], st.instLR[1], { ctx: { instMono: st.instMono, env: st.env, grid: st.grid, key: st.key }, user: u && u.vocal ? u : null, wantStem: !!m.wantStem }, prog('原曲を調べています', 0, 1));
          const stem = r.stem; delete r.stem; root.postMessage({ type: 'orig', reqId: m.reqId, res: r, stem: stem || null, ms: Math.round(performance.now() - t0) }, stem ? [stem.buffer] : []);
        } else if (m.cmd === 'origBP') { // v4.6 採譜AIの結果と、信号処理のハモリをまとめる
          const ai = harmFromNotes(m.post, m.fps, m.mainNotes, m.sepConf, 0.5), mg = mergeHarm(m.harm, ai);
          root.postMessage({ type: 'orig', reqId: m.reqId, res: { harm: mg, aiN: ai.length } });
        } else if (m.cmd === 'vprof') { // v4.5 ボーカル解析(トラックごと。同じ音程結果なら前の結果を使う)
          const tt = st.tr[m.id];
          if (!tt || !tt.pitch || !tt.vocal) { root.postMessage({ type: 'vprof', reqId: m.reqId, id: m.id, err: 'この歌の音程がまだ調べ終わっていません' }); return; }
          const t0 = performance.now(); if (!tt.vp || tt.vp.src !== tt.pitch) tt.vp = { src: tt.pitch, out: vocalProfile(tt.vocal, tt.pitch.f0, tt.pitch.amp) };
          root.postMessage({ type: 'vprof', reqId: m.reqId, id: m.id, vp: tt.vp.out, ms: Math.round(performance.now() - t0) });
        } else if (m.cmd === 'arrange') { // v4.4 AIアレンジ: メインの歌から各ハモリ声部の音を決めて返す(画面表示用。音は作らない)
          const tt = st.tr[m.id];
          if (!tt || !tt.pitch) { root.postMessage({ type: 'arrange', reqId: m.reqId, err: 'メインの歌の音程がまだ調べ終わっていません' }); return; }
          const t0 = performance.now(), p = planCorrection(tt.pitch.f0, st.key, { strength: 1, mode: m.mode, tonic: m.tonic, offsetSec: m.offsetSec, amp: tt.pitch.amp, cuts: m.cuts, joins: m.joins, refShift: m.refShift || 0 });
          const A = arrangeVoices(p, st.key, m.offsetSec, m.roles || [], m.ov || null, m.range || null);
          root.postMessage({ type: 'arrange', reqId: m.reqId, voices: A.voices, stat: A.stat, ms: Math.round(performance.now() - t0) });
        } else if (m.cmd === 'guide') { // ハモリ録音用のお手本(メインの音と、その3度上/下)
          const p = planCorrection(T(m.id).pitch.f0, st.key, { strength: 1, mode: 'scale', tonic: m.tonic, offsetSec: m.offsetSec, amp: T(m.id).pitch.amp });
          root.postMessage({ type: 'guide', id: m.id, target: p.target, up: harmonyPlan(p, 2, st.key, m.offsetSec).target, down: harmonyPlan(p, -2, st.key, m.offsetSec).target });
        } else if (m.cmd === 'render') {
          const o = m.opts, t = T(m.id), srcT = T(m.src), v = o.voice || {};
          t.src = m.src;
          if (o.kind === 'main' && o.useRef && st.refBase && !st.refByVocal && srcT.pitch) { // 自分の歌が入ったら、メロディでお手本の位置とキーを合わせ直す(1回だけ)
            const adj = st.refAdj || { sec: 0, st: 0 }, cur = st.ref.notes;
            const r = lagByMelody(srcT.pitch.f0, Math.round((o.offsetSec || 0) / HOP_SEC), cur, 30);
            st.refByVocal = true;
            if (r && r.score > 0.25 && (Math.abs(r.lagF * HOP_SEC) > 0.35 || r.st !== 0) && r.score > r.score0 + 0.08) { // 小さなずれ(録音の遅れなど)は、リズム補正の方で直す
              // r.lagF: 歌のフレーム + lagF = お手本のフレーム → お手本を -lagF ずらす
              const na = { sec: adj.sec - r.lagF * HOP_SEC, st: adj.st + r.st }; st.refAdj = na;
              const d = Math.round(na.sec / HOP_SEC), b = st.refBase, nn = new Float32Array(b.notes.length).fill(NaN);
              for (let k = 0; k < nn.length; k++) { const j = k - d; if (j >= 0 && j < b.notes.length) nn[k] = b.notes[j] + na.st; }
              st.ref = { notes: nn, onsets: Float32Array.from(b.onsets, t => t + d * HOP_SEC) }; st.refVer = (st.refVer || 0) + 1;
              root.postMessage({ type: 'refAuto', sec: na.sec, st: na.st, notes: nn });
            }
          }
          const ctxT = (o.ctxIds || []).map(id => st.tr[id]).filter(x => x && x.ctxNotes);
          const ctxSig = ctxT.map(x => x.tuneKey.length + ':' + x.ctxNotes.length + ':' + (x.ctxNotes[0] ? x.ctxNotes[0].n : '')).join(',');
          const tk = [o.aiClean && srcT.vocalClean ? 'aic' + srcT.cleanVer : '', o.aiGuard && srcT.mlConf ? 'aig' + srcT.mlVer : '', m.src, o.strength, ctxSig, JSON.stringify(o.harmSet || []), o.kind, o.mode, o.tonic, o.rhythm, o.offsetSec, o.harm, v.pitchSt, v.formant, v.robot, v.kero, v.vib, v.whisper, v.vocoder, JSON.stringify(o.edits || []), JSON.stringify(o.noteDt || []), JSON.stringify(o.cuts || []), JSON.stringify(o.joins || []), JSON.stringify(o.pen || null), o.useRef && st.ref ? st.refVer : 0, o.hForm ? 1 : 0, o.hTake ? 1 : 0, o.pitchMode || '', o.suit === false ? 0 : 1, o.markGuard === false ? 0 : 1, o.drift || 0, o.speed, o.natural || 0, o.natFl, o.deadzone || 0, o.ambW || 0, o.shortW || 0, o.splitTh, o.autoNotes || 0, o.refShift || 0, JSON.stringify(o.ai || null), JSON.stringify(o.vpSoft || null)].join('|');
          if (tk !== t.tuneKey) {
            let f0 = srcT.pitch.f0;
            // v4.2 AI MIX(選んだときだけ): AIの音程検出が「声ではない」と強く判断したフレーム(息・語尾のかすれ)は、音程を動かさない
            if (o.aiGuard && srcT.mlConf && o.kind !== 'auto') { f0 = Float32Array.from(f0); let n = 0; for (let k = 0; k < f0.length; k++) if (f0[k] > 0 && srcT.mlConf[k] < 0.3) { f0[k] = 0; n++; } t.aiGuarded = n; } else t.aiGuarded = 0;
            const vIn = o.aiClean && srcT.vocalClean && srcT.vocalClean.length === srcT.vocal.length ? srcT.vocalClean : srcT.vocal; // v4.2 AIノイズ除去を使うときは、AIで整えた声から作る
            const against = o.kind === 'harm' && ctxT.length ? [].concat(...ctxT.map(x => x.ctxNotes)) : null;
            t.plan = planCorrection(f0, st.key, { strength: o.strength, mode: o.mode, tonic: o.tonic, offsetSec: o.offsetSec, hard: !!v.kero, edits: o.edits, pen: o.pen, ap: o.kind === 'auto' ? null : (srcT.pitch && srcT.pitch.ap) || null, conf: o.kind === 'auto' || v.kero || o.suit === false ? null : srcT.pitch.conf || null, stab: o.kind === 'auto' || v.kero || o.suit === false ? null : srcT.pitch.stab || null, safeNat: o.pitchMode === 'natural' && o.suit !== false, cuts: o.cuts, joins: o.joins, drift: o.drift || 0, speed: o.speed, natural: o.natural || 0, natFl: o.natFl, deadzone: o.deadzone || 0, ambW: o.ambW || 0, shortW: o.shortW || 0, splitTh: o.splitTh == null ? 0.7 : o.splitTh, autoNotes: o.kind === 'auto' ? 0 : o.autoNotes || 0, refShift: o.refShift || 0, ref: o.useRef && o.kind !== 'harm' ? st.ref : null, amp: srcT.pitch.amp, against });
            // ハモリ(録音)は、メインの歌い出しにそろえる(声が重なったときにバラバラに聞こえない)
            const mainOn = o.kind === 'harm' ? [].concat(...ctxT.filter(x => x.isMain && x.onsOut).map(x => x.onsOut)).sort((p2, q2) => p2 - q2) : null;
            t.timing = planTiming(f0, t.plan.target, st.grid, { strength: o.rhythm, offsetSec: o.offsetSec, amp: srcT.pitch.amp, noteDt: o.noteDt, refOnsets: mainOn && mainOn.length >= 8 ? mainOn : (o.useRef && st.ref ? st.ref.onsets : null), refWin: mainOn && mainOn.length >= 8 ? 0.12 : null, refMap: !(mainOn && mainOn.length >= 8) && o.useRef && st.ref ? t.plan.refMap : null, refNotes: o.useRef && st.ref ? st.ref.notes : null, segStarts: t.plan.notes.map(n => n.k0) });
            let shift = t.plan.shift; t.hTarget = null;
            if (o.vpSoft && o.vpSoft.r && o.vpSoft.r.length && !o.harm) { shift = Float32Array.from(shift); const f = o.vpSoft.f; for (const [a2, b2] of o.vpSoft.r) for (let k = Math.max(0, Math.floor((a2 - (o.offsetSec || 0)) / HOP_SEC)); k < Math.min(shift.length, Math.ceil((b2 - (o.offsetSec || 0)) / HOP_SEC)); k++) shift[k] *= f; } // v4.5 裏声らしい所は補正を弱める
            t.aiStat = null; if (o.harm) { const h = harmonyPlan(t.plan, o.harm, st.key, o.offsetSec, o.harmSet, o.ai); shift = h.shift; t.hTarget = h.target; t.aiStat = h.ai || null; }
            { // ほかのパートが参照する: このパートの音(曲の時間)と歌い出し
              const tg = t.hTarget || t.plan.target, tO = t.timing.tOut, off = o.offsetSec || 0, at = k => (tO ? tO[Math.min(tO.length - 1, Math.max(0, k))] : k * HOP_SEC) + off;
              t.ctxNotes = t.plan.notes.map(sg => { const km = (sg.k0 + sg.k1) >> 1; return { t0: at(sg.k0), t1: at(sg.k1 - 1), n: tg[km] === tg[km] ? Math.round(tg[km]) : sg.note }; });
              t.isMain = o.kind === 'main';
              if (t.isMain) { const fr = []; vocalOnsets(f0, t.plan.target, srcT.pitch.amp, t.plan.notes.map(n => n.k0)).forEach(r => r.on.forEach(x => fr.push(x))); t.onsOut = fr.map(x => { const kf = x / HOP_SEC, k0 = Math.floor(kf), a2 = at(k0), b2 = at(k0 + 1); return a2 + (b2 - a2) * (kf - k0); }); }
            }
            t.shiftUsed = shift;
            const fm = (v.formant || 1) * (o.harm && o.hForm ? (o.harm > 0 ? 0.97 : 1.03) : 1), spec = v.whisper > 0 || !!v.vocoder; // v4.0: 自動ハモリの声質を少しだけ変える(上のハモリは少し太く・下は少し細く)
            const need = o.strength > 0 || (o.edits && o.edits.length > 0) || !!o.pen || t.timing.warp || o.harm || v.kero || v.pitchSt || v.vib || spec || Math.abs(fm - 1) > 0.01;
            if (v.pitchSt) { shift = Float32Array.from(shift); for (let k = 0; k < shift.length; k++) if (f0[k] > 0) shift[k] += v.pitchSt; }
            if (v.vib) { shift = Float32Array.from(shift); for (let k = 0; k < shift.length; k++) if (f0[k] > 0) shift[k] += v.vib * Math.sin(2 * Math.PI * 6 * k * HOP_SEC); }
            if (v.pitchSt || v.vib) t.shiftUsed = shift;
            let y = need ? psola(vIn, f0, shift, prog('音程とリズムを補正しています', 0, spec ? 0.45 : 0.8), t.timing.warp, fm, o.markGuard !== false && !v.kero && !v.pitchSt && !spec) : vIn;
            // 声の太さはPSOLAで変える。音程を大きく下げたときに足りなかった分(al < fm)だけ、あとでスペクトルで補う
            const al = y._al;
            let lowered = false; if (al) for (let j = 0; j < al.length; j++) if (al[j] < fm - 0.01) { lowered = true; break; }
            if (spec || lowered) { // 声の太さ(音程を大きく下げたときの補正も含む)・ささやき・ボコーダー
              const nOut = Math.ceil(y.length / HOP) + 2, tOut = t.timing.tOut;
              const pm = new Float32Array(f0.length); for (let k = 0; k < f0.length; k++) pm[k] = f0[k] > 0 ? t.plan.midi[k] + shift[k] : NaN;
              const mo = toOutFrames(pm, tOut, nOut), f0o = new Float32Array(nOut), beta = new Float32Array(nOut);
              for (let j = 0; j < nOut; j++) { f0o[j] = mo[j] === mo[j] ? 440 * Math.pow(2, (mo[j] - 69) / 12) : 0; beta[j] = al ? Math.max(1, fm / al[Math.min(al.length - 1, j)]) : 1; }
              let notes = null;
              if (v.vocoder) {
                const tg = t.hTarget || t.plan.target, nt = new Float32Array(tg.length);
                let med = 60;
                if (v.vocoder === 'mono') { const vs = []; for (let k = 0; k < tg.length; k++) if (tg[k] === tg[k]) vs.push(tg[k]); vs.sort((a, b) => a - b); if (vs.length) med = vs[vs.length >> 1]; }
                for (let k = 0; k < tg.length; k++) nt[k] = tg[k] === tg[k] ? (v.vocoder === 'mono' ? med : tg[k]) + (v.pitchSt || 0) : NaN;
                notes = toOutFrames(nt, tOut, nOut);
              }
              y = spectralVoice(y, { beta, f0: f0o, whisper: v.whisper || 0, notes, vocAmt: v.vocoder ? 0.92 : 0 }, prog('声を変えています', 0.45, 0.4));
            }
            if (o.edits && o.edits.some(e => e.gain)) { // ノートごとの音量
              y = y === vIn ? Float32Array.from(y) : y; const tO = t.timing.tOut, R = Math.round(0.015 * SR);
              for (const e of o.edits) { if (!e.gain) continue; const sg = t.plan.notes.find(n => e.k >= n.k0 && e.k < n.k1); if (!sg) continue;
                const a0 = Math.round(tO[sg.k0] * SR), a1 = Math.round(tO[Math.min(tO.length - 1, sg.k1 - 1)] * SR), g = Math.pow(10, e.gain / 20);
                for (let i = Math.max(0, a0 - R); i < Math.min(y.length, a1 + R); i++) { const w = Math.min(1, (i - (a0 - R)) / R, (a1 + R - i) / R); y[i] *= 1 + (g - 1) * Math.max(0, w); } }
            }
            if (o.harm) { // v3.6 自動ハモリ: 子音・息(音程のない所)はメインと全く同じ音になるので小さく(−12dB)。メインと重なって二重に聞こえるのを防ぐ
              y = y === vIn ? Float32Array.from(y) : y; const tO = t.timing.tOut, g = new Float32Array(Math.ceil(y.length / HOP) + 2).fill(0.25);
              for (let k = 0; k < f0.length; k++) if (f0[k] > 0) { const j = Math.round((tO ? tO[k] : k * HOP_SEC) * SR / HOP); for (let q = j - 1; q <= j + 1; q++) if (q >= 0 && q < g.length) g[q] = 1; }
              const gs = Float32Array.from(g); for (let j = 0; j < g.length; j++) { let a = 0, c = 0; for (let q = j - 2; q <= j + 2; q++) if (q >= 0 && q < g.length) { a += g[q]; c++; } gs[j] = a / c; }
              for (let i = 0; i < y.length; i++) { const fj = i / HOP, j0 = Math.floor(fj), w = fj - j0; y[i] *= gs[j0] * (1 - w) + (gs[Math.min(gs.length - 1, j0 + 1)] || 0) * w; }
              // v3.8: 位置がずれないように、メインとピッタリ同じタイミングのまま(以前は12ms後ろにずらしていた)
              t.harmQ = harmPolish(y, t.shiftUsed, f0, tO, !!o.hTake);
            }
            t.tuned = y;
            t.tuneKey = tk;
          }
          root.postMessage({ type: 'progress', label: '声を整えています', value: 0.85 });
          const pv = prepVocal(t.tuned, { denoise: o.denoise, gate: o.gate, deess: o.deess, reso: o.reso || 0, dyneq: o.dyneq || 0, phrase: o.phrase || 0 }), y = pv.y;
          if (o.breath > 0 && t.plan && t.timing) { // ブレスを小さく(声のある所は補正後の時間に並べ直して判断)
            const vf = new Float32Array(srcT.pitch.f0.length); for (let k = 0; k < vf.length; k++) vf[k] = srcT.pitch.f0[k] > 0 ? 1 : NaN;
            const vo = toOutFrames(vf, t.timing.tOut, Math.ceil(y.length / HOP) + 2);
            pv.info.breath = breathCtl(y, vo, o.breath);
          }
          const p = t.plan, tm = t.timing;
          if (t.harmQ) pv.info.harmQ = t.harmQ; if (t.aiStat) pv.info.aiHarm = t.aiStat;
          root.postMessage({ type: 'done', id: m.id, voc: y, noise: pv.info, tone: toneProfile(y),
            plan: { shift: t.shiftUsed, target: t.hTarget || p.target, midi: p.midi, tOut: tm.tOut, errBefore: p.errBefore, errAfter: p.errAfter, nNotes: (p.notes || []).length, autoFix: p.autoFix || [], lowConf: (p.lowConf || []).length, skipWhy: p.skipWhy || null },
            timing: { msBefore: tm.msBefore, msAfter: tm.msAfter, matched: tm.matched, total: tm.total } }, [y.buffer]);
        }
      } catch (e) {
        root.postMessage({ type: 'error', message: String(e && e.message || e) });
      }
    };
  }
})(typeof self !== 'undefined' ? self : globalThis);
