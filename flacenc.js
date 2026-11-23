/* FLAC(可逆圧縮)の書き出し。音はWAVとまったく同じまま、ファイルを小さくする。
   固定予測(0〜4次)とライス符号。フレーム4096サンプル。MD5は「不明(0)」で書く(仕様で許されている)。 */
(function (root) {
  'use strict';
  const CRC8 = new Uint8Array(256), CRC16 = new Uint16Array(256);
  for (let i = 0; i < 256; i++) { let c = i; for (let k = 0; k < 8; k++) c = c & 0x80 ? ((c << 1) ^ 0x07) & 0xFF : (c << 1) & 0xFF; CRC8[i] = c; let d = i << 8; for (let k = 0; k < 8; k++) d = d & 0x8000 ? ((d << 1) ^ 0x8005) & 0xFFFF : (d << 1) & 0xFFFF; CRC16[i] = d; }
  function BW(cap) { this.b = new Uint8Array(cap); this.p = 0; }
  BW.prototype.grow = function (n) { if ((this.p >> 3) + n + 8 > this.b.length) { const nb = new Uint8Array(Math.max(this.b.length * 2, (this.p >> 3) + n + 1024)); nb.set(this.b); this.b = nb; } };
  BW.prototype.put = function (v, n) { this.grow(8); for (let i = n - 1; i >= 0; i--) { if (Math.floor(v / Math.pow(2, i)) & 1) this.b[this.p >> 3] |= 0x80 >> (this.p & 7); this.p++; } };
  BW.prototype.putS = function (v, n) { this.put(v < 0 ? v + Math.pow(2, n) : v, n); };
  BW.prototype.unary = function (q) { this.grow((q >> 3) + 2); this.p += q; this.b[this.p >> 3] |= 0x80 >> (this.p & 7); this.p++; };
  BW.prototype.align = function () { this.p = (this.p + 7) & ~7; };
  const zz = v => v >= 0 ? 2 * v : -2 * v - 1;
  function residual(x, o) { const n = x.length, r = new Int32Array(n - o); for (let i = o; i < n; i++) { let p = 0; if (o === 1) p = x[i - 1]; else if (o === 2) p = 2 * x[i - 1] - x[i - 2]; else if (o === 3) p = 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3]; else if (o === 4) p = 4 * x[i - 1] - 6 * x[i - 2] + 4 * x[i - 3] - x[i - 4]; r[i - o] = x[i] - p; } return r; }
  function riceBits(u, a, z) { let best = 1e18, bk = 0, s = 0; for (let i = a; i < z; i++) s += u[i]; for (let k = 0; k <= 14; k++) { const b = (k ? Math.floor(s / Math.pow(2, k)) + (z - a) * (k + 1) : s + (z - a)); /* 近似(上限)。実際の符号化は下で正確に */ if (b < best) { best = b; bk = k; } } let exact = 0; for (let i = a; i < z; i++) exact += Math.floor(u[i] / Math.pow(2, bk)) + 1 + bk; return [exact, bk]; }
  function planResidual(r, bs, o) { // 区切り(partition order)と、区切りごとのライス係数
    const u = new Float64Array(r.length); for (let i = 0; i < r.length; i++) u[i] = zz(r[i]);
    let best = null;
    for (let po = 0; po <= 6; po++) { const np = 1 << po; if (bs % np || (bs >> po) <= o) break; let tot = 6 /* method+order */, ks = [], idx = 0; for (let p = 0; p < np; p++) { const len = (bs >> po) - (p === 0 ? o : 0), [b, k] = riceBits(u, idx, idx + len); tot += 4 + b; ks.push(k); idx += len; } if (!best || tot < best.bits) best = { po, ks, bits: tot }; }
    best.u = u; return best;
  }
  function encode(chs, sr, bps, onProg) {
    const nch = chs.length, n = chs[0].length, BS = 4096, q = Math.pow(2, bps - 1) - 1;
    const ints = chs.map(c => { const a = new Int32Array(n); for (let i = 0; i < n; i++) { const v = Math.round(Math.max(-1, Math.min(1, c[i])) * q); a[i] = v; } return a; });
    const w = new BW(n * nch * 2 + 4096);
    // ヘッダとSTREAMINFO
    w.put(0x664C6143, 32); w.put(1, 1); w.put(0, 7); w.put(34, 24);
    w.put(BS, 16); w.put(BS, 16); w.put(0, 24); w.put(0, 24); w.put(sr, 20); w.put(nch - 1, 3); w.put(bps - 1, 5); w.put(Math.floor(n / 4294967296), 4); w.put(n % 4294967296, 32); for (let i = 0; i < 16; i++) w.put(0, 8);
    const srCode = sr === 44100 ? 9 : sr === 48000 ? 10 : sr === 96000 ? 11 : sr === 88200 ? 0 : 0, ssCode = bps === 16 ? 4 : bps === 24 ? 6 : 0;
    let fn = 0;
    for (let s0 = 0; s0 < n; s0 += BS, fn++) {
      const bs = Math.min(BS, n - s0), start = w.p >> 3;
      w.put(0x3FFE, 14); w.put(0, 1); w.put(0, 1); w.put(bs === BS ? 12 : 7, 4); w.put(srCode, 4); w.put(nch - 1, 4); w.put(ssCode, 3); w.put(0, 1);
      // フレーム番号(UTF-8風)
      if (fn < 0x80) w.put(fn, 8); else if (fn < 0x800) { w.put(0xC0 | (fn >> 6), 8); w.put(0x80 | (fn & 63), 8); } else if (fn < 0x10000) { w.put(0xE0 | (fn >> 12), 8); w.put(0x80 | ((fn >> 6) & 63), 8); w.put(0x80 | (fn & 63), 8); } else { w.put(0xF0 | (fn >> 18), 8); w.put(0x80 | ((fn >> 12) & 63), 8); w.put(0x80 | ((fn >> 6) & 63), 8); w.put(0x80 | (fn & 63), 8); }
      if (bs !== BS) w.put(bs - 1, 16);
      let c8 = 0; for (let i = start; i < (w.p >> 3); i++) c8 = CRC8[c8 ^ w.b[i]]; w.put(c8, 8);
      for (let c = 0; c < nch; c++) {
        const x = ints[c].subarray(s0, s0 + bs);
        let allSame = true; for (let i = 1; i < bs; i++) if (x[i] !== x[0]) { allSame = false; break; }
        if (allSame) { w.put(0, 1); w.put(0, 6); w.put(0, 1); w.putS(x[0], bps); continue; }
        let best = null; for (let o = 0; o <= Math.min(4, bs - 1); o++) { const r = residual(x, o), pl = planResidual(r, bs, o), bits = o * bps + pl.bits; if (!best || bits < best.bits) best = { o, r, pl, bits }; }
        if (best.bits >= bs * bps) { w.put(0, 1); w.put(1, 6); w.put(0, 1); for (let i = 0; i < bs; i++) w.putS(x[i], bps); continue; } // そのまま(VERBATIM)
        w.put(0, 1); w.put(8 + best.o, 6); w.put(0, 1);
        for (let i = 0; i < best.o; i++) w.putS(x[i], bps);
        const pl = best.pl; w.put(0, 2); w.put(pl.po, 4);
        let idx = 0; for (let p = 0; p < (1 << pl.po); p++) { const len = (bs >> pl.po) - (p === 0 ? best.o : 0), k = pl.ks[p]; w.put(k, 4); const kp = Math.pow(2, k); for (let i = idx; i < idx + len; i++) { const u = pl.u[i]; w.unary(Math.floor(u / kp)); if (k) w.put(u % kp, k); } idx += len; }
      }
      w.align(); let c16 = 0; for (let i = start; i < (w.p >> 3); i++) c16 = ((c16 << 8) & 0xFFFF) ^ CRC16[(c16 >> 8) ^ w.b[i]]; w.put(c16, 16);
      if (onProg && (fn & 31) === 0) onProg(s0 / n);
    }
    return w.b.slice(0, w.p >> 3);
  }
  const api = { encode };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.FLACENC = api;
  if (typeof self !== 'undefined' && typeof window === 'undefined' && typeof module === 'undefined') self.onmessage = e => { const m = e.data; try { const f = encode(m.chs, m.sr, m.bps, p => self.postMessage({ type: 'prog', p })); self.postMessage({ type: 'done', flac: f }, [f.buffer]); } catch (err) { self.postMessage({ type: 'error', message: err.message || String(err) }); } };
})(typeof self !== 'undefined' ? self : globalThis);
