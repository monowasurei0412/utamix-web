// SentencePiece(Unigram)の単語分け。multilingual-e5-small(XLM-RoBERTa の辞書)用。
// sentencepiece.bpe.model(protobuf)を読み、正規化(NFKC・空白をまとめる)→ 先頭と空白を「▁」に → 最もスコアの高い分け方(Viterbi)。
// 番号は XLM-RoBERTa(fairseq)の並び: <s>=0, <pad>=1, </s>=2, <unk>=3, それ以外は sentencepiece の番号+1。
// 確認: "Hello world" → [0, 35378, 8999, 2](公開されている XLM-R の結果と同じ)。Python 版(test/v48_spm.py)と例文・確認文すべてで一致。
(function (root) {
  function varint(b, i) { let r = 0, s = 0, c; do { c = b[i++]; r += (c & 0x7f) * Math.pow(2, s); s += 7; } while (c >= 0x80); return [r, i]; }
  function fields(b, from, to, cb) {
    let i = from;
    while (i < to) {
      let k; [k, i] = varint(b, i); const f = Math.floor(k / 8), w = k & 7;
      if (w === 0) { let v; [v, i] = varint(b, i); cb(f, w, v); }
      else if (w === 1) { cb(f, w, i); i += 8; }
      else if (w === 5) { cb(f, w, i); i += 4; }
      else if (w === 2) { let l; [l, i] = varint(b, i); cb(f, w, i, i + l); i += l; }
      else throw new Error('protobuf wire ' + w);
    }
  }
  function SpmTok(buf) {
    const b = new Uint8Array(buf), dv = new DataView(b.buffer, b.byteOffset, b.byteLength), td = new TextDecoder();
    const map = new Map(); let n = 0, maxLen = 0, minScore = 0;
    fields(b, 0, b.length, (f, w, a, e) => {
      if (f !== 1 || w !== 2) return;
      let piece = '', score = 0, type = 1;
      fields(b, a, e, (f2, w2, a2, e2) => { if (f2 === 1) piece = td.decode(b.subarray(a2, e2)); else if (f2 === 2) score = dv.getFloat32(a2, true); else if (f2 === 3) type = a2; });
      if (type === 1 || type === 4) { map.set(piece, [n, score]); const L = [...piece].length; if (L > maxLen) maxLen = L; if (score < minScore) minScore = score; }
      n++;
    });
    this.size = n; this.map = map; this.maxLen = maxLen; this.unk = minScore - 10;
  }
  SpmTok.prototype.norm = function (t) { t = t.normalize('NFKC').split(/\s+/).filter(Boolean).join(' '); return '▁' + t.replace(/ /g, '▁'); };
  SpmTok.prototype.spmIds = function (t) {
    const x = [...this.norm(t)], n = x.length, best = new Float64Array(n + 1).fill(-1e18), bi = new Int32Array(n + 1).fill(-1), bid = new Int32Array(n + 1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] <= -1e17) continue;
      let s = '';
      for (let L = 1; L <= Math.min(this.maxLen, n - i); L++) { s += x[i + L - 1]; const e = this.map.get(s); if (e) { const v = best[i] + e[1]; if (v > best[i + L]) { best[i + L] = v; bi[i + L] = i; bid[i + L] = e[0]; } } }
      const v = best[i] + this.unk; if (bi[i + 1] < 0 && v > best[i + 1]) { best[i + 1] = v; bi[i + 1] = i; bid[i + 1] = 0; }
    }
    const ids = []; for (let j = n; j > 0; j = bi[j]) ids.push(bid[j]); ids.reverse();
    return ids.filter((id, k) => !(id === 0 && k > 0 && ids[k - 1] === 0));
  };
  SpmTok.prototype.encode = function (t, max) { const ids = this.spmIds(t).map(i => i === 0 ? 3 : i + 1).slice(0, (max || 128) - 2); return [0].concat(ids, [2]); };
  if (typeof module !== 'undefined') module.exports = SpmTok; else root.SpmTok = SpmTok;
})(this);
