// マイク入力をそのまま(無加工で)取り出す録音用ワークレット
class Rec extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false; this.startAt = 1e18;
    this.buf = new Float32Array(4096); this.n = 0; this.pk = 0; this.cnt = 0;
    this.lv = new Float32Array(2048); this.ln = 0; // 画面の「いま歌っている高さ」表示用
    this.port.onmessage = e => {
      if (e.data.startAt != null) { this.startAt = e.data.startAt; this.on = true; this.n = 0; }
      if (e.data.stop) { this.flush(); this.on = false; this.port.postMessage({ end: true }); }
    };
  }
  flush() {
    if (this.n > 0) { const c = this.buf.slice(0, this.n); this.port.postMessage({ c }, [c.buffer]); this.n = 0; }
  }
  process(inputs) {
    const inp = inputs[0];
    if (!inp || !inp[0]) return true;
    const a = inp[0], b = inp.length > 1 ? inp[1] : null, len = a.length;
    let from = len;
    if (this.on) {
      if (currentTime >= this.startAt) from = 0;
      else if (currentTime + len / sampleRate > this.startAt) from = Math.max(0, Math.round((this.startAt - currentTime) * sampleRate));
    }
    for (let i = 0; i < len; i++) {
      const v = b ? 0.5 * (a[i] + b[i]) : a[i], av = v < 0 ? -v : v;
      if (av > this.pk) this.pk = av;
      if (i >= from) { this.buf[this.n++] = v; if (this.n === this.buf.length) this.flush(); }
      this.lv[this.ln++] = v;
      if (this.ln === 2048) { this.port.postMessage({ l: this.lv.slice() }); this.ln = 0; }
    }
    if (++this.cnt >= 16) { this.port.postMessage({ pk: this.pk }); this.pk = 0; this.cnt = 0; }
    return true;
  }
}
registerProcessor('rec', Rec);
