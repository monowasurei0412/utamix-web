// マイク入力をそのまま(無加工で)取り出す録音用ワークレット
// v4.12: 左右(入力1・入力2)を別々に記録する。どちらを使うかは録音のあとで決める(auto/mix/l/r)
//        画面のメーター・音程表示・自分の声のモニターには、いまの設定で選んだ音を出す
class Rec extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false; this.startAt = 1e18; this.mode = 'auto';
    this.bufL = new Float32Array(4096); this.bufR = new Float32Array(4096); this.n = 0; this.stereo = false; this.pk = 0; this.cnt = 0;
    this.lv = new Float32Array(2048); this.ln = 0; // 画面の「いま歌っている高さ」表示用
    this.eL = 0; this.eR = 0; this.pick = 'mix'; // auto のときの、いまの選び方(ゆっくり追う)
    this.port.onmessage = e => {
      if (e.data.mode) this.mode = e.data.mode;
      if (e.data.startAt != null) { this.startAt = e.data.startAt; this.on = true; this.n = 0; }
      if (e.data.stop) { this.flush(); this.on = false; this.port.postMessage({ end: true }); }
    };
  }
  flush() {
    if (this.n > 0) {
      const c = this.bufL.slice(0, this.n), msg = { c }, tr = [c.buffer];
      if (this.stereo) { msg.c2 = this.bufR.slice(0, this.n); tr.push(msg.c2.buffer); }
      this.port.postMessage(msg, tr); this.n = 0;
    }
  }
  process(inputs, outputs) {
    const inp = inputs[0], out = outputs[0] && outputs[0][0];
    if (!inp || !inp[0]) return true;
    const a = inp[0], b = inp.length > 1 ? inp[1] : null, len = a.length;
    if (b && !this.stereo && this.n === 0) this.stereo = true;
    let from = len;
    if (this.on) {
      if (currentTime >= this.startAt) from = 0;
      else if (currentTime + len / sampleRate > this.startAt) from = Math.max(0, Math.round((this.startAt - currentTime) * sampleRate));
    }
    if (b && this.mode === 'auto') { // 片方だけに音がある(オーディオインターフェースの入力1だけにマイク等)ときは、そちらだけを使う
      let sl = 0, sr = 0; for (let i = 0; i < len; i++) { sl += a[i] * a[i]; sr += b[i] * b[i]; }
      this.eL = 0.97 * this.eL + 0.03 * sl / len; this.eR = 0.97 * this.eR + 0.03 * sr / len;
      const hi = Math.max(this.eL, this.eR), lo = Math.min(this.eL, this.eR);
      if (hi > 1e-7) { if (lo < hi * 0.03) this.pick = this.eL >= this.eR ? 'l' : 'r'; else if (lo > hi * 0.1) this.pick = 'mix'; }
    }
    const m = !b ? 'l' : this.mode === 'auto' ? this.pick : this.mode;
    for (let i = 0; i < len; i++) {
      const v = m === 'l' ? a[i] : m === 'r' ? b[i] : 0.5 * (a[i] + b[i]), av = v < 0 ? -v : v;
      if (out) out[i] = v;
      if (av > this.pk) this.pk = av;
      if (i >= from) {
        this.bufL[this.n] = a[i]; if (b) this.bufR[this.n] = b[i];
        if (++this.n === this.bufL.length) this.flush();
      }
      this.lv[this.ln++] = v;
      if (this.ln === 2048) { this.port.postMessage({ l: this.lv.slice() }); this.ln = 0; }
    }
    if (++this.cnt >= 16) { this.port.postMessage({ pk: this.pk, pick: m }); this.pk = 0; this.cnt = 0; }
    return true;
  }
}
registerProcessor('rec', Rec);
