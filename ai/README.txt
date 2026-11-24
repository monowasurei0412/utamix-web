AI MIX で使う部品（v4.2）。すべてローカルで動き、ネットには接続しない。
AI MIX で「AI診断」を押したときにだけ読み込む（かんたん・詳細モードでは読み込まない）。

ort.wasm.min.js / ort-wasm-simd-threaded.mjs / ort-wasm-simd-threaded.wasm
  ONNX Runtime Web 1.31.1（npm: onnxruntime-web, MIT License）。CPU（WebAssembly・SIMD）で1スレッド。
  WebGPU 版（ort-wasm-simd-threaded.jsep.wasm, 29.5MB）は入れていない:
   モデルが小さく(0.1MB と 8.6MB)、GPU への転送の方が重くなりやすいこと、exe が倍近くになること、開発環境で GPU の速さを測れないため。

models/swiftf0.onnx（135,090 bytes, sha256 6385e8c2ebc3872e…）
  SwiftF0（github.com/lars76/swift-f0, MIT License, LICENSE_swiftf0.txt）。16kHz・16ms ごとの音程と有声の信頼度。
models/dfn3_enc.onnx・dfn3_erb_dec.onnx・dfn3_df_dec.onnx（合計 8.6MB）
  DeepFilterNet3 の公式 ONNX 版（DeepFilterNet3_onnx.tar.gz, MIT／Apache-2.0 の選択式 → MIT）。48kHz のノイズ除去。
  前処理・後処理（STFT・ERB 特徴・マスク・ディープフィルタ）は aicore.js に書き、Python の参照実装と一致することを確認した。
models/basicpitch_nmp.onnx（230,444 bytes, sha256 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec）
  Basic Pitch（github.com/spotify/basic-pitch, Apache-2.0, LICENSE_basicpitch.txt）の公式 ONNX（basic_pitch/saved_models/icassp_2022/nmp.onnx・改変なし）。
  22050Hz・約86フレーム/秒・88鍵の音符の確率。v4.7 の AI原曲再現MIX で、取り出した歌（原曲−カラオケ）からハモリを探すときだけ使う。
  前処理（窓の長さ 43844、重なり 30 フレーム、先頭の無音 3840）と後処理（窓の両端15フレームを捨てる）は aiworker.js の bpitch。
  Python の公式実装（basic_pitch.inference）の音符と、合成テスト4件で同じ数を見つけることを確認した（数値の完全一致までは確認していない）。

models/e5small_q.onnx（118,308,185 bytes, sha256 f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193）
models/e5small_spm.model（sha256 cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865）
  multilingual-e5-small（huggingface.co/intfloat/multilingual-e5-small, MIT License, LICENSE_e5.txt）。
  ONNX は Xenova/multilingual-e5-small の onnx/model_quantized.onnx（int8・改変なし。ハッシュは配布元の表示と一致）。
  v4.8 の「言葉で直す」で、文を 384 次元の数にする（"query: " を付け、単語ごとの出力を平均して長さ1に）。
  単語分けは spmtok.js（sentencepiece の Unigram を自前で実装。Python 版 test/nlu/spm.py と、例文・確認文 308 件で番号が一致）。
models/nlu_examples.json
  操作ごとの例文 194 件（test/nlu/examples.json）を、開発時に test/nlu/build_nlu.py で数にしたもの（int8）。

