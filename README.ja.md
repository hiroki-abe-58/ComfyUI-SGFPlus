# Self Gradient Forcing+ for ComfyUI（日本語概要）

公式の **Self Gradient Forcing+（SGF+）** モデル（[Zihan-Su/Self_Gradient_Forcing_Plus](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus)、
論文 [arXiv:2610.10429](https://arxiv.org/abs/2610.10429)）で、ComfyUI 上でテキストから動画を生成するノードパックです。SGF+ は
自己回帰型の Wan2.1-T2V-1.3B で、ブロックごとに 4 ステップで動画を生成し、別の「メモリ」エキスパートが完成したブロックを
アテンションキャッシュへ書き込みます。公式のローリングキャッシュを使うと、学習時の 5 秒を大きく超える長さでも生成を続けられます。
処理はすべて **ComfyUI のプロセス内**で行い、ノードが何かをダウンロードすることはありません。

> **非公式**のインテグレーションです。SGF+ の著者、Wan チーム（Alibaba）、Comfy Org とは関係がなく、承認も受けていません。
> 詳細は英語版 [README.md](README.md) と [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。

## ノード

| ノード | 内容 |
| --- | --- |
| **SGF+ Loader** | `models/sgf_plus` のチェックポイント、バリアント（chunkwise / framewise）、テキストエンコーダーの置き場所、常駐の有無を選ぶ |
| **SGF+ Text to Video** | プロンプト・シード・長さ（潜在フレーム数）・長尺キャッシュ → `IMAGE`（832 x 480 のフレーム列）、`fps`（16）、JSON レポート |
| **SGF+ Unload** | 常駐モデルを解放（GPU と RAM） |

フレームは ComfyUI 標準の **Create Video**（fps はノードの出力）と **Save Video** に接続します。同梱のワークフローもこの構成です。

## 確認した範囲

Windows 11、RTX 5090（32 GB）、ComfyUI v0.38.0：

- **公式コードとビット単位で一致**：上流が固定しているライブラリ（Python 3.11、torch 2.8.0、diffusers 0.31.0、transformers
  4.49.0、numpy 1.24.4）の環境で、公式 `inference.py`（公式ランチャーの引数）と同じ動画（float フレームの SHA-256）を 5 件中
  5 件で再現しました。chunkwise の通常キャッシュ（2 プロンプト、81 フレーム）、chunkwise の長尺ローリングキャッシュ（189 フレーム）、
  framewise の通常キャッシュ（81 フレーム）、framewise のローリングキャッシュ（165 フレーム）です。flash-attn は Windows 版が
  ないため、公式側の実行では同じ計算をする PyTorch SDPA に置き換えています（パックと同じ変更）。
- **実際の ComfyUI キュー**：torch 2.14.1 / diffusers 0.41.0 / transformers 5.20.1 の ComfyUI 環境で、キュー実行はパックの
  プロセス内実行とバイト単位で同じフレームになりました（chunkwise と framewise）。もう一方の環境の公式実行と比べると、場面は同じで
  細部が異なります。bfloat16 の自己回帰モデルでは、ライブラリ間の小さな数値差（テキストエンコーダーの出力で約 2 %）がブロックを
  追うごとに積み重なるためです。コードの差ではなく環境の差で、同じ環境なら毎回同じ動画になります。
- **実モデルでの実行**：両バリアント、両キャッシュモード、5〜20 秒、中断（とその後の実行）、keep_loaded オフ、Unload、
  ブロックの倍数でない長さ（実行前にエラー）。

**未確認**：Linux / macOS の GPU、AMD / Apple の GPU、VRAM 24 GB 未満の GPU、他の ComfyUI バージョン、flash-attn を入れた環境、
論文ベンチマークでの画質評価（公式出力の再現までを確認しています）。

## インストール

1. V3 ノード API のある ComfyUI（**v0.38.0** で確認）、Python 3.10 以上、NVIDIA GPU（ピーク確保量は chunkwise で 13.7 GiB、
   framewise で 15.8 GiB。試したのは 32 GB のカードのみ）。空き VRAM が 40 GB 未満なら 11 GB のテキストエンコーダーは RAM に置かれます。
2. このフォルダーを `ComfyUI/custom_nodes/` に置き、ComfyUI の Python に依存パッケージを入れます：

   ```
   python -m pip install -r custom_nodes/ComfyUI-SGFPlus/requirements.txt
   ```

3. モデルをダウンロードします（chunkwise で約 29 GB、framewise を加えるとさらに約 11.4 GB）。確認済みのリビジョンに固定し、全ファイルの
   SHA-256 を検証します：

   ```
   python custom_nodes/ComfyUI-SGFPlus/tools/download_model.py --models-dir models/sgf_plus
   python custom_nodes/ComfyUI-SGFPlus/tools/download_model.py --models-dir models/sgf_plus --variants chunkwise framewise
   ```

   `models/sgf_plus/Wan2.1-T2V-1.3B/`（フォルダー名は固定）と `sgf_plus_chunkwise.pt`（必要なら `sgf_plus_framewise.pt`）が
   できます。別の場所は `extra_model_paths.yaml` のキー `sgf_plus` で指定できます。

4. ComfyUI を再起動し、`workflows/gui/t2v_basic.json`（5 秒）または `workflows/gui/t2v_long.json`（20 秒）を開きます。

flash-attn は不要です。

## 使い方の要点

- **長さ**：`latent_frames` は Wan の潜在フレーム数で、動画フレーム数 = 4 x latent_frames - 3 です。21 で 81 フレーム（16 fps で
  約 5 秒、学習時の長さ）。chunkwise では 3 の倍数が必要です。時間は長さにほぼ比例します。
- **long_video**（既定オン）は公式 SGF+ ランチャー（`scripts/infer_sgf_plus.sh`）の設定で、先頭の数フレームを固定した直近
  フレームのローリングキャッシュです。オフは `--long_video` なしの `inference.py` の通常キャッシュ（5 秒用）です。
- **シード**：同じモデル・プロンプト・シード・設定・環境なら同じ動画になります（確認済み）。ライブラリのバージョンが違うと
  同じシードでも別の動画になります。
- **text_encoder**：`auto` は公式の規則どおり、空き VRAM が 40 GB 未満なら 11 GB の umT5 エンコーダーを RAM に置き、使うときだけ
  GPU へ転送します（`dynamic_swap`）。
- **中断**は生成器の呼び出しの合間で止まります（5 秒の chunkwise で 35 回）。モデルは常駐したままで、次のジョブは普通に動きます。

## ライセンス

- このパック独自のコード（`sgfplus_comfy/vendor` 以外）：**MIT**（[LICENSE](LICENSE)）。
- `sgfplus_comfy/vendor/sgf`：上流 SGF+ のコード、**Apache-2.0**。6 ファイルに記録済みの変更があります
  （[vendor/UPSTREAM.md](sgfplus_comfy/vendor/UPSTREAM.md)）。
- **由来についての注意**：SGF+ は Self Gradient Forcing と Self Forcing（いずれも Apache-2.0）を土台にしています。Self Forcing は
  CausVid を土台にしたと書いており、CausVid のリポジトリは **CC BY-NC-SA 4.0** で公開されています。下流のプロジェクトはこのコードを
  Apache-2.0 で公開しており、このパックもそのライセンスで再配布しますが、その経緯を確定させるものではありません。商用利用の前には
  ご自身で確認してください（[NOTICE](NOTICE)）。
- モデル（同梱しません。`tools/download_model.py` を実行したときだけ取得）：`ZihanSu/Self_Gradient_Forcing_Plus` と
  `Wan-AI/Wan2.1-T2V-1.3B`、モデルカード上はいずれも Apache-2.0。
