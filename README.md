# Lora Lab — 複数Lora検証ヘルパー

> **English summary:** Lora Lab is an extension for AUTOMATIC1111 / Forge WebUI
> (Classic & Neo) that helps you test multiple Lora at different strengths
> without typing `<lora:name:weight>` tags by hand. Each slot can also add the
> Lora's trigger words (from its .civitai.info trainedWords, a JSON sidecar, or
> the safetensors "activation text" metadata) to the front of the prompt, with
> a per-slot checkbox, on by default and never duplicating words already in the
> prompt. It adds 10 Lora slots with
> search, preview thumbnails, +/-0.1 steppers and strength sliders, a
> click-to-select thumbnail picker (with folder filter, selection frames and
> sequence numbers), and a button that reflects the current tags into the
> prompt box. Applied tags are also recorded in the PNG info as `Lora Lab: ...`.
> Install: put this repository into your WebUI's `extensions` folder, or use
> Extensions → Install from URL with this repository's URL.

Forge (Classic / Neo)・AUTOMATIC1111系 WebUI 用の自作拡張です。
プロンプト欄に `<lora:名前:強度>` を手打ちする代わりに、画面上の枠で完結します。
UI上の表記はすべて **Lora** に統一しています。

## できること

- **Lora枠が10個**（txt2img / img2img 両方にアコーディオンで常駐。スクリプト選択は不要）
- 各枠：**ON/OFFチェック** ＋ **トリガーワードON/OFFチェック** ＋ **検索できるドロップダウン** ＋ **サムネ表示** ＋ **強度スライダー（-2.00〜+2.00、0.05刻み）**
- 強度の **−0.1 / ＋0.1 ボタン**（細かい調整はスライダー右の数値欄に直接入力も可）
- **トリガーワード自動追加**：枠の「トリガー」チェックがONの Lora は、その Lora のトリガーワードが生成時にプロンプト先頭へ自動で入る（デフォルトON、枠ごとに切替可）。取得元は `.civitai.info` の trainedWords → `.json` サイドカー → safetensors の `activation text` メタデータの順。既にプロンプトにあるワードは二重に入らない。使われたワードは PNG info に `Lora Lab Trigger: ...` として記録される
- **プロンプトに反映ボタン**（アコーディオン上部）：今の枠のトリガーワード＋タグをプロンプト欄へ直接入れる。生成時の自動追加と二重にはならない
- 選んだ Lora の**サムネ画像**（モデルと同じ場所にある `.preview.png` など）が枠の横に出る
- **サムネ一覧から即選べる**：フォルダと名前で絞り込んでカード一覧を表示し、サムネを**クリックした瞬間に選択**される。選んだカードには**枠と選択順の番号**が付き、開始枠から順に上の枠へ即反映される。もう一度クリックで解除。選択中の件数と内訳も表示される（サムネ画像がある Lora のみ、先頭60件まで表示）
- 一覧に無い Lora は**名前を手入力**でも指定可（一覧の取りこぼし対策）
- 生成時、選んだ枠が `<lora:名前:強度>` として**プロンプト末尾に自動追加**される（プロンプト欄に既にあるタグは二重に追加しない）
- 追加されるタグの**プレビュー表示**つき
- 使った組み合わせは画像の **PNG info に `Lora Lab: ...` として記録**される（検証の後追い用）
- 補助ボタン（横幅いっぱいの横長ボタン）：`🔄 Lora一覧を再読み込み` / `🧹 全枠クリア` / `強度を全部 1.0 に戻す`
- 同じ Lora を2枠で選んだ場合は**後ろの枠の強度が優先**（プレビューにも1個だけ出る）

## 使い方

1. txt2img（または img2img）画面で **「Lora Lab — 複数Lora検証（10枠）」** を開く
2. 枠ごとに Lora を選び、スライダー／±ボタンで強度を決める
   - ドロップダウンで選ぶ方法と、下の**サムネ一覧**をクリックして即入れる方法がある
   - サムネ一覧は**フォルダ**と名前で絞り込める
   - 一時的に外したい枠は左端のチェックを外す（選択と強度は保持される）
3. タグをプロンプト欄で見たい・編集したいときは上部の**プロンプトに反映**を押す
4. あとは普通に Generate。強度違いを比べたいときは、強度を変えて同じシードで Generate し直すだけ

## インストール

このフォルダ `sd-webui-lora-lab` を、対象WebUIの `extensions` フォルダに置いて WebUI を再起動するだけです。

例：
- Forge Classic: `D:\sd-webui-forge-classic\extensions\sd-webui-lora-lab`
- Forge Neo: `<Neoのインストール先>\extensions\sd-webui-lora-lab`

追加の pip インストールは不要です（WebUI標準の gradio だけを使用）。

## アンインストール

`extensions\sd-webui-lora-lab` フォルダを消して再起動するだけです。
モデル・設定・他の拡張には一切触りません。

## 仕組みと注意

- 生成時の適用は「プロンプトへ `<lora:...>` タグを足す」だけなので、素の Forge の Lora 適用と結果は同一です。
- Lora一覧は、WebUI内蔵の Lora 登録簿（実ファイルパス付き）と `models\Lora` 等のフォルダ走査（ジャンクション追従）から取得します。
- サムネ選択はカード形式のHTML＋小さな javascript で実装しています。クリックした瞬間にブラウザ側で枠と番号が付き、隠しフィールド経由で枠へ反映されます。
- サムネは Lora ファイルと同じフォルダにある `<名前>.preview.png` / `<名前>.png` 等を使います。無い Lora はサムネなしです。
- ネガティブプロンプト側への Lora 追加はしません（Loraタグは通常プロンプト側専用のため）。

v1.5.3 / 2026-11-10
