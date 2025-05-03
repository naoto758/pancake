from tflite_runtime.interpreter import Interpreter  # ← 軽量ランタイムに変更
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import numpy as np
import os
import gdown
import traceback

app = Flask(__name__)

# ----------------------------
# モデル取得
# ----------------------------
file_id = "1gC1ixOXhn1NGhBRBah71ODWKpuawSX9d"
url = f"https://drive.google.com/uc?id={file_id}"
tflite_model_path = "model_fp16.tflite"

if not os.path.exists(tflite_model_path):
    print("🔽 モデルをGoogle Driveからダウンロード中...")
    gdown.download(url, tflite_model_path, quiet=False)

# ----------------------------
# モデル読み込み
# ----------------------------
if not os.path.exists(tflite_model_path):
    raise FileNotFoundError(f"❌ モデルファイルが見つかりません: {tflite_model_path}")

print("✅ モデル読み込み中...")
interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ モデル準備完了")

# ----------------------------
# ラベル
# ----------------------------
labels = ['choco', 'classic', 'fruit']

# ----------------------------
# アップロード先
# ----------------------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/uploads'), filename)

# ----------------------------
# メイン画面
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    print("🔥 home() にアクセスされました")
    print("🔍 request.method:", request.method)

    if request.method == "POST":
        print("📥 POSTリクエスト受信！")
        print("📤 POSTデータ:", request.form)
        print("📤 FILES:", request.files)

        file = request.files.get("image")
        print("📦 file =", file)

        if file and file.filename:
            print("✅ ファイル名あり:", file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print("📁 ファイル保存完了:", filepath)

            try:
                print("🖼️ 画像読み込み＆リサイズ")
                img = Image.open(filepath).resize((512, 512)).convert("RGB")
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

                print("🤖 推論実行中...")
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                result = labels[np.argmax(output_data)]
                print("✅ 分類成功:", result)

            except Exception as e:
                result = "分類に失敗しました"
                print("❌ 分類エラー:", str(e))
                traceback.print_exc()
                return render_template("result.html", title="分類結果", result=result, image=None)

            print("✅ result.html を描画します")
            return render_template("result.html", title="分類結果", result=result, image=file.filename)

        else:
            print("⚠️ ファイルが存在しないか、ファイル名が空です")
            return render_template("index.html", title="パンケーキ画像分類", message="⚠️ 画像ファイルが選択されていません。")

    return render_template("index.html", title="パンケーキ画像分類", message="画像をアップロードして分類してみよう!!")
