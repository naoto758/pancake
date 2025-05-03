from tensorflow.lite.python.interpreter import Interpreter
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os
import gdown
from flask import send_from_directory

app = Flask(__name__)

# ----------------------------
# Google Driveからモデルを取得
# ----------------------------
file_id = "1gC1ixOXhn1NGhBRBah71ODWKpuawSX9d"
url = f"https://drive.google.com/uc?id={file_id}"
tflite_model_path = "model_fp16.tflite"

if not os.path.exists(tflite_model_path):
    print("🔽 モデルをGoogle Driveからダウンロード中...")
    gdown.download(url, tflite_model_path, quiet=False)

# ----------------------------
# モデルの読み込み（TFLite用）
# ----------------------------
if not os.path.exists(tflite_model_path):
    raise FileNotFoundError(f"❌ モデルファイルが見つかりません: {tflite_model_path}")

interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# ラベル設定
# ----------------------------
labels = ['choco', 'classic', 'fruit']

# ----------------------------
# アップロード用ディレクトリ設定
# ----------------------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # 画像ファイルへのアクセス設定
    return send_from_directory(os.path.join(app.root_path, 'static/uploads'), filename)

# ----------------------------
# ルーティング定義
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # 画像読み込みと前処理（512x512で正規化）
            img = Image.open(filepath).resize((512, 512)).convert("RGB")
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            # 推論実行（TFLite専用コード）
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            result = labels[np.argmax(output_data)]
            image_url = os.path.join('uploads', file.filename)  # 相対パスに変更

            return render_template("result.html", title="分類結果", result=result, image=image_url)

    return render_template("index.html", title="パンケーキ画像分類", message="画像をアップロードして分類してみよう!!")

# ----------------------------
# アプリ起動
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
