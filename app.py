from flask import Flask, render_template, request
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# ----------------------------
# モデルの読み込み（TFLite用）
# ----------------------------
tflite_model_path = "model_fp16.tflite"

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
            img = load_img(filepath, target_size=(512, 512))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float16)  # Float16でキャスト

            # 推論実行（TFLite専用コード）
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            result = labels[np.argmax(output_data)]

            return render_template("result.html", title="分類結果", result=result, image=filepath)

    return render_template("index.html", title="パンケーキ画像分類", message="画像をアップロードして分類してみよう!!")

# ----------------------------
# アプリ起動
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
