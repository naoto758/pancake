from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import gdown

app = Flask(__name__)

# モデルのダウンロードと読み込み
file_id = "1iVogIOYWPWwzwtEgoW_jH1Xb9GXFdmVz"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "model.keras"

if not os.path.exists(model_path):
    print("🔽 gdownでGoogle Driveからモデルをダウンロードします...")
    gdown.download(url, model_path, quiet=False)

# モデル存在チェックとサイズログ出力
if not os.path.exists(model_path):
    print("❌ モデルファイルが存在しません")
    raise FileNotFoundError("モデルファイルが見つかりません。")
else:
    size_kb = os.path.getsize(model_path) / 1024
    print(f"📦 モデルファイルサイズ: {size_kb:.2f} KB")
    if size_kb < 100:
        print("⚠️ モデルファイルが非常に小さいため、破損している可能性があります。")
        raise ValueError("モデルファイルが壊れている可能性があります。ダウンロードに失敗していませんか？")

try:
    model = load_model(model_path)
    print("✅ モデル読み込み成功")
except Exception as e:
    print(f"❌ モデル読み込み失敗: {e}")
    raise

labels = ['choco', 'classic', 'fruit']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(512, 512))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = labels[np.argmax(prediction)]

            return render_template("result.html", title="分類結果", result=result, image=filepath)

    return render_template("index.html", title="パンケーキ画像分類", message="画像をアップロードして分類してみよう!!")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
