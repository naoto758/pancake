
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests

app = Flask(__name__)

# モデルのダウンロードと読み込み
file_id = "1UTgrarpDvqhYB-5zZVg1ckVowfkyUYIm"  # ← 自分のファイルID
url = f"https://drive.google.com/uc?export=download&id={file_id}"
model_path = "model.keras"

def download_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    # トークンが必要な場合の確認
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

if not os.path.exists(model_path):
    print("🔽 モデルをGoogle Driveからダウンロードします...")
    download_from_google_drive(file_id, model_path)

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

labels = ['choco', 'classic', 'fruit']  # モデルの出力順に応じたクラス名

# アップロード画像の保存先
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            # 画像保存
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # 画像前処理
            img = load_img(filepath, target_size=(512, 512))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 推論
            prediction = model.predict(img_array)
            result = labels[np.argmax(prediction)]

            return render_template("result.html", title="分類結果", result=result, image=filepath)

    return render_template("index.html", title="パンケーキ画像分類", message="画像をアップロードして分類してみよう!!")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # デフォルト10000（ローカル用）
    app.run(host="0.0.0.0", port=port)



