


from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# ✅ モデル読み込み（事前に保存済みのモデル）
model = load_model("/Users/ishiinaoto/Desktop/model/BEST_MODEL_EARLY_CNN1_DENSE2_LR0.001.keras")
labels = ['choco', 'classic', 'fruit']  # モデルの出力順に応じたクラス名

# ✅ アップロード画像の保存先
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
    app.run(debug=True)
