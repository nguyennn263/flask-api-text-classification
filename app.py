from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import torch
from Model import TextClassifier  # Giả sử bạn đã có lớp TextClassifier

app = Flask(__name__)

# Tải tokenizer từ Hugging Face
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Khởi tạo mô hình TextClassifier và tải trọng số
model = TextClassifier(vocable_size=30522, embedding_dim=128, num_classes=2)  # Tạo lại mô hình
model.load_state_dict(torch.load("TextClassifier_trained.pth"))  # Tải mô hình đã huấn luyện
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Lấy dữ liệu JSON từ yêu cầu
    sentence = data.get('text', '')  # Lấy văn bản cần phân loại

    # Tokenize câu
    encoding = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Đưa dữ liệu vào device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model.to(device)

    # Dự đoán với mô hình
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()  # Dự đoán lớp (positive/negative)

    # Các nhãn phân loại
    labels = ['negative', 'positive']
    prediction = labels[predicted_class]

    return jsonify({'prediction': prediction})  # Trả về kết quả dưới dạng JSON

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
