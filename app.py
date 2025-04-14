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
    print(f"Received sentence: {sentence}")  # In câu nhận được để kiểm tra
    # Tokenize câu
    model.eval()
    encode = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    
    input_ids = encode['input_ids']
    attention_mask = encode['attention_mask']
    
    device = torch.device("cpu")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    labels = ['negative', 'positive']
    return jsonify({'prediction': labels[predicted[0]]})  # Trả về kết quả dưới dạng JSON

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
