from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Define your TextClassifier model
class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.fc(x)

# Load the model
model_path = "text_classifier.pth"
model_state_dict = torch.load(model_path)
model = TextClassifier()
model.load_state_dict(model_state_dict)
model.eval()

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define your preprocess_text function
def preprocess_text(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state
    embeddings = torch.mean(last_hidden_state, 1)  # Mean pooling
    return embeddings.squeeze().numpy()

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template('prediction.html')
    elif request.method == 'POST':
        text = request.form['news']
        
        # Preprocess the text data
        processed_text = preprocess_text(text)
        
        # Convert to tensor
        inputs = torch.tensor(processed_text, dtype=torch.float32).unsqueeze(0)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        # Convert predictions to text labels
        label = "TRUE" if preds.item() == 1 else "FALSE"
        
        # Return the prediction label
        return render_template('prediction.html', prediction_text=label)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
