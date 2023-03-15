from flask import Flask,render_template,request
import pickle
from transformers import GPT2Tokenizer
#import torch

app = Flask(__name__)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

#model = GPT2LMHeadModel.from_pretrained('gpt2')
#model.load_state_dict(torch.load("/kaggle/input/poem-pt/poem_model.pt"))
#torch.save(model,"poem_model.pt")
#model = torch.load('poem_model.pt')

file_path = 'poem_model.pkl'

# Save the model to file using pickle
#with open(file_path, 'wb') as f:
    #pickle.dump(model, f)

with open(file_path, 'rb') as f:
    model = pickle.load(f)



@app.route('/')
def home():
    return render_template('html.html')

@app.route('/infer', methods=['POST','GET'])
def infer():
    input_text = request.form['input-box']
    print(input_text)
    inp = tokenizer(input_text,return_tensors="pt") #must be a string
    X = inp["input_ids"] #.to(device)
    a = inp["attention_mask"] #.to(device)
    output = model.generate(X, 
                            attention_mask=a,
                            max_length=150,
                            early_stopping=True,
                            num_beams=2, 
                            no_repeat_ngram_size=2)
    
    output_text = tokenizer.decode(output[0])
    
    return render_template('html.html', pred=output_text)

#output = infer("I ask you not to cry for thy\n")

#print(output)

if __name__== '__main__':
    app.run(host='0.0.0.0', port='8080')
