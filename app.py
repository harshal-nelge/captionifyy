from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import io



# from flask_cors import CORS

app =Flask(__name__)
# CORS(app)  # Enable CORS for all routes

@app.route('/')
def hello():
    return "hello"

# Load the model and tokenizer
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}




@app.route('/predict2', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'


    if request.method == 'POST':
        if 'file' not in request.files:
            return "File Not found"
        file = request.files.get('file')
        if not file:
            return "Something went Wrong"
        try:
            # prediction = predict_image(img)
            prediction =0
            print(prediction)
            contents = file.read()
            image = Image.open(io.BytesIO(contents))
            if image.mode != "RGB":
                image = image.convert(mode="RGB")

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            output_ids = model.generate(**inputs, **gen_kwargs)

            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            print(preds)


            # prediction = Markup(str(disease_dic[prediction]))
            return {
                "crop":preds
            }
        except Exception as e:
            print(e)
            return {
                "success":"false"
            }
    return "..."



# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     contents = file.read()
#     image = Image.open(io.BytesIO(contents))
#     if image.mode != "RGB":
#         image = image.convert(mode="RGB")

#     inputs = processor(images=image, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     output_ids = model.generate(**inputs, **gen_kwargs)

#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]

#     return jsonify({'predictions': preds})

if __name__ == '__main__':
    app.run(debug=True)
