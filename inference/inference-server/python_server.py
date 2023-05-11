# %% ---------------------------------------------
import sys
import torch
import onnxruntime
from transformers import AutoTokenizer, AutoModelForMaskedLM
from fastapi import FastAPI, Request
import logging
import numpy as np

# print(len(sys.argv))
if len(sys.argv) != 2:
    print("Usage : python",sys.argv[0],"<runtime>\n<runtime> options are 'pytorch', 'jit', 'onnx'")
    exit
runtime = sys.argv[1]
if runtime not in ['pytorch','jit','onnx']:
    print("Usage : python",sys.argv[0],"<runtime>\n<runtime> options are 'pytorch', 'jit', 'onnx'")
    exit
print("Runtime :",runtime)

model = None
app = FastAPI()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

if runtime == 'pytorch':
    model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")

elif runtime == 'jit':
    model = torch.jit.load('../model/large_lm.pt')

elif runtime == 'onnx':
    onnx_session = onnxruntime.InferenceSession("../model/bert_onnx.pt")

if model:
    model.eval()
    model.to(device)

logger = logging.getLogger('hypercorn.error')
app = FastAPI()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def pytorch_inference(tokens, idx_to_predict):
    global model
    tokens = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        output = model(tokens.to(device))[0].detach().cpu()
    output = torch.softmax(output[0][idx_to_predict], 0)
    prob, idx = output.topk(k=5, dim=0)
    result = {}
    result["Prediction"] = [int(i.numpy()) for i in idx]
    result["Confidence"] = [float(p.numpy()) for p in prob]
    return result

def onnx_inference(tokens, idx_to_predict):
    global onnx_session
    tokens = tokens + [0] * (512 - len(tokens))
    tokens = np.array(tokens).reshape(1, -1)

    ort_inputs = {onnx_session.get_inputs()[0].name: tokens}
    ort_outs = onnx_session.run(None, ort_inputs)
    output = np.array(ort_outs)[0][0]
    output = softmax(output[idx_to_predict])

    idx = np.argpartition(output, -5)[-5:]
    prob = output[idx]
    result = {}
    result["Prediction"] = [int(i) for i in idx]
    result["Confidence"] = [float(p) for p in prob]

    return result

@app.post('/predict')
async def predict(request: Request):
    global runtime

    result = (await request.json())
    request_id = result.get('request_id')
    tokens = result.get('tokens')
    idx_to_predict = result.get('idx_to_predict')

    logger.info(f'Request id: {request_id}')
    if runtime == 'onnx':
        out = onnx_inference(tokens, int(idx_to_predict))        
    else:
        out = pytorch_inference(tokens, int(idx_to_predict))
    return out    