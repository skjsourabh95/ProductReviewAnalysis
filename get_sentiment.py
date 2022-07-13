import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

np.random.seed(123456)


def predict_sentiment(sent, model, tokenizer):
    """Takes a review and returns the sentiment using pre-trained fine tuned BertClassifier"""
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=512,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_id = [encoded_dict['input_ids']]

    # And its attention mask (simply differentiates padding from non-padding).
    attention_mask = [encoded_dict['attention_mask']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert the lists into tensors.
    input_ids = torch.cat(input_id, dim=0).to(device)
    attention_masks = torch.cat(attention_mask, dim=0).to(device)

    # Copy the model to the GPU.
    model.to(device)

    model.eval()

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()

    prediction = np.argmax(logits, axis=1).flatten()

    x = Variable(torch.from_numpy(logits))

    probs = F.softmax(x, dim=1)

    sentiment = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }
    # final prediction for the review
    #     print(prediction[0])

    # calculate the sentiment and confidence scores
    for i, p in enumerate(probs[0]):
        if i == 0:
            sentiment["Negative"] = float("{:.3f}".format(p))
        if i == 1:
            sentiment["Neutral"] = float("{:.3f}".format(p))
        if i == 2:
            sentiment["Positive"] = float("{:.3f}".format(p))

    return sentiment
