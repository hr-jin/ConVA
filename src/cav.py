import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_reps(model, tokenizer, input_examples: list, layers: list):
    repres = {}

    for layer_name in layers:
        repres[layer_name] = []

    with torch.no_grad():
        for concept in input_examples:
            inputs   = tokenizer(concept, max_length=256, truncation=True, return_tensors="pt")
            outputs  = model(**inputs.to(model.device))

            for layer_name in layers:
                representation = model.representations[layer_name][0]
                if isinstance(representation, tuple):
                    repre    = representation[0][:, -1, :]
                else:
                    repre    = representation[:, -1, :]
                repres[layer_name].append(repre)

    for layer_name in layers:
        repres[layer_name] = torch.cat(repres[layer_name], dim =0).cpu().detach().numpy()

    return repres


def get_cavs(positive_reps, negative_reps, train_size=0.7, delta=None):
    positive_labels = np.ones ((len(positive_reps),))  
    negative_labels = np.zeros((len(negative_reps),))  

    X = np.vstack((positive_reps, negative_reps))
    Y = np.concatenate((positive_labels, negative_labels))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=train_size)
    if delta is None:
        log_reg = LogisticRegression(solver='saga', max_iter=10000)
        log_reg.fit(x_train, y_train)
    else:
        log_reg = LogisticRegression(penalty='l1', C=delta, solver='saga', max_iter=10000)
        log_reg.fit(x_train, y_train)

    predictions_test = log_reg.predict(x_test)
    predictions_train = log_reg.predict(x_train)
    acc_test = accuracy_score(y_test, predictions_test)
    acc_train = accuracy_score(y_train, predictions_train)
    cav = log_reg.coef_[0]
    
    return cav, log_reg, acc_train, acc_test


def get_cavs_by_contrast(model, tokenizer, positive_concept_examples, negative_concept_examples, train_size=0.7,
                          delta=0.5, num_runs=5, token_idx = -1):
    # get activation vector
    positive_activation_vectors = get_reps(model, tokenizer, positive_concept_examples,
                                           token_idx=token_idx)
    negative_activation_vectors = get_reps(model, tokenizer, negative_concept_examples, 
                                           token_idx=token_idx)

    positive_labels = np.ones ((len(positive_concept_examples),))  
    negative_labels = np.zeros((len(negative_concept_examples),))  

    X = np.vstack((positive_activation_vectors, negative_activation_vectors))
    Y = np.concatenate((positive_labels, negative_labels))

    cavs = []
    intercepts = []
    accuracy_train_list = []
    accuracy_test_list = []

    for i in range(num_runs):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size, random_state=i)
        if delta is None:
          log_reg = LogisticRegression(solver='saga', max_iter=10000)
        else:
          log_reg = LogisticRegression(penalty='l1', C=delta, solver='saga', max_iter=10000)
        # train
        log_reg.fit(x_train, y_train)

        # test
        predictions_test = log_reg.predict(x_test)
        predictions_train = log_reg.predict(x_train)
        # acc
        accuracy_test = accuracy_score(y_test, predictions_test)
        accuracy_train = accuracy_score(y_train, predictions_train)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        cav = log_reg.coef_[0]
        intercept = log_reg.intercept_
        intercepts.append(intercept)
        cavs.append(cav)
    
    acc = np.mean(accuracy_test_list)
    return cavs, acc, intercepts

def generate_(model, tokenizer, user_prompt, max_new_tokens=100, llama3=False, temperature=0.01):
    inputs           = tokenizer(user_prompt, return_tensors="pt")
    outputs          = model.generate(**inputs.to(model.device), temperature=temperature, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    origin_text    = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text    = origin_text[len(user_prompt):].strip()
    if llama3:
        return origin_text
    else:
        return response_text





