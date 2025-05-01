def generate_token_dict(vocal):
    """
    generate a dictionary of tokens from a list of words.

    input:
    - vocal(list): a list of words

    output:
    - token_dict(dict): a dictionary of tokens
    """
    # create a dictionary of tokens
    token_dict = {}
    for i, word in enumerate(vocal):
        token_dict[word] = i
    return token_dict


def preprocessing_data(data, token_dict, special_tokens):
    """
    preprocess the data.
    input:
    - data(str): a data string
    - token_dict(dict): a dictionary of tokens
    - special_tokens(list): a list of special tokens

    output:
    - data(list): a list of integers
    """
    output = []

    for item in data.split():
        # if the item is a special token, add it to the output
        if item in special_tokens:
            output.append(token_dict[item])
        else:
            # if the item is a number, split it into digits and add them to the output
            for digit in item:
                output.append(token_dict[digit])

    return output
