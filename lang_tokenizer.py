def custom_tokenizer(text_location='tiny-shakespeare.txt'):
    with open(text_location, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # store all unique characters in a set
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping of unique characters to integers
    stoi = { char:i for i,char in enumerate(chars) }
    itos = { i:char for i,char in enumerate(chars) }

    # encode and decode
    encode = lambda x: [stoi[char] for char in x]
    decode = lambda x: ''.join([itos[i] for i in x])

    return encode, decode, vocab_size

encode, decode, the_vocab_size = custom_tokenizer()