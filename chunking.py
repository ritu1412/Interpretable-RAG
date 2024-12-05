import nltk
nltk.download('punkt')

def chunk_text_fixed_size(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def chunk_text_by_sentence(text, sentences_per_chunk=5):
    nltk.download('punkt_tab', quiet=True)
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]
    return chunks