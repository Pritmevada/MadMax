from sentence_transformers import SentenceTransformer, util

class TextAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def similarity(self, text1, text2):
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)

        return float(util.cos_sim(emb1, emb2))
