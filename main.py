import torch
import pickle

from PytorchReviewsAnalysis.tools import Predictor

MAXLEN = 200
MODEL_PATH = "ReviewModel.pt"
TOKENIZER_PATH = "tokenizer.pkl"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

sents_list = ["it was really good",
              "desipite flaws I think its still good",
              "it was not worth it's price",
              "For the price a cheek, the temple is much too small and has hardly any details and inside equipment, the figures are standard figures without great features.",
              "Thank you your music comforts me",
              "it's not that good",
              "I recently purchased the XYZ Wireless Earbuds, and I couldn't be happier with my purchase. The sound quality is outstanding, delivering crisp highs and deep bass that truly enhances my music and podcasts. They connect effortlessly to my devices, and the Bluetooth range is impressive. The battery life is also fantastic, lasting all day with ease. Comfort-wise, they fit snugly in my ears and don't fall out, even during workouts. The included charging case is compact and convenient. Overall, these earbuds have exceeded my expectations, and I highly recommend them to anyone in need of high-quality wireless earbuds.",
              "I bought the ABC Coffee Maker, and it has been nothing but a disappointment. Firstly, it takes ages to brew a pot of coffee, which is incredibly frustrating when you're in a hurry. The coffee it produces is lukewarm at best, and the flavor is bland and weak, no matter how much coffee I use. Cleaning this machine is also a hassle, with inaccessible areas that are hard to reach and prone to mold growth. The construction feels flimsy, and I worry it won't last long. To make matters worse, the customer service for this brand is virtually non-existent. Save your money and invest in a better coffee maker – this one is just not worth it."]

model = torch.jit.load(MODEL_PATH, map_location=torch.device('cpu'))  # when using cpu add map_location=torch.device('cpu')
model.eval()

predictor = Predictor()
predictions = predictor.make_predictions(model, sents_list, tokenizer, device=device, max_length=MAXLEN)
print(predictions.head(10))


