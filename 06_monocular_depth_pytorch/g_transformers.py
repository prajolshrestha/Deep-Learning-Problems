from transformers import pipeline

classifier = pipeline('sentiment-analysis')

res = classifier('I Love You')

print(res)