import pandas as pd

df = pd.read_csv (r'./mtsamples.csv')

#unnamed / medical_specialty / sample_name / transcription / keywords

descr=df.description
spe=df.medical_specialty
name=df.sample_name
transcript=df.transcription
key=df.keywords
