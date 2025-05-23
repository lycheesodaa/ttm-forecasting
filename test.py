# code to extract the candles in the periods for content-only and headlines-only

from pathlib import Path
import pandas as pd
import os

# Content
# w_emotion = "datasets/stocks/candle_w_emotion/day_average_content/"
# candle_only = "datasets/stocks/candle_only/content/"
# cols = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
#
# Path(candle_only).mkdir(parents=True, exist_ok=True)
#
# files = os.listdir(w_emotion)
#
# for file in files:
#     df = pd.read_csv(os.path.join(w_emotion, file))
#     df = df.sort_values(by='date')
#     df.to_csv(os.path.join(w_emotion, file), index=False)
#
#     df = df[cols] # extract only the first 7 columns - date and OHCLVA
#     df.to_csv(os.path.join(candle_only, file), index=False)
#
# # Headlines
# w_emotion = "datasets/stocks/candle_w_emotion/day_average_headlines/"
# candle_only = "datasets/stocks/candle_only/headlines/"
#
# Path(candle_only).mkdir(parents=True, exist_ok=True)
#
# files = os.listdir(w_emotion)
#
# for file in files:
#     df = pd.read_csv(os.path.join(w_emotion, file))
#     df = df.sort_values(by='date')
#     df.to_csv(os.path.join(w_emotion, file), index=False)
#
#     df = df[cols] # extract only the first 7 columns - date and OHCLVA
#     df.to_csv(os.path.join(candle_only, file), index=False)


folders = ['stocks_content', 'stocks_content_emotion', 'stocks_headlines', 'stocks_headlines_emotion']
for folder in folders:
    files = os.listdir(Path('results') / folder)

    for file in files:
        df = pd.read_csv(Path('results') / folder / file)
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'] + pd.tseries.offsets.BusinessDay(1)
        df.to_csv(Path('results') / folder / file, index=False)

