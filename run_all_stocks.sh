for news_type in content headlines
do
#  # non-emotion
  python run_stocks.py \
  --filepath datasets/stocks/candle_w_emotion/day_average_${news_type}/ \
  --news_type $news_type \
  --batch_size 32

  # emotion
  python run_stocks.py \
  --filepath datasets/stocks/candle_w_emotion/day_average_${news_type}/ \
  --news_type ${news_type}_emotion \
  --batch_size 32
done