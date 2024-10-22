batch_size=32

for news_type in content headlines
do
  # historical only
  python run_stocks.py \
  --filepath datasets/stocks/candle_w_emotion/day_average_${news_type}/ \
  --news_type ${news_type}_historical \
  --batch_size $batch_size

#  # base sentiment
#  python run_stocks.py \
#  --filepath datasets/stocks/candle_w_emotion/day_average_${news_type}/ \
#  --news_type ${news_type}_sentiment \
#  --batch_size $batch_size
#
#  # emotional data
#  python run_stocks.py \
#  --filepath datasets/stocks/candle_w_emotion/day_average_${news_type}/ \
#  --news_type ${news_type}_emotion \
#  --batch_size $batch_size
done

python calc_losses.py